"""Text2CT adapter — radiology report → 3D chest CT via MAISI latent rectified-flow diffusion.

Wraps ``third_party/text2ct`` (danielemolino/Text2CT, pinned 4fa286a) without modifying the
submodule (P2: ``third_party/`` is read-only). Text2CT is a MAISI-latent latent-diffusion
generator in the same family as Report2CT:

    report text ─► FrozenCLIP3D.encode_text ─► context (B, 1, 768)
                                                    │
    N(0,1) latent (1, 4, 128, 128, 32) ─► RFlowScheduler denoise loop (30 steps, CFG=5.0,
                                          v-prediction) over MAISI DiffusionModelUNetMaisi
                                                    │
                                          ReconModel + SlidingWindowInferer decode
                                                    │
                                          CT volume (512, 512, 128), HU[-1000,1000] int16

Pretrained weights (HuggingFace ``dmolino/text2ct-weights`` — see docs/text2ct_runbook.md):
    autoencoder_epoch273.pt             MAISI VAE        (built via ``autoencoder_def``)
    unet_rflow_200ep.pt                 UNet + scale_factor (stored in the checkpoint dict)
    CLIP3D_Finding_Impression_30ep.pt   FrozenCLIP3D text/vision encoder

Design — maximal reuse, minimal re-implementation (CLAUDE.md "reuse external code"):
  * The full text→CT sampler is upstream ``scripts.diff_model_demo.run_inference``, called
    verbatim. We do NOT re-implement the RFlow loop / CFG mix / decode / HU rescale — they
    carry subtle correctness. In particular ``run_inference`` (third_party/text2ct/
    scripts/diff_model_demo.py:114-116) gates the spacing tensor on the *unresolved* string
    ``"@include_body_region"`` (which is truthy), so the real spacing tensor is kept; the
    MAISI UNet has no ``None`` guard for ``spacing_layer``, so "cleaning up" that line to use
    the resolved boolean would crash. Verified by random-tensor I/O before wiring (UNet out
    (1,4,128,128,32) with the real spacing tensor; scheduler step shape-stable).
  * Models are built with upstream ``define_instance`` / ``load_models`` from the submodule's
    JSON configs (``autoencoder_def`` / ``diffusion_unet_def`` / ``noise_scheduler``), never
    re-declared.

MONAI version shim: Text2CT samples with ``RFlowScheduler``, added in MONAI 1.5; this repo is
pinned to MONAI 1.4. We vendor the single MONAI-1.5 scheduler file at
``src/baselines/_vendored/rectified_flow.py`` and (a) override the config ``noise_scheduler``
``_target_`` to that on-disk module (MONAI's ConfigParser locates classes from disk, not from
``sys.modules``) and (b) set the attribute ``monai.networks.schedulers.RFlowScheduler`` so the
``from monai.networks.schedulers import RFlowScheduler`` at diff_model_demo.py:18 and
``run_inference``'s ``isinstance(...)`` resolve to that same class.

Collision note: Text2CT imports its CTViT *relatively* as
``core.models.encoders.transformer_maskgit.*``, so — unlike ct_clip/generatect — it does NOT
register a top-level ``transformer_maskgit`` and cannot collide with them (verified). The only
generic top-level package names it uses are ``core`` and ``scripts``; we scrub stale copies of
those from ``sys.modules`` and prepend the text2ct root so its packages win.

CUDA-only: ``FrozenCLIP3D.__init__`` calls ``self.model.cuda()`` (clip.py:208) and the UNet
config sets ``use_flash_attention=True``, so a real build / inference requires a GPU. The
adapter still *constructs* (no build) on CPU so config-compose and CI smoke tests pass.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch
from lightning.pytorch import LightningModule

# ----------------------------------------------------------------------------
# Submodule + checkpoint locations.
# ----------------------------------------------------------------------------
_TEXT2CT_ROOT: Final[Path] = Path("/workspace/third_party/text2ct")
_CFG_DIR: Final[Path] = _TEXT2CT_ROOT / "configs"
_ENV_CONFIG: Final[Path] = _CFG_DIR / "environment_diff_model_eval.json"
_MODEL_CONFIG: Final[Path] = _CFG_DIR / "config_diff_model.json"
_MODEL_DEF: Final[Path] = _CFG_DIR / "config_rflow.json"

CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/text2ct")
AUTOENCODER_CKPT: Final[Path] = CKPT_DIR / "autoencoder_epoch273.pt"
UNET_CKPT: Final[Path] = CKPT_DIR / "unet_rflow_200ep.pt"
CLIP_CKPT: Final[Path] = CKPT_DIR / "CLIP3D_Finding_Impression_30ep.pt"

# Our vendored MONAI-1.5 scheduler (see module docstring). On-disk path so MONAI's
# ConfigParser can locate it; also injected into the monai namespace at build time.
_VENDORED_RFLOW_TARGET: Final[str] = (
    "src.baselines._vendored.rectified_flow.RFlowScheduler"
)

# Output contract — taken from config_diff_model.json `diffusion_unet_inference`.
OUTPUT_SIZE: Final[tuple[int, int, int]] = (512, 512, 128)
# Native voxel spacing (mm), same `diffusion_unet_inference.spacing` block. In the squeezed
# array's axis order (H, W, D) — matches how upstream `diff_model_demo.py:save_image` applies
# out_spacing[i] positionally, so building `diag(*spacing)` reproduces upstream byte-for-byte.
OUTPUT_SPACING: Final[tuple[float, float, float]] = (0.75, 0.75, 3.0)

log = logging.getLogger(__name__)


def _resolve_ckpt(ckpt_dir: Path, name: str) -> Path:
    """Locate a checkpoint under ``ckpt_dir``, tolerating the HF ``models/`` subfolder.

    ``huggingface_hub.snapshot_download("dmolino/text2ct-weights")`` lays the three ``.pt``
    files under ``<ckpt_dir>/models/`` (mirroring upstream's ``./models/`` convention). Accept
    either ``<ckpt_dir>/<name>`` or ``<ckpt_dir>/models/<name>``; fall back to the top-level
    path so a not-found error points at the canonical location.
    """
    direct = ckpt_dir / name
    if direct.is_file():
        return direct
    nested = ckpt_dir / "models" / name
    if nested.is_file():
        return nested
    return direct


def weights_present(ckpt_dir: Path = CKPT_DIR) -> bool:
    """True iff all three Text2CT checkpoints resolve (top-level or ``models/`` subfolder)."""
    return all(
        _resolve_ckpt(ckpt_dir, n.name).is_file()
        for n in (AUTOENCODER_CKPT, UNET_CKPT, CLIP_CKPT)
    )


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily change CWD.

    Required because Text2CT's ``model_cfg_bank`` (core/cfg_helper.py:106) reads
    ``configs/model/clip.yaml`` via a path relative to the current working directory.
    """
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _prioritize_text2ct_on_path() -> None:
    """Scrub stale ``core`` / ``scripts`` modules and prepend the text2ct root.

    Text2CT uses absolute imports rooted at its repo (``from core...`` / ``from
    scripts...``). ``core`` does not exist elsewhere in this repo, and ``/workspace/scripts``
    is a bare directory (no ``__init__.py``); prepending the text2ct root makes ``import
    scripts`` / ``import core`` resolve to Text2CT's packages. We also drop any cached copies
    so a stale resolution from another context cannot win. (Text2CT's ``transformer_maskgit``
    is imported *relatively* and never registers a top-level name, so unlike the
    ct_clip/generatect adapters there is no ``transformer_maskgit`` collision to manage.)
    """
    for _name in [
        m
        for m in list(sys.modules)
        if m in ("core", "scripts") or m.startswith("core.") or m.startswith("scripts.")
    ]:
        sys.modules.pop(_name, None)
    root = str(_TEXT2CT_ROOT)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _install_rflow_shim() -> None:
    """Make our vendored RFlowScheduler resolvable everywhere Text2CT expects MONAI 1.5's.

    ``diff_model_demo.py:18`` does ``from monai.networks.schedulers import RFlowScheduler``
    at import time, which would ``ImportError`` on MONAI 1.4. Setting the attribute on the
    already-imported package satisfies that import, and because ``run_inference`` does
    ``isinstance(noise_scheduler, RFlowScheduler)`` against the same imported name, the
    instance (built from the overridden ``_target_``, same class object) passes the check.
    """
    import monai.networks.schedulers as _ms  # noqa: PLC0415

    from src.baselines._vendored import rectified_flow as _rf  # noqa: PLC0415

    _ms.RFlowScheduler = _rf.RFlowScheduler
    # Also expose it as the submodule path so a plain import works if anything tries it.
    sys.modules.setdefault("monai.networks.schedulers.rectified_flow", _rf)


def _patch_clip_tokenizer_for_transformers_compat() -> None:
    """Swap Text2CT's vendored CLIPTokenizer/CLIPProcessor for the real ``transformers`` ones.

    Text2CT vendors a copy of HF's CLIP tokenizer (``core/models/encoders/clip_modules/
    tokenization_clip.py``) written for ``transformers==4.30``. Under our transformers 4.46 the
    base tokenizer ``__init__`` calls ``get_vocab()`` before the subclass sets ``self.encoder``,
    raising ``AttributeError: 'CLIPTokenizer' object has no attribute 'encoder'``. The vendored
    vocab/merges ARE the standard ``openai/clip-vit-large-patch14`` tokenizer, so the real
    ``transformers.CLIPTokenizer`` produces byte-identical token ids (verified: BOS=49406, ...).
    We keep the vendored ``CLIPConfig`` / ``CLIP3DModel`` (the architecture the checkpoint
    expects) and swap only the two tokenizer/processor utilities. ``third_party/`` is read-only
    (P2), so we patch the module attributes at runtime — same approach as the ctclip_adapter
    ``ContinuousPositionBias`` patch. ``FrozenCLIP3D.__init__`` references both names as module
    globals (clip.py:182-183), so patching the module object is sufficient.
    """
    import core.models.encoders.clip as _clipmod  # noqa: PLC0415

    from transformers import (  # noqa: PLC0415
        CLIPProcessor as _RealProc,
        CLIPTokenizer as _RealTok,
    )

    _clipmod.CLIPTokenizer = _RealTok
    _clipmod.CLIPProcessor = _RealProc


class Text2CTAdapter(LightningModule):
    """Inference-only wrapper around Text2CT's report→CT pipeline.

    Args:
        ckpt_dir: directory holding the 3 released checkpoints (see module docstring).
        device_str: where to build/run. Default ``"cpu"`` so the module can be *constructed*
            in CI / config-compose; a real build is CUDA-only (see module docstring) and the
            config sets ``cuda:0``.
        load_weights: if False, build the nets with random init (architecture sanity only) and
            skip the three ``load_state_dict`` calls. Default True.
        guidance_scale: classifier-free-guidance scale (upstream default 5.0).
        num_inference_steps: RFlow sampling steps (upstream default 30).
    """

    output_size: tuple[int, int, int] = OUTPUT_SIZE
    # Read by src/inference.py to write a truthful NIfTI affine (visualization path).
    output_spacing: tuple[float, float, float] = OUTPUT_SPACING

    def __init__(
        self,
        ckpt_dir: str | Path = CKPT_DIR,
        device_str: str = "cpu",
        load_weights: bool = True,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
    ) -> None:
        super().__init__()
        self._ckpt_dir = Path(ckpt_dir)
        self._device_str = device_str
        self._load_weights = load_weights
        self._guidance_scale = float(guidance_scale)
        self._num_inference_steps = int(num_inference_steps)

        # Built lazily by `_ensure_built`.
        self._args = None
        self._autoencoder: torch.nn.Module | None = None
        self._unet: torch.nn.Module | None = None
        self._clip: torch.nn.Module | None = None
        self._scale_factor: float | None = None
        self._run_inference = None
        self._prepared: tuple | None = None
        self._divisor: int | None = None

    # -- build ---------------------------------------------------------------

    def _ensure_built(self) -> None:
        if self._unet is not None:
            return

        device = torch.device(self._device_str)
        if self._device_str.startswith("cuda"):
            idx = int(self._device_str.split(":")[-1]) if ":" in self._device_str else 0
            torch.cuda.set_device(idx)

        _install_rflow_shim()
        _prioritize_text2ct_on_path()

        # Upstream helpers — imported here (not at module level) so this adapter file is
        # importable in CI without the submodule deps / GPU. diff_model_demo.py imports
        # RFlowScheduler at module load, hence the shim must be installed first.
        from scripts.diff_model_demo import (  # noqa: PLC0415
            load_models,
            prepare_tensors,
            run_inference,
        )
        from scripts.diff_model_setting import load_config  # noqa: PLC0415
        from scripts.utils import define_instance  # noqa: PLC0415

        args = load_config(str(_ENV_CONFIG), str(_MODEL_CONFIG), str(_MODEL_DEF))
        # Point the config at our checkpoints and our vendored scheduler; thread sampler knobs.
        ae_path = _resolve_ckpt(self._ckpt_dir, AUTOENCODER_CKPT.name)
        unet_path = _resolve_ckpt(self._ckpt_dir, UNET_CKPT.name)
        args.trained_autoencoder_path = str(ae_path)
        args.model_dir = str(unet_path.parent)
        args.model_filename = unet_path.name
        args.noise_scheduler["_target_"] = _VENDORED_RFLOW_TARGET
        args.use_cfg = True
        args.guidance_scale = self._guidance_scale
        args.diffusion_unet_inference["num_inference_steps"] = self._num_inference_steps

        # --- VAE + UNet (+ scale_factor) ---
        if self._load_weights:
            autoencoder, unet, scale_factor = load_models(args, device, log)
        else:
            autoencoder = define_instance(args, "autoencoder_def").to(device)
            unet = define_instance(args, "diffusion_unet_def").to(device)
            scale_factor = (
                1.0287  # diff_model_demo.py:63 fallback when no ckpt scale_factor
            )
        autoencoder.eval()
        unet.eval()

        # --- CLIP3D text encoder (cfg-bank build; relative-path cwd dependency) ---
        from core.cfg_helper import model_cfg_bank  # noqa: PLC0415
        from core.models.common.get_model import get_model  # noqa: PLC0415

        _patch_clip_tokenizer_for_transformers_compat()
        with _chdir(_TEXT2CT_ROOT):
            cfgm = model_cfg_bank()("clip_3D")
            clip = get_model()(cfgm).to(device)
        if self._load_weights:
            clip_ckpt = _resolve_ckpt(self._ckpt_dir, CLIP_CKPT.name)
            if not clip_ckpt.is_file():
                raise FileNotFoundError(
                    f"Text2CT CLIP3D checkpoint missing at {clip_ckpt}. "
                    f"See docs/text2ct_runbook.md for the download recipe."
                )
            clip.load_state_dict(
                torch.load(str(clip_ckpt), map_location=device), strict=True
            )
        clip.eval()

        # --- divisor + conditioning tensors (mirror diff_model_demo.py:245-256) ---
        num_channels = args.diffusion_unet_def["num_channels"]
        num_downsample_level = max(
            1,
            len(num_channels)
            if isinstance(num_channels, list)
            else len(args.diffusion_unet_def["attention_levels"]),
        )
        divisor = 2 ** (num_downsample_level - 2)
        prepared = prepare_tensors(args, device)  # (top, bottom, spacing, modality)

        self._args = args
        self._autoencoder = autoencoder
        self._unet = unet
        self._clip = clip
        self._scale_factor = scale_factor
        self._run_inference = run_inference
        self._prepared = prepared
        self._divisor = divisor

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def inference(self, prompt: str, **_) -> torch.Tensor:
        """Generate one CT volume from a report string.

        Calls upstream ``run_inference`` verbatim and wraps the returned int16 HU array
        (H, W, D) = (512, 512, 128) as a ``(1, 1, 512, 512, 128)`` tensor to match the
        ``model.inference(text, **cfg.sampler) -> (1,1,D,H,W)`` contract in src/inference.py.
        The ``**_`` absorbs the shared ``cfg.sampler`` keys other adapters consume.
        """
        self._ensure_built()
        top, bottom, spacing, modality = self._prepared
        arr = self._run_inference(
            self._args,
            torch.device(self._device_str),
            self._autoencoder,
            self._unet,
            self._clip,
            self._scale_factor,
            top,
            bottom,
            spacing,
            modality,
            self.output_size,
            self._divisor,
            log,
            impression_text=prompt,
        )
        vol = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int16)
        return vol.unsqueeze(0).unsqueeze(0)
