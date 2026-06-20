"""GenerateCT LightningModule adapter (text → 3D CT volume).

Wraps `third_party/generatect/transformer_maskgit/*` and the 3 pretrained
checkpoints (`ctvit_pretrained.pt`, `transformer_pretrained.pt`,
`superres_pretrained.pt`) without modifying the submodule itself
(P2: third_party/ is read-only).

Pipeline (paper-grounded — see GenerateCT ECCV 2024):
    text prompt  ──► MaskGITTransformer (text-conditioned token AR)
                           │
                           ▼
                       CTViT decoder
                           │
                           ▼
                  low-res volume (128 × 128 × 201)
                           │
                           ▼  (optional, Phase B)
            super-resolution diffusion → 512 × 512 × 201

For Phase A Day 3 we only wire the low-res pipeline (CTViT + MaskGIT).
Super-resolution is loaded lazily on demand (Phase B B.3 diagnostic time).

The submodule isn't `pip install`-ed; we add it to sys.path so its
package `transformer_maskgit` is importable. See
docs/report2ct_external_components.md for why.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Final

import torch
from lightning.pytorch import LightningModule

# Both ct_clip and generatect ship a top-level `transformer_maskgit` package
# with DIFFERENT `CTViT.forward` signatures (ct_clip's accepts
# `return_encoded_tokens=True`, this one does not). To avoid sys.modules
# poisoning across adapters, we do NOT mutate sys.path at module import;
# `_import_transformer_maskgit` scrubs + re-prioritizes at build time.
_GENERATECT_DIR: Final[Path] = Path("/workspace/third_party/generatect")
_TM_PKG_DIR: Final[Path] = _GENERATECT_DIR / "transformer_maskgit"
_CTCLIP_TM_PKG_DIR: Final[Path] = Path(
    "/workspace/third_party/ct_clip/transformer_maskgit"
)


# Pretrained checkpoint locations (downloaded Day 3 from huggingface.co/generatect/GenerateCT).
CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/generatect")
CTVIT_CKPT: Final[Path] = CKPT_DIR / "ctvit_pretrained.pt"
TRANSFORMER_CKPT: Final[Path] = CKPT_DIR / "transformer_pretrained.pt"
SUPERRES_CKPT: Final[Path] = CKPT_DIR / "superres_pretrained.pt"

# Upstream super-res config yaml (read in full via OmegaConf — no kwarg duplication needed).
SUPERRES_YAML: Final[Path] = _GENERATECT_DIR / "superres_inference.yaml"


# DUPLICATION INTENTIONAL — GenerateCT submodule encodes its model spec as Python literals
# inside `inference_ctvit.py` / `inference_transformer.py` (no YAML/JSON config file). We copy
# the literals here verbatim. If the submodule ever ships a config file, switch to ConfigParser
# like src/baselines/maisi.py (or the YAML-driven instantiate in configs/model/report2ct.yaml).
# Source: third_party/generatect/inference_ctvit.py:5-15, inference_transformer.py:44-54
_CTVIT_KWARGS: Final[dict] = {
    "dim": 512,
    "codebook_size": 8192,
    "image_size": 128,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_depth": 4,
    "temporal_depth": 4,
    "dim_head": 32,
    "heads": 8,
}

# DUPLICATION INTENTIONAL — see _CTVIT_KWARGS note above.
# Source: third_party/generatect/inference_transformer.py:56-62
_MASKGIT_KWARGS: Final[dict] = {
    "num_tokens": 8192,
    "max_seq_len": 10000,
    "dim": 512,
    "dim_context": 768,
    "depth": 6,
}


def _import_transformer_maskgit():
    """Defer imports until first use so we don't crash module import when deps are missing.

    Also scrubs any cached `transformer_maskgit` modules from sys.modules and
    re-prioritizes our path so a prior CTCLIPBackbone build in the same
    process doesn't poison our import with ct_clip's incompatible CTViT.
    """
    # See module-level comment for the cross-adapter collision rationale.
    for _name in [
        m
        for m in sys.modules
        if m == "transformer_maskgit" or m.startswith("transformer_maskgit.")
    ]:
        sys.modules.pop(_name, None)
    _ctclip_tm_str = str(_CTCLIP_TM_PKG_DIR)
    _ctclip_tm_had_path = _ctclip_tm_str in sys.path
    while _ctclip_tm_str in sys.path:
        sys.path.remove(_ctclip_tm_str)
    while str(_TM_PKG_DIR) in sys.path:
        sys.path.remove(str(_TM_PKG_DIR))
    sys.path.insert(0, str(_TM_PKG_DIR))

    try:
        from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer  # noqa: PLC0415
        from transformer_maskgit.attention import ContinuousPositionBias  # noqa: PLC0415
        from einops import rearrange  # noqa: PLC0415
    finally:
        # Restore ct_clip's path so a later CTCLIPBackbone build can scrub-and-reimport.
        if _ctclip_tm_had_path and _ctclip_tm_str not in sys.path:
            sys.path.append(_ctclip_tm_str)

    # Monkey-patch: attention.py:260 hardcodes `device=torch.device('cuda')`, overriding the
    # `device` arg that MaskGit.forward correctly passes as `x.device`. This causes a device
    # mismatch when the model is on any GPU other than cuda:0.
    # third_party/ is READ-ONLY (P2), so we patch the class method here instead.
    # Source being fixed: third_party/generatect/transformer_maskgit/transformer_maskgit/attention.py:257-276
    def _cpb_forward_patched(self, *dimensions, device=torch.device("cpu")):
        if self.rel_pos is None or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer("rel_pos", rel_pos, persistent=False)
        rel_pos = self.rel_pos.to(torch.float32)
        for layer in self.net:
            rel_pos = layer(rel_pos.float())
        return rearrange(rel_pos, "i j h -> h i j")

    ContinuousPositionBias.forward = _cpb_forward_patched

    return CTViT, MaskGit, MaskGITTransformer


def build_ctvit(
    load_weights: bool = True, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Construct CTViT with GenerateCT paper kwargs; optionally load pretrained weights."""
    CTViT, _, _ = _import_transformer_maskgit()
    ctvit = CTViT(**_CTVIT_KWARGS)
    if load_weights:
        if not CTVIT_CKPT.is_file():
            raise FileNotFoundError(f"CTViT checkpoint missing at {CTVIT_CKPT}")
        ctvit.load(str(CTVIT_CKPT))
    ctvit.eval()
    return ctvit.to(device)


def build_text_to_volume_model(
    load_weights: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Construct the full text → low-res-volume model (CTViT + MaskGITTransformer)."""
    CTViT, MaskGit, MaskGITTransformer = _import_transformer_maskgit()

    ctvit = CTViT(**_CTVIT_KWARGS)
    if load_weights:
        if not CTVIT_CKPT.is_file():
            raise FileNotFoundError(f"CTViT checkpoint missing at {CTVIT_CKPT}")
        ctvit.load(str(CTVIT_CKPT))
    ctvit.eval()

    maskgit = MaskGit(**_MASKGIT_KWARGS)
    model = MaskGITTransformer(ctvit=ctvit, maskgit=maskgit)
    if load_weights:
        if not TRANSFORMER_CKPT.is_file():
            raise FileNotFoundError(
                f"Transformer checkpoint missing at {TRANSFORMER_CKPT}"
            )
        model.load(str(TRANSFORMER_CKPT))
    model.eval()
    return model.to(device)


def build_super_resolution(device: str | torch.device = "cuda:0"):
    """Construct SuperResolutionTrainer with pretrained weights.

    Mirrors third_party/generatect/inference_superres.py:106-141 but skips
    the file-based VideoTextDataset — caller supplies the low-res tensor
    via SuperResolutionTrainer.sample(start_image_or_video=...).
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    # `super_resolution` is a setuptools package: outer dir is the source root,
    # inner dir is the importable package. We need the outer dir on sys.path.
    _sr_pkg_root = _GENERATECT_DIR / "super_resolution"
    if str(_sr_pkg_root) not in sys.path:
        sys.path.insert(0, str(_sr_pkg_root))
    from super_resolution import (  # noqa: PLC0415
        ElucidatedSuperres,
        NullUnet,
        SuperResolutionTrainer,
        Superres,
        Unet,
    )

    if not SUPERRES_CKPT.is_file():
        raise FileNotFoundError(f"Super-res checkpoint missing at {SUPERRES_CKPT}")
    if not SUPERRES_YAML.is_file():
        raise FileNotFoundError(f"Super-res config missing at {SUPERRES_YAML}")

    cfg = OmegaConf.load(str(SUPERRES_YAML))
    unet1 = NullUnet()
    # Pass v as OmegaConf DictConfig (no to_container) to match upstream
    # inference_superres.py:108 exactly. The YAML has dim_mults (3 items) vs
    # layer_cross_attns (4 items); cast_tuple() in superres_pytorch.py:70 silently
    # wraps the whole ListConfig as a scalar when isinstance(val,list) is False,
    # so all layers end up with cross-attention — this is what the checkpoint was
    # trained with and must be reproduced faithfully.
    unet2 = [Unet(**v, lowres_cond=(i > 0)) for i, v in enumerate(cfg.unets.values())]
    klass = ElucidatedSuperres if cfg.superres.get("elucidated", False) else Superres
    sr = klass(
        unets=(unet1, unet2[0]),
        **OmegaConf.to_container(cfg.superres.params),
    )
    infer = SuperResolutionTrainer(
        superres=sr, **OmegaConf.to_container(cfg.trainer.params)
    ).to(device)
    infer.load(str(SUPERRES_CKPT))
    return infer


class GenerateCTAdapter(LightningModule):
    """LightningModule wrapper around GenerateCT's text-to-volume pipeline.

    Used for Phase A Day 3 + 5 smoke (low-res only) and Phase B B.3 diagnostics
    (cross-attention / token-region hooks on the transformer + CTViT).
    Inference-only — no training step.

    Args:
        device_str: cpu or cuda. We default to CPU at construction so the
            module can be imported into a unit test without GPU.
        load_super_resolution: also load the super-res diffusion model
            (defers to Phase B if False).
    """

    def __init__(
        self,
        device_str: str = "cpu",
        load_super_resolution: bool = False,
    ) -> None:
        super().__init__()
        self._device_str = device_str
        self._load_super_resolution = load_super_resolution
        self._model: torch.nn.Module | None = None
        self._sr_infer: object | None = None

    @property
    def output_spacing(self) -> tuple[float, float, float]:
        """Native voxel spacing (mm) in the SQUEEZED-ARRAY axis order (D, H, W) = (z, y, x).

        GenerateCT trains on CT-RATE resampled to (x, y, z) = (0.75, 0.75, 1.5) mm at 512²
        in-plane (third_party/vlm3d_dockers/.../data_preprocess/preprocess_ctrate_train.py:93-95).
        `.inference()` returns (1, 1, D, H, W); after squeeze the axes are (D, H, W), so the
        spacing is (1.5, 0.75, 0.75). The low-res CTViT path covers the same FOV at 128² →
        in-plane 0.75 × 512/128 = 3.0 mm.

        Read ONLY by src/inference.py (visualization). NOTE the eval sampler
        (src/eval/samplers/generatect.py) stamps the same physical truth but in ITK (x, y, z)
        order — do not "reconcile" the two; they encode identical geometry in different axis
        conventions (CLAUDE.md audit checklist #4).
        """
        in_plane = 0.75 if self._load_super_resolution else 3.0
        return (1.5, in_plane, in_plane)

    # Lazy build so an instance can be created without ckpts present (e.g. in tests).
    def _ensure_built(self) -> None:
        # third_party t5.py uses bare `.cuda()` and attention.py uses `torch.device('cuda')`,
        # both of which resolve to the current default device — pin it here so they land on
        # the correct GPU instead of always defaulting to cuda:0.
        if self._device_str.startswith("cuda"):
            idx = int(self._device_str.split(":")[-1]) if ":" in self._device_str else 0
            torch.cuda.set_device(idx)
        if self._model is None:
            self._model = build_text_to_volume_model(
                load_weights=True, device=self._device_str
            )
        if self._load_super_resolution and self._sr_infer is None:
            self._sr_infer = build_super_resolution(device=self._device_str)

    @torch.no_grad()
    def text_to_volume(
        self,
        prompt: str | list[str],
        num_frames: int = 201,
        cond_scale: float = 5.0,
        starting_temperature: float = 0.9,
    ) -> torch.Tensor:
        """Generate a single low-res CT volume from a text prompt.

        Defaults match third_party/generatect/inference_transformer.py:84.

        Returns:
            Tensor of shape `(B, C=1, D, H, W)` from the MaskGITTransformer
            pipeline. For `num_frames=201` the typical shape is
            `(1, 1, 201, 128, 128)`. Caller may apply super-resolution separately.
        """
        self._ensure_built()
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        with redirect_stdout(io.StringIO()):
            return self._model.sample(
                texts=texts,
                num_frames=num_frames,
                cond_scale=cond_scale,
                starting_temperature=starting_temperature,
            )

    @torch.no_grad()
    def text_to_volume_hires(
        self,
        prompt: str,
        num_frames: int = 201,
        cond_scale_lowres: float = 5.0,
        cond_scale_sr: float = 1.0,
    ) -> torch.Tensor:
        """Full text → 512×512 pipeline (low-res CTViT → per-frame super-res).

        Mirrors third_party/generatect/inference_superres.py:172-185 — slice the
        low-res volume along its depth axis and run SR per 2D frame, then stack.
        """
        assert self._load_super_resolution, (
            "Construct GenerateCTAdapter(load_super_resolution=True) first."
        )
        self._ensure_built()
        lowres = self.text_to_volume(
            prompt, num_frames=num_frames, cond_scale=cond_scale_lowres
        )
        # (1, 1, D, 128, 128) -> (D, 1, 128, 128)
        frames = lowres.squeeze(0).permute(1, 0, 2, 3)
        out_frames = []
        for k in range(frames.shape[0]):
            with redirect_stdout(io.StringIO()):
                sr_img = (
                    self._sr_infer.sample(
                        cond_scale=cond_scale_sr,
                        texts=[prompt],
                        start_image_or_video=frames[k : k + 1],
                        start_at_unet_number=2,
                    )
                    .detach()
                    .cpu()
                )
            out_frames.append(sr_img[0])  # (1, 512, 512)
        # (D, 1, 512, 512) -> (1, 1, D, 512, 512)
        return torch.stack(out_frames, dim=1).unsqueeze(0)

    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        num_frames: int = 201,
        cond_scale: float = 5.0,
        starting_temperature: float = 0.9,
        cond_scale_sr: float = 1.0,
        **_,  # absorb keys other adapters care about (shared cfg.sampler contract)
    ) -> torch.Tensor:
        """Generic hook called by src/inference.py — dispatches based on SR flag."""
        if self._load_super_resolution:
            return self.text_to_volume_hires(
                prompt,
                num_frames=num_frames,
                cond_scale_lowres=cond_scale,
                cond_scale_sr=cond_scale_sr,
            )
        return self.text_to_volume(
            prompt,
            num_frames=num_frames,
            cond_scale=cond_scale,
            starting_temperature=starting_temperature,
        )
