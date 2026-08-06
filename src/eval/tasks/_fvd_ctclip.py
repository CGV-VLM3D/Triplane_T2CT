"""Local FVD via a CT-CLIP zero-shot 18-abnormality profile (feature-extractor swap).

The VLM3D-docker FVD harness (``evaluate_fvd.py``) turns each volume into a **96-d**
CTNet feature, groups volumes by ``CHUNK=4``, computes a per-group Frechet distance, and
averages. (96, not 18: ``CTNetModel.forward`` returns the penultimate ReLU/dropout output —
upstream comments out the final ``classifier[6]`` 96→18 layer, so the disease logits are
never produced. Verified by a forward pass; ``FVD/fvd_pytorch_model.py:13`` also says ``d=96``.)
The shuttled CTNet checkpoint is a corrupt stub, so we swap **only** the feature extractor:
CT-CLIP zero-shot over the 18 CT-RATE abnormalities yields an 18-d ``P(present)`` vector per
volume. 18 is CT-CLIP's natural zero-shot width, **not** a match to CTNet's 96; it is chosen
over the 512-d contrastive latent on measured stability (same-distribution self-FVD at N=652:
d=18 → 0.28, d=512 → 204), and being lower-dimensional than CTNet it is *more* sample-stable
than the official metric rather than equally so. Absolute values live in a different space
from ``FVD_CTNet`` either way (bounded probabilities vs unbounded post-ReLU activations) —
this is a cross-model ranking signal, never a leaderboard-comparable number.

Faithfulness to the docker eval is maximised by **importing the upstream code unchanged**
wherever it imports cleanly in-process:
  * the Frechet distance  → ``FVD.fvd_pytorch_model.compute_fvd`` (imported),
  * pairing + chunk size  → ``evaluate_fvd`` ``_list_volumes`` / ``map_to_gt`` / ``CHUNK`` (imported).
Only the CT-CLIP preprocessing (``evaluate_clip._load_vol`` — that module can't be imported
in-process, its top-level ``from ct_clip import CTCLIP`` needs the adapter's sys.path setup)
is copied verbatim, and only the feature extractor is swapped CTNet → CT-CLIP (this module).

``CTCLIPZeroShotProfiler`` reproduces CT-CLIP's zero-shot classifier exactly
(third_party/ct_clip/scripts/zero_shot.py): per pathology, encode ``"{p} is present."`` /
``"{p} is not present."``, score each against the image latent as ``(img·txt) * temperature``,
softmax the pair, keep ``P(present)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from src.baselines.ctclip_adapter import CTCLIPBackbone
from src.eval._vlm3d_paths import ctgen_eval_dir

log = logging.getLogger(__name__)

# DUPLICATION INTENTIONAL: the 18 CT-RATE abnormality labels, verbatim from
# third_party/ct_clip/scripts/zero_shot.py:124 (also text_classifier/infer.py:80).
# Order is load-bearing — it fixes the 18 feature axes across all model evals.
PATHOLOGIES: tuple[str, ...] = (
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)


class CTCLIPZeroShotProfiler:
    """Maps a preprocessed CT volume to CT-CLIP's 18-d zero-shot disease profile.

    The 36 prompt latents (18 pathologies × {present, absent}) and the temperature
    are volume-independent, so they are encoded once and cached on first use.
    """

    def __init__(self, device_str: str = "cuda:0") -> None:
        self._bb = CTCLIPBackbone(device_str=device_str)
        self._present: torch.Tensor | None = None  # (18, 512) L2 text latents
        self._absent: torch.Tensor | None = None  # (18, 512) L2 text latents
        self._temp: torch.Tensor | None = None  # scalar CT-CLIP temperature

    def _ensure_prompts(self) -> None:
        """Encode + cache the 18×2 prompt latents and the CT-CLIP temperature."""
        if self._present is not None:
            return
        pos = self._bb.tokenize([f"{p} is present." for p in PATHOLOGIES])
        neg = self._bb.tokenize([f"{p} is not present." for p in PATHOLOGIES])
        self._present = self._bb.encode_text(
            pos["input_ids"], pos["attention_mask"]
        )  # (18, 512)
        self._absent = self._bb.encode_text(
            neg["input_ids"], neg["attention_mask"]
        )  # (18, 512)
        # temp = self.temperature.exp()  (ct_clip.py:796); extra_latent_projection=False
        # ⇒ encode_image/encode_text latents are exactly CTCLIP.forward's latents.
        self._temp = self._bb.clip.temperature.exp().detach()  # scalar

    @torch.no_grad()
    def profile(self, vol: torch.Tensor) -> torch.Tensor:
        """CT volume → 18-d ``P(present)`` disease profile.

        Args:
            vol: preprocessed CT volume ``(1, 1, D, 480, 480)``, HU/1000 in ``[-1, 1]``,
                ``D % 10 == 0`` (the ``encode_image`` contract).

        Returns:
            ``(18,)`` float32 CPU tensor; entry i = ``P(pathology_i present)`` ∈ [0, 1],
            ordered as ``PATHOLOGIES``.
        """
        self._ensure_prompts()
        img = self._bb.encode_image(vol)  # (1, 512) L2, on CT-CLIP device
        img = img[0]  # (512,)
        # per-pathology similarity logits, scaled by CT-CLIP temperature
        sim_pos = (self._present @ img) * self._temp  # (18,)
        sim_neg = (self._absent @ img) * self._temp  # (18,)
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # (18, 2): [present, absent]
        prob_present = torch.softmax(logits, dim=1)[:, 0]  # (18,) softmax over the pair
        return prob_present.float().cpu()


# ---------------------------------------------------------------------------- #
#  Unit 2 — faithful docker FVD harness with CT-CLIP features                   #
# ---------------------------------------------------------------------------- #

# CT-CLIP preprocessing, DUPLICATION INTENTIONAL — copied verbatim from
# third_party/.../ctgen_evaluation/evaluate_clip.py:40-91. That module cannot be
# imported in-process (its top-level `from ct_clip import CTCLIP` needs the CT-CLIP
# adapter's sys.path juggling), so the four pure-SimpleITK/torch helpers are relocated
# here unchanged. Using the CLIPScore preprocessing keeps FVD_CTCLIP's CT-CLIP input
# identical to how CT-CLIP is fed for CLIPScore in the same eval.
TARGET_SPACING: tuple[float, float, float] = (1.5, 0.75, 0.75)  # (z, x, y)
TARGET_DHW: tuple[int, int, int] = (240, 480, 480)  # (D, H, W)
HU_MIN, HU_MAX = -1000.0, 1000.0
PAD_VALUE = -1.0


def _read_mha(p: Path) -> np.ndarray:
    """MetaImage → float32 ``(D, H, W)`` ndarray (evaluate_clip.py:47-50)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)  # (Z, Y, X)
    return arr.transpose(0, 2, 1)  # (D, H, W)


def _resize_array(
    vol: torch.Tensor, cur_spacing: tuple[float, float, float]
) -> torch.Tensor:
    """Resample ``(D,H,W)`` to TARGET_SPACING, trilinear (evaluate_clip.py:52-63)."""
    d, h, w = vol.shape
    scale = [cur_spacing[i] / TARGET_SPACING[i] for i in range(3)]
    new_size = [int(round(dim * sc)) for dim, sc in zip((d, h, w), scale)]
    return F.interpolate(
        vol.unsqueeze(0).unsqueeze(0),
        size=new_size,
        mode="trilinear",
        align_corners=False,
    )[0, 0]  # (D', H', W')


def _centre_crop_or_pad(v: torch.Tensor) -> torch.Tensor:
    """Centre-crop / pad ``(D,H,W)`` to TARGET_DHW (evaluate_clip.py:65-74)."""
    D, H, W = TARGET_DHW
    pad = [
        (W - v.shape[2]) // 2,
        W - v.shape[2] - (W - v.shape[2]) // 2,
        (H - v.shape[1]) // 2,
        H - v.shape[1] - (H - v.shape[1]) // 2,
        (D - v.shape[0]) // 2,
        D - v.shape[0] - (D - v.shape[0]) // 2,
    ]
    v = F.pad(v, pad, value=PAD_VALUE)  # (W1,W2,H1,H2,D1,D2)
    return v[:D, :H, :W]


def _load_vol(p: Path) -> torch.Tensor:
    """Read, HU-clip, resample, crop/pad a volume → ``(240,480,480)`` (evaluate_clip.py:76-91)."""
    arr = _read_mha(p)  # (D, H, W)
    arr = np.clip(arr, HU_MIN, HU_MAX)
    itk_img = sitk.ReadImage(str(p))
    sx, sy, sz = itk_img.GetSpacing()  # (x, y, z)
    v = torch.from_numpy(arr).permute(0, 1, 2)  # (D, H, W)
    v = _resize_array(v, (sz, sy, sx))  # spacing order → (z, x, y)
    v = _centre_crop_or_pad(v)  # (240, 480, 480)
    return v


_UPSTREAM: tuple | None = None


def _load_upstream() -> tuple:
    """Import + cache the unchanged upstream FVD pieces (distance, pairing, CHUNK).

    Returns:
        ``(compute_fvd, evaluate_fvd)`` — the pure-numpy Frechet distance and the
        docker harness module (for ``_list_volumes`` / ``map_to_gt`` / ``CHUNK``).
    """
    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM
    eval_dir = ctgen_eval_dir()
    for p in (str(eval_dir), str(eval_dir / "FVD")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from FVD.fvd_pytorch_model import compute_fvd  # noqa: PLC0415  (numpy/scipy only)
    import evaluate_fvd  # noqa: PLC0415  (pairing + CHUNK; pulls ctnet at import, not the ckpt)

    _UPSTREAM = (compute_fvd, evaluate_fvd)
    return _UPSTREAM


def compute_fvd_ctclip(
    gen_dir: str | Path, gt_dir: str | Path, device_str: str = "cuda:0"
) -> float:
    """Docker FVD harness with CT-CLIP zero-shot features (feature-extractor swap).

    Mirrors ``evaluate_fvd.py::main`` 1:1 — same file ordering (``_list_volumes``),
    same GT pairing (``map_to_gt``), same ``CHUNK`` grouping and per-group Frechet +
    mean — but each volume is CT-CLIP-preprocessed (``_load_vol`` → ``/1000``) and turned
    into an 18-d profile instead of a CTNet forward. Distance is the unchanged upstream
    ``compute_fvd``. All of pairing / CHUNK / distance are imported from upstream.

    Args:
        gen_dir: directory of generated ``*.nii.gz`` / ``*.mha`` volumes.
        gt_dir: directory of ground-truth volumes (matched by case stem).

    Returns:
        Mean per-group ``FVD_CTCLIP`` (float), or NaN if no generated volumes.
    """
    compute_fvd, efvd = _load_upstream()
    profiler = CTCLIPZeroShotProfiler(device_str=device_str)

    gen_files = efvd._list_volumes(Path(gen_dir))
    if not gen_files:
        return float("nan")
    # A trailing group smaller than CHUNK is a singular (rank-deficient) covariance
    # matrix, which can crash scipy.linalg.sqrtm inside upstream compute_fvd (confirmed
    # for group size 1: rank-0 covariance -> LinAlgError -102). Drop the minimum number
    # of volumes off the end so the count is an exact multiple of CHUNK, rather than
    # silently NaN-ing the whole metric — this never mattered before n=253 came up
    # (every prior run used n=100 or n=300, both multiples of 4).
    remainder = len(gen_files) % efvd.CHUNK
    if remainder != 0:
        dropped = gen_files[-remainder:]
        log.warning(
            "FVD_CTCLIP: %d generated volumes is not a multiple of CHUNK=%d — the "
            "trailing group of %d has a singular covariance matrix. Dropping the last "
            "%d volume(s) (%s) to make the count a multiple of %d.",
            len(gen_files),
            efvd.CHUNK,
            remainder,
            remainder,
            [f.name for f in dropped],
            efvd.CHUNK,
        )
        gen_files = gen_files[:-remainder]
    gt_root = Path(gt_dir)

    gb: list[np.ndarray] = []  # generated 18-d profiles in the current chunk
    rb: list[np.ndarray] = []  # matched GT 18-d profiles
    scores: list[float] = []
    for g in gen_files:
        r = efvd.map_to_gt(g, gt_root)
        # reshape (not view): a crop returns a non-contiguous slice for volumes taller
        # than 240 slices (the common valid_v2 case).
        vg = (_load_vol(g) / 1000.0).reshape(1, 1, *TARGET_DHW)  # HU/1000 ∈ [-1,1]
        vr = (_load_vol(r) / 1000.0).reshape(1, 1, *TARGET_DHW)
        gb.append(profiler.profile(vg).numpy())  # (18,)
        rb.append(profiler.profile(vr).numpy())  # (18,)
        if len(gb) == efvd.CHUNK:  # CHUNK imported from upstream (=4)
            # arg order matches upstream `fvd.compute_fvd(r_np, g_np)` → (GT, gen);
            # Frechet distance is symmetric, so this is a faithfulness choice only.
            scores.append(compute_fvd(np.stack(rb), np.stack(gb)))  # (4,18) each
            gb, rb = [], []
    if gb:  # trailing < CHUNK group (upstream does the same)
        scores.append(compute_fvd(np.stack(rb), np.stack(gb)))
    # observability: distinguishes "all paired + computed" from a silent short/empty run
    log.info(
        "FVD_CTCLIP: %d generated volumes → %d chunk(s) (CHUNK=%d)",
        len(gen_files),
        len(scores),
        efvd.CHUNK,
    )
    return float(np.mean(scores)) if scores else float("nan")
