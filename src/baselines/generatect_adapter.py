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

import sys
from pathlib import Path
from typing import Final

import torch
from lightning.pytorch import LightningModule

# Add submodule to import path before importing transformer_maskgit.
_GENERATECT_DIR: Final[Path] = Path("/workspace/third_party/generatect")
_TM_PKG_DIR: Final[Path] = _GENERATECT_DIR / "transformer_maskgit"
if str(_TM_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_TM_PKG_DIR))


# Pretrained checkpoint locations (downloaded Day 3 from huggingface.co/generatect/GenerateCT).
CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/generatect")
CTVIT_CKPT: Final[Path] = CKPT_DIR / "ctvit_pretrained.pt"
TRANSFORMER_CKPT: Final[Path] = CKPT_DIR / "transformer_pretrained.pt"
SUPERRES_CKPT: Final[Path] = CKPT_DIR / "superres_pretrained.pt"


# CTViT init kwargs from GenerateCT/inference_ctvit.py + inference_transformer.py.
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

# MaskGit init kwargs from GenerateCT/inference_transformer.py.
_MASKGIT_KWARGS: Final[dict] = {
    "num_tokens": 8192,
    "max_seq_len": 10000,
    "dim": 512,
    "dim_context": 768,
    "depth": 6,
}


def _import_transformer_maskgit():
    """Defer imports until first use so we don't crash module import when deps are missing."""
    from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer  # noqa: PLC0415

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

    # Lazy build so an instance can be created without ckpts present (e.g. in tests).
    def _ensure_built(self) -> None:
        if self._model is None:
            self._model = build_text_to_volume_model(
                load_weights=True, device=self._device_str
            )
            if self._load_super_resolution:
                raise NotImplementedError(
                    "Super-resolution loading lands in Phase B; toggle off for Phase A smoke."
                )

    @torch.no_grad()
    def text_to_volume(self, prompt: str) -> torch.Tensor:
        """Generate a single low-res CT volume from a text prompt.

        Returns:
            Tensor of shape `(B=1, C=1, D, H, W)` produced by the MaskGITTransformer
            pipeline. Exact spatial shape depends on internal sampling; typical
            output is `(1, 1, 201, 128, 128)` per the paper. Caller may apply the
            super-resolution stage separately.
        """
        self._ensure_built()
        return self._model.sample(texts=[prompt])
