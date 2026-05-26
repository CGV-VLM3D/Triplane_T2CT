"""Report2CT LightningModule adapter (skeleton).

Wraps `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` (the submodule's
own model definition) via `monai.bundle.ConfigParser`. Architecture kwargs are NOT
duplicated here — any upstream change in the submodule config flows through.

Day 4 status: **skeleton only**. We:
  - build the `DiffusionModelUNetMaisi` from the submodule config (233 M params)
  - run a 1-step forward smoke on a tiny latent (CPU-friendly) to confirm shape
    invariance: in_shape == out_shape
  - do NOT load weights (the user runs full training in Phase B and produces them)

Phase B 6/1 will:
  - add the multi-text-encoder precompute step (3 HF text encoders, concat → 2560)
  - add the MAISI VAE-encoded image latent precompute step
  - launch `torchrun` against `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py`
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from lightning.pytorch import LightningModule
from monai.bundle import ConfigParser

REPORT2CT_DIR: Final[Path] = Path("/workspace/third_party/report2ct")
MODEL_DEF_CONFIG: Final[Path] = (
    REPORT2CT_DIR / "vlm3D_work_dir" / "config_maisi_2560.json"
)
ENV_CONFIG: Final[Path] = (
    REPORT2CT_DIR
    / "vlm3D_work_dir"
    / "environment_maisi_diff_model_vlm3D_FI_2560_multi.json"
)
TRAIN_CONFIG: Final[Path] = (
    REPORT2CT_DIR / "vlm3D_work_dir" / "config_maisi_diff_model_vlm3D.json"
)

# Standard Report2CT latent shape (MAISI VAE output).
LATENT_SHAPE: Final[tuple[int, int, int, int]] = (4, 120, 120, 64)  # (C, H, W, D)
CONTEXT_DIM: Final[int] = 2560  # 3 text encoders concatenated (768 + 1024 + 768)


def build_unet(config_path: Path | str = MODEL_DEF_CONFIG) -> torch.nn.Module:
    """Instantiate the DiffusionModelUNetMaisi from the Report2CT submodule config.

    Does NOT load weights — Report2CT does not release them. User-driven Phase B
    training will produce a checkpoint via the submodule's training script.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Report2CT model def not found at {config_path}")
    parser = ConfigParser()
    parser.read_config(str(config_path))
    unet = parser.get_parsed_content("diffusion_unet_def")
    return unet


def forward_one_step(
    unet: torch.nn.Module,
    latent: torch.Tensor,
    context: torch.Tensor,
    spacing: torch.Tensor,
    timestep: int = 500,
    class_label: int = 0,
) -> torch.Tensor:
    """Single denoising step.

    Args:
        latent: (B, 4, H, W, D) noisy latent.
        context: (B, L, 2560) text conditioning (L=2 for findings+impression).
        spacing: (B, 3) voxel spacing in mm.
        timestep: int — single timestep shared across the batch for the smoke.
        class_label: int in [0, num_class_embeds=128) — MAISI's modality class.
            The submodule training script passes modality=1 (CT). Default 0 for smoke.

    Returns:
        Predicted velocity / noise of the same shape as `latent`.
    """
    b = latent.shape[0]
    t = torch.full((b,), timestep, dtype=torch.long, device=latent.device)
    cls = torch.full((b,), class_label, dtype=torch.long, device=latent.device)
    return unet(
        x=latent, timesteps=t, context=context, class_labels=cls, spacing_tensor=spacing
    )


class Report2CTAdapter(LightningModule):
    """LightningModule skeleton wrapping the Report2CT UNet.

    Phase B 6/1: training_step / validation_step are wired through the submodule's
    `diff_model_train_vlm3D_2560_multi_text.py` (we call its main as a subprocess
    via scripts/run_report2ct_training.sh). For Phase A Day 4 the LightningModule
    only exposes `forward` for the 1-step gate and Phase B B.3 diagnostic hooks.
    """

    def __init__(self, device_str: str = "cpu") -> None:
        super().__init__()
        self._device_str = device_str
        self.unet = build_unet()

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        spacing: torch.Tensor,
        timestep: int = 500,
    ) -> torch.Tensor:
        return forward_one_step(self.unet, latent, context, spacing, timestep=timestep)
