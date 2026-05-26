"""MAISI VAE frozen loader, sourced directly from the MAISI bundle config.

Architecture kwargs are NOT duplicated here — we load `third_party/maisi_bundle/configs/inference.json`
via `monai.bundle.ConfigParser` and let it instantiate `autoencoder_def`. This way, any
upstream MAISI bundle update is picked up automatically.

R6 mitigation test: tests/test_maisi_frozen_load.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from monai.bundle import ConfigParser

MAISI_BUNDLE_DIR: Final[Path] = Path("/workspace/third_party/maisi_bundle")
MAISI_INFERENCE_CONFIG: Final[Path] = MAISI_BUNDLE_DIR / "configs" / "inference.json"
MAISI_CKPT_PATH: Final[Path] = MAISI_BUNDLE_DIR / "models" / "autoencoder.pt"

# Sliding-window inference defaults (matches the bundle's autoencoder_sliding_window_infer_size).
SW_ROI_SIZE: Final[tuple[int, int, int]] = (80, 80, 80)
SW_OVERLAP: Final[float] = 0.4


def load_frozen(
    ckpt_path: Path | str = MAISI_CKPT_PATH,
    bundle_config: Path | str = MAISI_INFERENCE_CONFIG,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Load the MAISI VAE from the bundle and freeze every parameter.

    Args:
        ckpt_path: override the bundle's checkpoint location (tests only).
        bundle_config: bundle inference config that defines `autoencoder_def`.
        device: torch device. Default CPU; callers move to GPU when they need it.
        eval_mode: if True, call .eval().

    Returns:
        AutoencoderKlMaisi instance with all params requires_grad=False.
    """
    ckpt_path = Path(ckpt_path)
    bundle_config = Path(bundle_config)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"MAISI checkpoint not found at {ckpt_path}")
    if not bundle_config.is_file():
        raise FileNotFoundError(
            f"MAISI bundle inference config not found at {bundle_config}"
        )

    parser = ConfigParser()
    parser.read_config(str(bundle_config))
    autoencoder = parser.get_parsed_content("autoencoder_def")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    autoencoder.load_state_dict(state_dict)

    for p in autoencoder.parameters():
        p.requires_grad_(False)

    autoencoder = autoencoder.to(device)
    if eval_mode:
        autoencoder.eval()
    return autoencoder


def is_fully_frozen(model: torch.nn.Module) -> bool:
    """Return True iff every parameter has requires_grad=False."""
    return all(not p.requires_grad for p in model.parameters())
