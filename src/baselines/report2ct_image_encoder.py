"""Report2CT image encoder — ported from upstream `diff_model_create_training_data_vlm3D_all.py`.

Encodes a NIfTI CT volume through the frozen MAISI VAE and saves a 4-channel
latent `*_emb.nii.gz` in upstream's expected shape (H, W, D, C) = (120, 120, 64, 4)
for 480×480×256 inputs.

CT-RATE `_fixed` inputs already have HU baked in (the offline fix_data step applied
RescaleSlope/Intercept once), so we do NOT re-apply slope/intercept here — doing so would
corrupt every volume (intercept ≈ -8192 → all-air). See [[ctrate-fixed-hu-no-rescale]].

Source line correspondences:
- transforms (LoadImaged → ... → Resized): upstream :85-116
- encode: upstream process_file :287-301
- output NIfTI layout (transpose 1,2,3,0): upstream :299
- ScaleIntensityRanged (a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True): upstream :110-112
- Resized spatial_size=(480,480,256), mode='trilinear': upstream :113 + main fn :389

Frozen VAE loader is reused from `src.baselines.maisi.load_frozen` (already R6-tested).

Parity safety net: `tests/test_report2ct_parity.py::test_image_encoder_parity`.
"""

from __future__ import annotations

import os
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose
from torch.amp import autocast

from src.baselines.maisi import load_frozen

OUTPUT_SPATIAL_SIZE: tuple[int, int, int] = (
    480,
    480,
    256,
)  # upstream diff_model_create_training_data_vlm3D_all.py:389
HU_CLIP_RANGE: tuple[float, float] = (-1000.0, 1000.0)  # upstream :110-112
NORM_RANGE: tuple[float, float] = (0.0, 1.0)  # upstream :110-112


def _build_transforms() -> Compose:
    # `_fixed` inputs are already HU — clip + scale only, no slope/intercept rescale.
    return Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
            monai.transforms.ScaleIntensityRanged(
                keys="image",
                a_min=HU_CLIP_RANGE[0],
                a_max=HU_CLIP_RANGE[1],
                b_min=NORM_RANGE[0],
                b_max=NORM_RANGE[1],
                clip=True,
            ),
            monai.transforms.Resized(
                keys="image", spatial_size=OUTPUT_SPATIAL_SIZE, mode="trilinear"
            ),
        ]
    )


class Report2CTImageEncoder:
    """Encode a NIfTI CT → MAISI latent → save as 4-channel `*_emb.nii.gz` (H,W,D,C)."""

    def __init__(
        self,
        vae_ckpt: str | Path | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.device = torch.device(device)
        load_kwargs = {"device": self.device}
        if vae_ckpt is not None:
            load_kwargs["ckpt_path"] = Path(vae_ckpt)
        self.vae = load_frozen(**load_kwargs)

        self.transforms = _build_transforms()

    @torch.inference_mode()
    def encode(self, nifti_path: str | Path) -> nib.Nifti1Image:
        """Encode a NIfTI volume and return the latent as a `nib.Nifti1Image` (float32, HWDC)."""
        nifti_path = str(nifti_path)
        if not os.path.isfile(nifti_path):
            raise FileNotFoundError(nifti_path)
        out = self.transforms({"image": nifti_path})
        nda_image = out["image"]
        affine = (
            nda_image.meta["affine"].numpy()
            if hasattr(nda_image, "meta")
            else np.eye(4)
        )
        x = (
            nda_image.numpy().squeeze()
            if hasattr(nda_image, "numpy")
            else np.asarray(nda_image).squeeze()
        )

        pt = (
            torch.from_numpy(x).float().to(self.device).unsqueeze(0).unsqueeze(0).half()
        )
        with autocast(
            "cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float16
        ):
            z = self.vae.encode_stage_2_inputs(pt)  # (1, C, H, W, D)
        out_nda = (
            z.squeeze(0).cpu().detach().float().numpy().transpose(1, 2, 3, 0)
        )  # (H,W,D,C)
        return nib.Nifti1Image(np.float32(out_nda), affine=affine)

    def encode_to_file(self, nifti_path: str | Path, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = self.encode(nifti_path)
        nib.save(img, str(out_path))
        return out_path


def latent_dim_and_spacing(emb_nifti_path: str | Path) -> tuple[list[int], list[float]]:
    """Read back dim (3 ints) and spacing (3 floats) from an `*_emb.nii.gz` file.

    These two fields are what the Report2CT training script's transforms expect
    in the companion JSON (`<id>_emb.nii.gzmulti_2560.json` keys `dim`, `spacing`).
    """
    img = nib.load(str(emb_nifti_path))
    dim = list(img.shape[:3])
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    return dim, spacing


__all__ = ["Report2CTImageEncoder", "latent_dim_and_spacing", "OUTPUT_SPATIAL_SIZE"]
