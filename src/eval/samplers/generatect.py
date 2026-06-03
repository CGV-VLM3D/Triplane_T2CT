"""GenerateCT inference sampler — sampling pipeline mirrors upstream vlm3d_dockers process.py.

Source of truth: /workspace/third_party/vlm3d_dockers/ctgen_example_docker/process.py
  - TRANSFORMER_NUM_FRAMES = 201      (process.py:58)
  - TRANSFORMER_COND_SCALE = 5.0      (process.py:59)
  - SUPERRES_COND_SCALE    = 1.0      (process.py:62)
  - Low-res → super-res:   clip(-1, 1) then (x+1)/2 → [0, 1]  (process.py:209-212)
  - HU conversion:         clip([0, 1]) * 2000 - 1000 → clip(-1000, 1000)  (process.py:152-155)

ONE deliberate divergence from the docker: the saved voxel spacing. The docker stamps
(1, 1, 1) (process.py:65), but GenerateCT's true native spacing is (0.75, 0.75, 1.5) mm
(CT-RATE preprocessing). We stamp the truthful native so the FID runner resamples it to
1 mm like every other volume (GT / report2ct / text2ct) — a fair same-GT comparison. Pass
`final_spacing_mm=[1, 1, 1]` only to reproduce the published docker number. See plan
`silent-bug-floating-sunset.md` (WI-2).
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from src.baselines.generatect_adapter import GenerateCTAdapter
from src.eval.samplers.base import AbstractSampler, EvalCase

log = logging.getLogger(__name__)

# upstream process.py:58-65 (sampling knobs bit-identical to the official docker)
_TRANSFORMER_NUM_FRAMES = 201
_TRANSFORMER_COND_SCALE = 5.0
_SUPERRES_COND_SCALE = 1.0
_VALUE_RANGE = (-1000, 1000)

# Saved voxel spacing in ITK (x, y, z) = (in-plane, in-plane, slice) order. GenerateCT's 512²
# output is trained on CT-RATE resampled to (0.75, 0.75, 1.5) mm
# (third_party/vlm3d_dockers/reportgen_example_docker/CT-CLIP/data_preprocess/preprocess_ctrate_train.py:93-95).
# Stamping this truthful native (not the docker's distorting (1, 1, 1)) lets the FID runner
# resample generatect to 1 mm like the proxy-GT / report2ct / text2ct volumes.
_NATIVE_SPACING = (0.75, 0.75, 1.5)
_DOCKER_SPACING = (
    1.0,
    1.0,
    1.0,
)  # for `final_spacing_mm` override only (published-number repro)


class GenerateCTSampler(AbstractSampler):
    """Inference sampler that mirrors upstream ctgen_example_docker/process.py exactly.

    Loads the pretrained CTViT + MaskGITTransformer + super-resolution diffusion
    via our existing GenerateCTAdapter, but inlines the inter-stage conversion so
    we don't depend on the adapter's hires helper (which lacks the [-1,1]→[0,1]
    rescale that the super-res model requires).
    """

    def __init__(
        self,
        num_frames: int = _TRANSFORMER_NUM_FRAMES,
        transformer_cond_scale: float = _TRANSFORMER_COND_SCALE,
        superres_cond_scale: float = _SUPERRES_COND_SCALE,
        name: str | None = None,  # absorbed from Hydra config
        # backwards-compat with older configs
        cond_scale: float | None = None,
        starting_temperature: float | None = None,
        use_super_resolution: bool = True,
        n_samples_per_case: int = 1,
        final_spacing_mm: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.transformer_cond_scale = (
            cond_scale if cond_scale is not None else transformer_cond_scale
        )
        self.superres_cond_scale = superres_cond_scale
        # Saved voxel spacing: truthful native by default; pass [1,1,1] for docker-parity repro.
        self._final_spacing = (
            tuple(float(s) for s in final_spacing_mm)
            if final_spacing_mm is not None
            else _NATIVE_SPACING
        )
        # use_super_resolution is required for VLM3D evaluation (512×512 output)
        if not use_super_resolution:
            raise ValueError(
                "VLM3D Task 4 requires super-resolution (512×512). Set use_super_resolution=true."
            )

        self._adapter: GenerateCTAdapter | None = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------ #
    #  Lazy init                                                          #
    # ------------------------------------------------------------------ #

    def _init(self, device: torch.device) -> None:
        if self._adapter is not None:
            return
        self._device = device
        device_str = str(device)
        log.info("Loading GenerateCT on %s …", device_str)
        self._adapter = GenerateCTAdapter(
            device_str=device_str,
            load_super_resolution=True,
        )
        self._adapter._ensure_built()

    # ------------------------------------------------------------------ #
    #  Per-case inference (mirrors process.py:196-235)                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_one(self, prompt: str) -> torch.Tensor:
        """Run the upstream two-stage pipeline for one prompt.

        Returns the high-res volume tensor of shape (1, D, 512, 512) in [0, 1].
        """
        # Stage 1: low-res CTViT + MaskGITTransformer
        with redirect_stdout(io.StringIO()):
            low_res_tensor = self._adapter._model.sample(
                texts=[prompt],
                num_frames=self.num_frames,
                cond_scale=self.transformer_cond_scale,
            )
        # low_res_tensor shape: (1, 1, D, 128, 128) in approximately [-1, 1]

        # Stage 2: per-slice super-resolution (process.py:209-227)
        # permute (B, C, D, H, W) → (B, D, C, H, W), then squeeze batch → (D, C, H, W)
        low_res_slices = low_res_tensor.permute(0, 2, 1, 3, 4).squeeze(0)
        low_res_slices = np.clip(low_res_slices.cpu().numpy(), -1, 1)
        low_res_slices = (low_res_slices + 1) / 2  # → [0, 1]
        low_res_slices = torch.tensor(low_res_slices, device=self._device)

        high_res_slices: list[torch.Tensor] = []
        for slice_tensor in low_res_slices:
            input_slice = slice_tensor.unsqueeze(0).to(self._device)  # (1, C, H, W)
            with redirect_stdout(io.StringIO()):
                high_res_slice = (
                    self._adapter._sr_infer.sample(
                        start_image_or_video=input_slice,
                        texts=[prompt],
                        cond_scale=self.superres_cond_scale,
                        start_at_unet_number=2,
                    )
                    .detach()
                    .cpu()
                )
            high_res_slices.append(high_res_slice.squeeze(0))

        # process.py:228-229: stack(dim=0) → (D, 1, 512, 512), permute(1, 0, 3, 2) → (1, D, 512, 512)
        high_res_volume = torch.stack(high_res_slices, dim=0)
        high_res_volume = high_res_volume.permute(1, 0, 3, 2)
        return high_res_volume  # (1, D, 512, 512) in approximately [0, 1]

    # ------------------------------------------------------------------ #
    #  Save (process.py:save_final_mha)                                   #
    # ------------------------------------------------------------------ #

    def _save_mha(self, tensor: torch.Tensor, out_path: Path) -> tuple[int, int]:
        """Mirror upstream save_final_mha (HU pipeline identical); stamp truthful native spacing.

        Returns final (min, max) HU for logging. `self._final_spacing` is ITK (x, y, z): the
        squeezed array (D, 512, 512) → GetImageFromArray (z, y, x) so spacing (0.75, 0.75, 1.5)
        puts 1.5 mm on the slice axis and 0.75 mm in-plane.
        """
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected single channel tensor, got shape {tensor.shape}"
            )
        array = tensor.squeeze(0).cpu().numpy()
        array = np.clip(array, 0, 1)
        array = array * 2000
        array = array - 1000
        array = np.clip(array, _VALUE_RANGE[0], _VALUE_RANGE[1])

        itk = sitk.GetImageFromArray(array)
        itk.SetSpacing(self._final_spacing)
        itk.SetOrigin((0, 0, 0))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(itk, str(out_path))
        return int(array.min()), int(array.max())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        cases: list[EvalCase],
        out_dir: Path,
        device: torch.device,
    ) -> list[Path]:
        self._init(device)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for case in tqdm(cases, desc="GenerateCT inference"):
            out_path = out_dir / f"{case.scan_id}.mha"
            if out_path.exists():
                log.info("Skipping %s (already exists)", out_path.name)
                written.append(out_path)
                continue

            prompt = ((case.findings or "") + " " + (case.impression or "")).strip()
            if not prompt:
                prompt = case.findings or case.impression or ""

            high_res = self._generate_one(prompt)
            mn, mx = self._save_mha(high_res, out_path)
            written.append(out_path)
            log.info(
                "Saved %s  shape=%s  HU range [%d, %d]",
                out_path.name,
                tuple(high_res.shape),
                mn,
                mx,
            )

        return written


__all__ = ["GenerateCTSampler"]
