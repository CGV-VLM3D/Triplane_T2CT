"""Text2CT inference sampler for VLM3D evaluation.

Wraps the verified ``Text2CTAdapter`` (which reuses upstream
``scripts.diff_model_demo.run_inference`` verbatim) and writes one ``.mha`` per
``EvalCase`` — same contract / save convention as ``report2ct.py`` and ``generatect.py``.

The adapter already returns the final HU ``int16`` volume ``(1, 1, 512, 512, 128)``
(text → CLIP3D → RFlow loop → sliding-window VAE decode → HU rescale), so this sampler
only formats the report text, calls ``inference``, and serialises to ``.mha``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from src.baselines.text2ct_adapter import Text2CTAdapter
from src.eval.samplers.base import AbstractSampler, EvalCase

log = logging.getLogger(__name__)

# Text2CT native output spacing — config_diff_model.json `diffusion_unet_inference.spacing`.
# The model always emits a (512, 512, 128) volume at this physical spacing, so we tag the
# saved .mha with it (CT-CLIP / FID preprocessing resamples from physical spacing, like the
# valid-GT scans). This mirrors upstream save_image(out_spacing=(0.75, 0.75, 3.0)).
_OUTPUT_SPACING_MM = (0.75, 0.75, 3.0)


class Text2CTSampler(AbstractSampler):
    """Generate CT volumes from radiology reports with the released Text2CT weights.

    Args:
        ckpt_dir: directory with the 3 Text2CT checkpoints (top-level or ``models/`` subdir).
        n_steps: RFlow sampling steps (upstream default 30).
        guidance_scale: classifier-free-guidance scale (upstream default 5.0).
        output_spacing_mm: physical spacing written to the .mha (defaults to Text2CT's native).
        name: absorbed from the Hydra model config; unused.
    """

    def __init__(
        self,
        ckpt_dir: str = "/workspace/data/checkpoints/text2ct",
        n_steps: int = 30,
        guidance_scale: float = 5.0,
        output_spacing_mm: list[float] | None = None,
        name: str | None = None,  # absorbed from Hydra config
    ) -> None:
        """Store sampling configuration; adapter and weights are loaded lazily on first call.

        Args:
            ckpt_dir: directory containing the three Text2CT checkpoint files
                (accepted by ``Text2CTAdapter``; may contain a ``models/`` subdirectory).
            n_steps: number of RFlow denoising steps (upstream default 30).
            guidance_scale: classifier-free-guidance scale (upstream default 5.0).
            output_spacing_mm: physical voxel spacing (mm) written to the saved ``.mha``.
                Defaults to Text2CT's native ``(0.75, 0.75, 3.0)`` so the FID runner
                resamples to 1 mm correctly (mirrors upstream ``save_image`` call).
        """
        self.ckpt_dir = ckpt_dir
        self.n_steps = int(n_steps)
        self.guidance_scale = float(guidance_scale)
        self.output_spacing_mm = (
            list(output_spacing_mm)
            if output_spacing_mm is not None
            else list(_OUTPUT_SPACING_MM)
        )
        self._adapter: Text2CTAdapter | None = None

    # ------------------------------------------------------------------ #
    def _init(self, device: torch.device) -> None:
        """Lazily build and load the ``Text2CTAdapter`` (CLIP3D encoder + RFlow UNet).

        No-ops if already initialised.  Called automatically by ``generate``.

        Args:
            device: target device for model weights and sampling tensors.
        """
        if self._adapter is not None:
            return
        if device.type == "cuda":
            dev = f"cuda:{device.index if device.index is not None else 0}"
        else:
            dev = "cpu"
        self._adapter = Text2CTAdapter(
            ckpt_dir=self.ckpt_dir,
            device_str=dev,
            load_weights=True,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.n_steps,
        )

    @staticmethod
    def _report_text(findings: str, impression: str) -> str:
        """Format the conditioning text as Text2CT was trained.

        Mirrors upstream ``scripts/save_embeddings_ctrate.py`` which embeds the report as
        ``"Findings: {findings} Impression: {impressions}"`` before CLIP3D encoding.
        """
        return f"Findings: {findings} Impression: {impression}".strip()

    @staticmethod
    def _save_mha(hu: np.ndarray, spacing_mm: list[float], out_path: Path) -> None:
        """Save HU array as .mha. hu (H, W, D) == (X, Y, Z) → SimpleITK (Z, Y, X)."""
        arr_zyx = hu.transpose(2, 1, 0)  # (D, W, H)
        img = sitk.GetImageFromArray(arr_zyx)
        img.SetSpacing([float(s) for s in spacing_mm])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out_path))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        cases: list[EvalCase],
        out_dir: Path,
        device: torch.device,
    ) -> list[Path]:
        """Generate one .mha prediction per case (resumable: existing files are skipped)."""
        self._init(device)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for case in tqdm(cases, desc="Text2CT inference"):
            out_path = out_dir / f"{case.scan_id}.mha"
            if out_path.exists():
                written.append(out_path)
                continue

            text = self._report_text(case.findings, case.impression)
            vol = self._adapter.inference(text)  # (1, 1, 512, 512, 128) int16 HU
            hu = vol.squeeze().cpu().numpy().astype(np.int16)  # (512, 512, 128)
            self._save_mha(hu, self.output_spacing_mm, out_path)
            written.append(out_path)
            log.info(
                "Saved %s  HU range [%d, %d]",
                out_path.name,
                int(hu.min()),
                int(hu.max()),
            )

        return written


__all__ = ["Text2CTSampler"]
