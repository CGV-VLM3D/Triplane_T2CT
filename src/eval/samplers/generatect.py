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
# resample generatect to 1 mm like the valid-GT / report2ct / text2ct volumes.
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
        use_metadata_template: bool = True,
    ) -> None:
        """Store sampling knobs; model weights are loaded lazily on first ``generate`` call.

        Args:
            num_frames: number of frames for the MaskGITTransformer sampler
                (upstream process.py:58 hardcodes 201).
            transformer_cond_scale: classifier-free-guidance scale for the low-res
                CTViT+Transformer stage (upstream default 5.0).
            superres_cond_scale: guidance scale for the super-resolution stage
                (upstream default 1.0).
            cond_scale: deprecated alias for ``transformer_cond_scale``; takes
                precedence when set (backwards-compat with older Hydra configs).
            use_super_resolution: must be True for VLM3D Task 4 (512×512 output);
                passing False raises immediately.
            final_spacing_mm: ITK (x, y, z) spacing written to the saved .mha.
                Defaults to the truthful native spacing ``(0.75, 0.75, 1.5)`` so the
                FID runner resamples correctly; pass ``[1, 1, 1]`` to reproduce the
                published docker number.
            use_metadata_template: if True, prompt is formatted as the training
                distribution (``"{age} years old {sex}: {impression}"``); if False,
                feeds raw findings+impression to reproduce the docker's behaviour.
        """
        # Build the conditioning prompt the way GenerateCT was TRAINED:
        #   f'{age} years old {sex}: {impression}'   (impression only, NO findings)
        # (third_party/generatect/transformer_maskgit/transformer_maskgit/videotextdataset.py:84).
        # The official docker feeds item["report"] verbatim (process.py:188), but the released
        # weights never saw findings or a raw-report format, so the templated prompt is the
        # in-distribution input. Set False to reproduce the docker's raw findings+impression.
        self.use_metadata_template = use_metadata_template
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
        """Lazily build the ``GenerateCTAdapter`` (CTViT + Transformer + super-res).

        No-ops if already initialised. Called automatically by ``generate``.

        Args:
            device: target device for all model weights and sampling tensors.
        """
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
    #  Prompt construction (mirrors training, videotextdataset.py)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_age(patient_age: str) -> str:
        """Mirror videotextdataset.py:71-72 byte-for-byte (incl. the >99 hundreds quirk).

        '036Y' -> '036'[:-1].zfill(3)='036' -> [1:]='36'. Empty/garbage -> 'None'.
        """
        try:
            if not patient_age:
                return "None"
            age = str(patient_age)[:-1].zfill(3)
            return age[1:]
        except Exception:
            return "None"

    @staticmethod
    def _format_sex(patient_sex: str) -> str:
        """Mirror videotextdataset.py:75-82 — M->male, F->female, else raw/'None'."""
        sex = str(patient_sex) if patient_sex else "None"
        if sex.lower() == "m":
            sex = "male"
        if sex.lower() == "f":
            sex = "female"
        return sex

    def _build_prompt(self, case: EvalCase) -> str:
        """Construct the text prompt; templated to match training by default.

        Templated path reproduces videotextdataset.py:84 +130-133 (impression only,
        age/sex prefix, then strip quotes/parens). Missing metadata yields the same
        'None years old None: ...' the model saw at train time (videotextdataset.py:74,78),
        NOT a findings fallback — that would reintroduce the off-distribution format we are
        fixing. use_metadata_template=False reproduces the docker's raw findings+impression.
        """
        if self.use_metadata_template:
            age = self._format_age(case.age)
            sex = self._format_sex(case.sex)
            text = f"{age} years old {sex}: {case.impression}"
        else:
            text = ((case.findings or "") + " " + (case.impression or "")).strip()
            if not text:
                text = case.findings or case.impression or ""
        # Upstream __getitem__ cleaning (videotextdataset.py:130-133).
        for ch in ('"', "'", "(", ")"):
            text = text.replace(ch, "")
        return text

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
        """Run the two-stage GenerateCT pipeline for each case and save as .mha.

        For each case, builds the conditioning prompt (templated or raw), runs
        CTViT+MaskGITTransformer (low-res ``(1, D, 128, 128)``) then per-slice
        super-resolution (high-res ``(1, D, 512, 512)``), converts to HU, and
        writes a ``<scan_id>.mha``. Existing files are skipped (resumable).

        Args:
            cases: evaluation cases supplying report text and patient metadata.
            out_dir: directory where ``.mha`` files are written.
            device: device for model inference.

        Returns:
            Ordered list of paths to written (or already-existing) ``.mha`` files.
        """
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

            prompt = self._build_prompt(case)

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
