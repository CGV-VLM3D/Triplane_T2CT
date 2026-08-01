"""Abstract base sampler for VLM3D evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EvalCase:
    """One evaluation case: an ID, its radiology report, and optional voxel spacing.

    ``age`` / ``sex`` are the raw CT-RATE metadata strings (e.g. ``"036Y"`` / ``"M"``);
    they are populated for the GenerateCT prompt template (which was trained on
    ``"{age} years old {sex}: {impression}"`` — see GenerateCTSampler) and left empty
    by callers that don't need them.

    The last four fields carry one row of a mask-intervention manifest
    (docs/mask_intervention_manifest.md) and are ``None`` in every plain run, where one scan is
    generated exactly once. They exist because a manifest run generates the SAME scan several
    times under different conditioning masks, so the output filename, the mask source and the
    noise seed can no longer be derived from ``scan_id`` alone.
    """

    scan_id: str
    findings: str
    impression: str
    spacing_mm: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    age: str = ""
    sex: str = ""
    sample_id: str | None = None
    condition: str | None = None
    cond_mask_source_id: str | None = None
    seed: int | None = None

    @property
    def out_stem(self) -> str:
        """Filename stem of this case's generated volume.

        The manifest ``sample_id`` when this case came from one, else ``scan_id`` — i.e. a plain
        run keeps the existing "1 scan = 1 prediction named after the scan" convention.
        """
        return self.sample_id or self.scan_id


class AbstractSampler(ABC):
    """Generate predicted volumes for a list of EvalCases.

    Subclasses implement model-specific loading and inference. The contract:
    - Accept ``cases`` (list of EvalCase) and an output directory.
    - For each case write ``out_dir/{case.scan_id}.mha`` (HU int16, 1 mm isotropic).
    - Return the list of written paths (same order as ``cases``).
    """

    @abstractmethod
    def generate(
        self,
        cases: list[EvalCase],
        out_dir: Path,
        device: torch.device,
    ) -> list[Path]:
        """Generate one .mha file per case; return written paths."""


__all__ = ["EvalCase", "AbstractSampler"]
