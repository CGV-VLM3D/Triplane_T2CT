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
    """

    scan_id: str
    findings: str
    impression: str
    spacing_mm: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    age: str = ""
    sex: str = ""


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
