"""Load + validate the subgroup definitions YAML (label_burden_bands / organ_clusters / small_n)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.eval.analysis.labels import LABEL_NAMES


@dataclass
class SubgroupConfig:
    """Validated subgroup definitions (see ``configs/eval/subgroup/default.yaml``)."""

    label_burden_bands: dict[str, tuple[int, int]] = field(default_factory=dict)
    organ_clusters: dict[str, list[str]] = field(default_factory=dict)
    subgroup_fid_small_n: int = 100


def load(path: str | Path) -> SubgroupConfig:
    """Load and validate a subgroup config YAML.

    Args:
        path: path to the YAML file (e.g. ``configs/eval/subgroup/default.yaml``).

    Returns:
        A validated :class:`SubgroupConfig`.

    Raises:
        ValueError: if any label name in ``organ_clusters`` is not one of the 18 canonical
            label names (:data:`src.eval.analysis.labels.LABEL_NAMES`) — catches typos before
            they silently drop a label from every subgroup.
    """
    raw = yaml.safe_load(Path(path).read_text())

    known = set(LABEL_NAMES)
    unknown: list[str] = []
    for cluster_name, label_list in raw.get("organ_clusters", {}).items():
        for label in label_list:
            if label not in known:
                unknown.append(f"{cluster_name}: {label!r}")
    if unknown:
        raise ValueError(
            f"Unknown label name(s) in organ_clusters (typo? must match src.eval.analysis."
            f"labels.LABEL_NAMES exactly): {unknown}"
        )

    bands = {
        name: (int(lo), int(hi))
        for name, (lo, hi) in raw.get("label_burden_bands", {}).items()
    }

    return SubgroupConfig(
        label_burden_bands=bands,
        organ_clusters=raw.get("organ_clusters", {}),
        subgroup_fid_small_n=int(raw.get("subgroup_fid_small_n", 100)),
    )


__all__ = ["SubgroupConfig", "load"]
