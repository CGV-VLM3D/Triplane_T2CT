"""Minimal CT-RATE EDA helpers (Phase A Day 4).

Per Critic-approved deflated scope:
    1. Label co-occurrence (heatmap, 18 × 18)
    2. Report token-length distribution (findings + impression)
    3. Spacing violin (XY + Z, per split)
    4. HU intensity histogram on a stratified sample of ≤200 scans

All functions take a list[CTRateRecord] (from src.data.ct_rate_datamodule) +
an output directory; they save PNGs and return summary dicts (counts, percentiles)
so the orchestrating notebook/script can also print a textual digest.

Heavy I/O (HU histogram) is gated by `sample_size` to fit within Day 4 wall-clock.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.ct_rate_datamodule import CTRateRecord

logger = logging.getLogger(__name__)


def plot_label_cooccurrence(records: list[CTRateRecord], out_path: Path) -> dict:
    """Save an 18×18 co-occurrence heatmap and return per-label counts."""
    label_names: list[str] = []
    for r in records:
        if r.labels:
            label_names = list(r.labels.keys())
            break
    if not label_names:
        raise RuntimeError("No labels found in records")
    n = len(label_names)
    matrix = np.zeros((n, n), dtype=np.int64)
    counts = Counter()
    for r in records:
        if not r.labels:
            continue
        present = [
            i for i, name in enumerate(label_names) if r.labels.get(name, 0) == 1
        ]
        for i in present:
            counts[label_names[i]] += 1
        for i in present:
            for j in present:
                matrix[i, j] += 1
    diag = matrix.diagonal()
    safe = np.where(diag > 0, diag, 1)
    cond_prob = matrix / safe[None, :]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cond_prob, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_title("CT-RATE label co-occurrence (P(row | column))")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "per_label_counts": dict(counts),
        "n_samples_with_labels": sum(1 for r in records if r.labels),
    }


def plot_report_token_length(records: list[CTRateRecord], out_path: Path) -> dict:
    """Histogram of word-counts in findings + impression. Approximation by whitespace split."""
    findings_len = np.array([len(r.findings.split()) for r in records if r.findings])
    impression_len = np.array(
        [len(r.impression.split()) for r in records if r.impression]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].hist(findings_len, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title(f"Findings word count (n={len(findings_len)})")
    axes[0].set_xlabel("# words")
    axes[0].set_ylabel("samples")
    axes[1].hist(impression_len, bins=50, color="orange", edgecolor="black")
    axes[1].set_title(f"Impression word count (n={len(impression_len)})")
    axes[1].set_xlabel("# words")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "findings": {
            "median": float(np.median(findings_len)),
            "p95": float(np.percentile(findings_len, 95)),
            "max": int(findings_len.max()),
        },
        "impression": {
            "median": float(np.median(impression_len)),
            "p95": float(np.percentile(impression_len, 95)),
            "max": int(impression_len.max()),
        },
    }


def plot_spacing_violin(records: list[CTRateRecord], out_path: Path) -> dict:
    """Violin of XY (first scalar) + Z spacings."""
    xys = np.array([r.spacing_xy for r in records if r.spacing_xy is not None])
    zs = np.array([r.spacing_z for r in records if r.spacing_z is not None])
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([xys, zs], showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], ["steelblue", "orange"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"XY (mm)\n n={len(xys)}", f"Z (mm)\n n={len(zs)}"])
    ax.set_title("CT-RATE spacing distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "xy": {
            "median": float(np.median(xys)),
            "p5_p95": [float(np.percentile(xys, 5)), float(np.percentile(xys, 95))],
        },
        "z": {
            "median": float(np.median(zs)),
            "p5_p95": [float(np.percentile(zs, 5)), float(np.percentile(zs, 95))],
        },
    }


def plot_hu_histogram_sample(
    records: list[CTRateRecord], out_path: Path, sample_size: int = 50, seed: int = 0
) -> dict:
    """HU histogram on a random sample of `sample_size` NIfTI volumes."""
    import nibabel as nib  # noqa: PLC0415

    rng = random.Random(seed)
    pool = [r for r in records if r.nifti_path.is_file()]
    sample = rng.sample(pool, k=min(sample_size, len(pool)))

    all_values: list[np.ndarray] = []
    n_loaded = 0
    for r in sample:
        try:
            arr = nib.load(str(r.nifti_path)).get_fdata().astype(np.float32)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", r.nifti_path, e)
            continue
        # `_fixed` volumes already have HU baked in — no slope/intercept rescale; plot raw HU.
        flat = arr.reshape(-1)
        sub = flat[:: max(1, len(flat) // 50000)]  # subsample voxels to keep RAM small
        all_values.append(sub)
        n_loaded += 1

    values = np.concatenate(all_values) if all_values else np.zeros(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=200, color="slategray", edgecolor="none")
    ax.set_title(f"HU intensity (sampled {n_loaded} volumes; ~50k voxels each)")
    ax.set_xlabel("HU (raw, before clip)")
    ax.set_ylabel("voxel count")
    ax.set_xlim(-1500, 2000)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "n_volumes_loaded": n_loaded,
        "voxel_p5_p95": [
            float(np.percentile(values, 5)),
            float(np.percentile(values, 95)),
        ]
        if values.size
        else None,
        "voxel_median": float(np.median(values)) if values.size else None,
    }


def run_all(
    records: list[CTRateRecord], out_dir: Path, hu_sample_size: int = 50
) -> dict:
    """Produce all 4 figs in `out_dir`. Returns a summary dict for logging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_records": len(records),
        "label_cooccurrence": plot_label_cooccurrence(
            records, out_dir / "label_cooccurrence.png"
        ),
        "report_token_len": plot_report_token_length(
            records, out_dir / "report_token_len.png"
        ),
        "spacing_violin": plot_spacing_violin(records, out_dir / "spacing_violin.png"),
        "hu_histogram": plot_hu_histogram_sample(
            records, out_dir / "hu_histogram.png", sample_size=hu_sample_size
        ),
    }
    return summary
