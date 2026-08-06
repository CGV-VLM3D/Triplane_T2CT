#!/usr/bin/env python3
"""E1 — the two geometry distributions that frame the whole spacing question.

F1: real GT field-of-view (mm) over the 1304 valid_v2 volumes, with the FOV that each
    candidate declared spacing implies for our fixed 480x480x256 output grid.
F2: the spacing values report2ct was actually *conditioned on* during training, read from
    the datalist entries (= post-resize effective spacing, FOV/480).

Both are header-only reads — no volume data is loaded.

The point of putting them side by side: training spacing IS FOV/480 for the same CT-RATE
population, so "the spacing the model saw most" and "the spacing whose FOV matches GT" are
the same number by construction. That degeneracy is why the optimum alone cannot separate
hypothesis A (declared scale) from B (conditioning) — see the plan's H0.

Usage:
    python tests/spacing_fov/geometry_stats.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

sys.path.insert(0, "/workspace")

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs"
RESULTS = HERE / "results"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_valid_full_3001")
DATALIST = pathlib.Path("/workspace/data/report2ct_work_dir/datalist_v2.json")

# report2ct's output grid is fixed by the (4,120,120,64) latent (report2ct.py:33), so the
# declared FOV is exactly grid x spacing.
GRID_INPLANE, GRID_Z = 480, 256
CANDIDATES_INPLANE = [0.70, 0.757, 0.80, 1.00]  # mm
CANDIDATES_Z = [1.31, 1.50]  # mm


def _hdr_fov(path: pathlib.Path) -> tuple[float, float]:
    """Read one .mha header → (in-plane FOV mm, z FOV mm). No pixel data is read."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    size, spacing = r.GetSize(), r.GetSpacing()
    return size[0] * spacing[0], size[2] * spacing[2]


def gt_fov_mm() -> np.ndarray:
    """FOV of every GT volume. Returns ``(N, 2)`` = (in-plane mm, z mm)."""
    files = sorted(GT_DIR.glob("*.mha"))
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_hdr_fov, files))
    return np.array(rows, dtype=np.float64)  # (N, 2)


def train_spacing_mm() -> np.ndarray:
    """Spacing the UNet was conditioned on, per training sample. Returns ``(N, 2)``.

    Reads the ``"spacing"`` key of each datalist entry's merged JSON — the same value the
    datamodule feeds the UNet (report2ct_datamodule.py:69).
    """
    entries = json.loads(DATALIST.read_text())["training"]

    def _one(entry: dict) -> tuple[float, float] | None:
        try:
            sp = json.loads(pathlib.Path(entry["spacing"]).read_text())["spacing"]
        except (OSError, KeyError, json.JSONDecodeError):
            return None
        return sp[0], sp[2]

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = [r for r in ex.map(_one, entries) if r is not None]
    return np.array(rows, dtype=np.float64)  # (N, 2)


def _stats(x: np.ndarray) -> dict[str, float]:
    """mean / median / p5 / p95 / min / max of a 1-D array."""
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _panel(ax, data, stats, title, xlabel, marks) -> None:
    """One histogram panel with median/p5/p95 bands and candidate vertical lines."""
    ax.hist(data, bins=60, color="0.75", edgecolor="none")
    ax.axvspan(
        stats["p5"], stats["p95"], color="tab:blue", alpha=0.10, label="GT p5–p95"
    )
    ax.axvline(
        stats["median"], color="tab:blue", lw=2, label=f"median {stats['median']:.3g}"
    )
    for value, label, color in marks:
        ax.axvline(value, color=color, ls="--", lw=1.6, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=7)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    fov = gt_fov_mm()  # (N, 2)
    spc = train_spacing_mm()  # (N, 2)
    out = {
        "n_gt": int(fov.shape[0]),
        "n_train": int(spc.shape[0]),
        "gt_fov_inplane_mm": _stats(fov[:, 0]),
        "gt_fov_z_mm": _stats(fov[:, 1]),
        "train_spacing_inplane_mm": _stats(spc[:, 0]),
        "train_spacing_z_mm": _stats(spc[:, 1]),
        "declared_fov_inplane_mm": {
            str(s): s * GRID_INPLANE for s in CANDIDATES_INPLANE
        },
        "declared_fov_z_mm": {str(s): s * GRID_Z for s in CANDIDATES_Z},
        # fraction of GT volumes whose FOV is smaller than what each candidate declares
        "gt_frac_below_declared_inplane": {
            str(s): float((fov[:, 0] < s * GRID_INPLANE).mean())
            for s in CANDIDATES_INPLANE
        },
        "gt_frac_below_declared_z": {
            str(s): float((fov[:, 1] < s * GRID_Z).mean()) for s in CANDIDATES_Z
        },
    }

    colors = ["tab:green", "tab:olive", "tab:orange", "tab:red"]

    # ---- F1: GT FOV vs declared FOV ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _panel(
        axes[0],
        fov[:, 0],
        out["gt_fov_inplane_mm"],
        "GT in-plane FOV vs declared (480 px x spacing)",
        "in-plane FOV (mm)",
        [
            (s * GRID_INPLANE, f"s={s} → {s * GRID_INPLANE:.0f}mm", c)
            for s, c in zip(CANDIDATES_INPLANE, colors)
        ],
    )
    _panel(
        axes[1],
        fov[:, 1],
        out["gt_fov_z_mm"],
        "GT z FOV vs declared (256 px x spacing)",
        "z FOV (mm)",
        [
            (s * GRID_Z, f"z={s} → {s * GRID_Z:.0f}mm", c)
            for s, c in zip(CANDIDATES_Z, ["tab:olive", "tab:red"])
        ],
    )
    fig.tight_layout()
    fig.savefig(FIGS / "F1_gt_fov_hist.png", dpi=150)
    plt.close(fig)

    # ---- F2: training conditioning distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _panel(
        axes[0],
        spc[:, 0],
        out["train_spacing_inplane_mm"],
        "Training conditioning: in-plane spacing",
        "spacing (mm)",
        [(s, f"s={s}", c) for s, c in zip(CANDIDATES_INPLANE, colors)],
    )
    _panel(
        axes[1],
        spc[:, 1],
        out["train_spacing_z_mm"],
        "Training conditioning: z spacing",
        "spacing (mm)",
        [(s, f"z={s}", c) for s, c in zip(CANDIDATES_Z, ["tab:olive", "tab:red"])],
    )
    fig.tight_layout()
    fig.savefig(FIGS / "F2_train_spacing_hist.png", dpi=150)
    plt.close(fig)

    (RESULTS / "geometry_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[E1] figs → {FIGS}/F1_gt_fov_hist.png, F2_train_spacing_hist.png")


if __name__ == "__main__":
    main()
