"""GT-vs-generated 3-plane QC figures — orientation check + visual quality spot-check.

Rendering helpers (``_load_lps``/``_body_centroid``/``_win``) are small, standalone re-implementations
of the ones in [tests/abnormality_fidelity/make_figures.py](../../../tests/abnormality_fidelity/make_figures.py)
(a test-only script, not importable into ``src/``), reduced to a single model (rows = {GT, model},
cols = {axial, coronal, sagittal}).

**Orientation is fixed, not per-run flip-searched**: every current sampler saves LPS content
(the ``ras_to_lps`` flip applied at save time), and GT is raw/LPS too — so both are read with
NO in-plane flip (``is_lps=True`` always). This "no-flip alignment" IS the orientation check the
figures are for: if a sampler regresses to RAS, the figures will visibly mismatch GT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.eval.analysis.labels import LABEL_NAMES

log = logging.getLogger(__name__)

# (window level, window width) per abnormality; generic default for anything not listed / normal.
_DEFAULT_WINDOW = (-550, 1500)  # lung window
_WINDOWS: dict[str | None, tuple[float, float]] = {
    "Cardiomegaly": (40, 400),
    "Pericardial effusion": (40, 400),
    "Hiatal hernia": (40, 400),
    "Pleural effusion": (40, 400),
    "Lung nodule": (-550, 1500),
    "Emphysema": (-550, 1500),
    "Consolidation": (-450, 1400),
    "Arterial wall calcification": (100, 700),
    "Coronary artery wall calcification": (100, 700),
    None: _DEFAULT_WINDOW,  # normal / all-zero showcase
}

# Priority order for picking diverse showcase cases; None = an all-zero (normal) case.
_SHOWCASE_ORDER: list[str | None] = [
    "Cardiomegaly",
    "Lung nodule",
    "Pleural effusion",
    "Arterial wall calcification",
    None,
    "Emphysema",
    "Consolidation",
    "Hiatal hernia",
    "Coronary artery wall calcification",
]


def _load_lps(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a volume as (Z,Y,X) HU array + (sx,sy,sz) mm. No flip — see module docstring."""
    im = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(im).astype(np.float32)
    return np.ascontiguousarray(arr), im.GetSpacing()


def _body_centroid(arr: np.ndarray) -> tuple[int, int, int]:
    body = arr > -500
    if not body.any():
        return tuple(s // 2 for s in arr.shape)
    zc, yc, xc = (int(round(c)) for c in np.array(np.where(body)).mean(axis=1))
    return zc, yc, xc


def _win(sl: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def _select_cases(
    scan_ids: list[str], label_map: dict[str, np.ndarray], n: int
) -> list[tuple[str, str | None]]:
    """Pick up to ``n`` (scan_id, showcase_label) pairs spanning diverse abnormalities.

    Prefers the lowest label-burden positive case per showcase label (a "cleaner" example, same
    heuristic as the old make_figures.py picker), then backfills from any remaining label if the
    curated showcase order doesn't fill all ``n`` slots.
    """
    by_label: dict[str | None, list[tuple[int, str]]] = {}
    for scan_id in scan_ids:
        vec = label_map.get(scan_id)
        if vec is None:
            continue
        burden = int(vec.sum())
        if burden == 0:
            by_label.setdefault(None, []).append((burden, scan_id))
        for name, val in zip(LABEL_NAMES, vec):
            if val:
                by_label.setdefault(name, []).append((burden, scan_id))
    for cands in by_label.values():
        cands.sort()

    picked: list[tuple[str, str | None]] = []
    used: set[str] = set()

    def _take(label: str | None) -> bool:
        for _, scan_id in by_label.get(label, []):
            if scan_id not in used:
                picked.append((scan_id, label))
                used.add(scan_id)
                return True
        return False

    for label in _SHOWCASE_ORDER:
        if len(picked) >= n:
            break
        _take(label)

    if len(picked) < n:
        for label in by_label:
            if len(picked) >= n:
                break
            _take(label)

    return picked[:n]


def _render_case(
    scan_id: str,
    label: str | None,
    gt_dir: Path,
    pred_dir: Path,
    model_name: str,
    out_png: Path,
) -> None:
    wl, ww = _WINDOWS.get(label, _DEFAULT_WINDOW)
    rows = [("GT", gt_dir), (model_name, pred_dir)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)))
    for r, (name, d) in enumerate(rows):
        arr, (sx, sy, sz) = _load_lps(d / f"{scan_id}.mha")
        _, yc, xc = _body_centroid(arr)
        zs = np.where((arr > -500).any(axis=(1, 2)))[0]
        z0, z1 = (int(zs.min()), int(zs.max())) if zs.size else (0, arr.shape[0] - 1)
        za = int(np.clip(z0 + round(0.5 * (z1 - z0)), 0, arr.shape[0] - 1))

        panels = [
            (_win(arr[za, :, :], wl, ww), sy / sx, "axial"),
            (_win(arr[::-1, yc, :], wl, ww), sz / sx, "coronal"),
            (_win(arr[::-1, :, xc], wl, ww), sz / sy, "sagittal"),
        ]
        for c, (img, aspect, plane) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", aspect=aspect, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(plane, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")

    title = "NORMAL (no abnormality labels)" if label is None else label
    fig.suptitle(
        f"{scan_id}  —  {title}\nwindow WL={wl} WW={ww} (LPS, superior up)",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(
    pred_dir: str | Path,
    gt_dir: str | Path,
    label_map: dict[str, np.ndarray],
    out_dir: str | Path,
    model_name: str = "model",
    n: int = 5,
) -> list[Path]:
    """Render up to ``n`` GT-vs-model QC figures spanning diverse abnormalities.

    Args:
        pred_dir: generated ``.mha`` directory.
        gt_dir: GT ``.mha`` directory (same scan_id stems).
        label_map: ``{scan_id: label_vector(18,)}`` (e.g. from
            ``src.eval.analysis.labels.load_label_csv("valid")``).
        out_dir: output directory (``analysis/figures/`` per plan §3.13).
        model_name: row label for the generated-volume row.
        n: number of cases to render (fewer if the prediction set can't span that many distinct
            abnormalities — logged, not padded with duplicates).

    Returns:
        List of written PNG paths.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)

    scan_ids = sorted(
        p.stem for p in pred_dir.glob("*.mha") if (gt_dir / p.name).is_file()
    )
    cases = _select_cases(scan_ids, label_map, n)
    if len(cases) < n:
        log.warning(
            "QC figures: only %d/%d distinct-abnormality cases available.",
            len(cases),
            n,
        )

    written: list[Path] = []
    manifest = []
    for scan_id, label in cases:
        out_png = out_dir / f"{scan_id}.png"
        _render_case(scan_id, label, gt_dir, pred_dir, model_name, out_png)
        written.append(out_png)
        manifest.append({"case": scan_id, "label": label})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    log.info("QC figures: wrote %d/%d -> %s", len(written), n, out_dir)
    return written


__all__ = ["run"]
