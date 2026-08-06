"""Stage 3 (main env): compose GT | MAISI | Wan montages from the stage-1/2 npz.

Per patient, two PNGs under out/:
  * <id>_lung.png       — 3 axial levels x {full, zoom} x {GT, MAISI, Wan}, lung window,
                          with the zoom box drawn on the full panels.
  * <id>_medi.png       — 3 axial levels x {GT, MAISI, Wan}, mediastinal window.

Run: python tests/vae_recon_compare/make_montage.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rootutils  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tests.vae_recon_compare import common as C  # noqa: E402

METHODS = ("gt", "maisi", "wan")
TITLES = {"gt": "GT", "maisi": "MAISI recon", "wan": "Wan recon"}
DIFF_HU = 250.0  # symmetric HU range for the (recon - GT) diverging diff maps
DIFF_SIGMA = 1.0  # low-pass both before diffing: kills per-pixel CT-noise realization
#                   mismatch (not a detail-loss signal), leaving multi-pixel structural
#                   smearing (vessels/edges) — the thing we actually want to compare.


def _load(sid: str) -> dict[str, np.ndarray]:
    mg = np.load(C.OUT_DIR / f"{sid}_maisi_gt.npz")
    wn = np.load(C.OUT_DIR / f"{sid}_wan.npz")
    return {
        "gt": mg["gt"],
        "maisi": mg["maisi"],
        "wan": wn["wan"],
    }  # each (n_frac,512,512) HU


def _imshow(ax, img01: np.ndarray) -> None:
    ax.imshow(img01, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _suptitle(sid: str, p: dict) -> str:
    if p["positive"]:
        return f"{sid}   |   positive: {', '.join(p['positive'])}"
    return f"{sid}   |   HEALTHY control (no positive findings)"


def lung_figure(sid: str, p: dict, sl: dict[str, np.ndarray]) -> None:
    r0, r1, c0, c1 = C.crop_box()
    nf = len(C.Z_FRACTIONS)
    fig, axes = plt.subplots(nf, 6, figsize=(6 * 2.4, nf * 2.4))
    for i, frac in enumerate(C.Z_FRACTIONS):
        for j, m in enumerate(METHODS):
            full = C.apply_window(sl[m][i], "lung")
            _imshow(axes[i, j], full)
            axes[i, j].add_patch(
                mpatches.Rectangle(
                    (c0, r0), c1 - c0, r1 - r0, ec="yellow", fc="none", lw=1.2
                )
            )
            _imshow(axes[i, j + 3], full[r0:r1, c0:c1])  # zoom
        axes[i, 0].set_ylabel(f"z={frac:.2f}", fontsize=11)
    for j, m in enumerate(METHODS):
        axes[0, j].set_title(TITLES[m], fontsize=12)
        axes[0, j + 3].set_title(f"{TITLES[m]} (zoom)", fontsize=12)
    fig.suptitle(_suptitle(sid, p) + "   —   lung window", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(C.OUT_DIR / f"{sid}_lung.png", dpi=110)
    plt.close(fig)


def medi_figure(sid: str, p: dict, sl: dict[str, np.ndarray]) -> None:
    nf = len(C.Z_FRACTIONS)
    fig, axes = plt.subplots(nf, 3, figsize=(3 * 2.6, nf * 2.6))
    for i, frac in enumerate(C.Z_FRACTIONS):
        for j, m in enumerate(METHODS):
            _imshow(axes[i, j], C.apply_window(sl[m][i], "mediastinal"))
        axes[i, 0].set_ylabel(f"z={frac:.2f}", fontsize=11)
    for j, m in enumerate(METHODS):
        axes[0, j].set_title(TITLES[m], fontsize=12)
    fig.suptitle(_suptitle(sid, p) + "   —   mediastinal window", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(C.OUT_DIR / f"{sid}_medi.png", dpi=110)
    plt.close(fig)


def diff_figure(sid: str, p: dict, sl: dict[str, np.ndarray]) -> None:
    """GT (lung) reference + (recon - GT) diverging diff maps, full + zoom, per axial level.

    Diff is in raw HU (physically meaningful); a diverging colormap centered at 0 makes
    blur show as a bright/dark dipole along vessel/edge boundaries — larger, more structured
    residual = more detail loss.
    """
    r0, r1, c0, c1 = C.crop_box()
    nf = len(C.Z_FRACTIONS)
    fig, axes = plt.subplots(nf, 5, figsize=(5 * 2.4, nf * 2.4))
    from scipy.ndimage import gaussian_filter

    im = None
    for i, frac in enumerate(C.Z_FRACTIONS):
        gt = sl["gt"][i]
        gt_s = gaussian_filter(gt, DIFF_SIGMA)  # low-passed GT (structural reference)
        _imshow(axes[i, 0], C.apply_window(gt, "lung"))  # grayscale GT reference
        for k, (m, col_full, col_zoom) in enumerate([("maisi", 1, 3), ("wan", 2, 4)]):
            d = (
                gaussian_filter(sl[m][i], DIFF_SIGMA) - gt_s
            )  # (512,512) HU structural residual
            im = axes[i, col_full].imshow(
                d, cmap="seismic", vmin=-DIFF_HU, vmax=DIFF_HU, interpolation="nearest"
            )
            axes[i, col_full].set_xticks([])
            axes[i, col_full].set_yticks([])
            axes[i, col_zoom].imshow(
                d[r0:r1, c0:c1],
                cmap="seismic",
                vmin=-DIFF_HU,
                vmax=DIFF_HU,
                interpolation="nearest",
            )
            axes[i, col_zoom].set_xticks([])
            axes[i, col_zoom].set_yticks([])
        axes[i, 0].set_ylabel(f"z={frac:.2f}", fontsize=11)
    for col, t in [
        (0, "GT (lung)"),
        (1, "MAISI−GT"),
        (2, "Wan−GT"),
        (3, "MAISI−GT (zoom)"),
        (4, "Wan−GT (zoom)"),
    ]:
        axes[0, col].set_title(t, fontsize=12)
    fig.suptitle(
        _suptitle(sid, p)
        + f"   —   recon − GT structural residual (σ={DIFF_SIGMA}px low-pass, ±{DIFF_HU:.0f} HU)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="HU (recon − GT)")
    fig.savefig(C.OUT_DIR / f"{sid}_diff.png", dpi=110)
    plt.close(fig)


def detail_figure(sid: str, p: dict, sl: dict[str, np.ndarray]) -> None:
    """Edge-strength (|∇| gaussian-gradient-magnitude) maps + recon−GT edge deficit, zoomed.

    Blur = loss of high-frequency energy. Gradient magnitude measures local edge strength and
    is (unlike an intensity diff) insensitive to sub-pixel misregistration — a shifted-but-sharp
    edge keeps full |∇|, only a *smeared* edge weakens it. So the deficit ``|∇recon| − |∇GT|``
    goes negative (blue) exactly where the recon smears detail. All panels are the zoom crop
    (where fine vessels/texture live).
    """
    from scipy.ndimage import gaussian_gradient_magnitude

    r0, r1, c0, c1 = C.crop_box()
    edge_vmax, deficit = 180.0, 120.0  # HU/px display scales
    nf = len(C.Z_FRACTIONS)
    fig, axes = plt.subplots(nf, 5, figsize=(5 * 2.4, nf * 2.4))
    im = None
    for i, frac in enumerate(C.Z_FRACTIONS):
        gm = {
            m: gaussian_gradient_magnitude(sl[m][i], 1.5) for m in METHODS
        }  # (512,512) HU/px
        for col, m in [(0, "gt"), (1, "maisi"), (2, "wan")]:
            axes[i, col].imshow(
                gm[m][r0:r1, c0:c1],
                cmap="magma",
                vmin=0,
                vmax=edge_vmax,
                interpolation="nearest",
            )
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
        for col, m in [(3, "maisi"), (4, "wan")]:
            d = (
                gm[m] - gm["gt"]
            )  # (512,512) edge-strength deficit; blue<0 = detail lost
            im = axes[i, col].imshow(
                d[r0:r1, c0:c1],
                cmap="seismic",
                vmin=-deficit,
                vmax=deficit,
                interpolation="nearest",
            )
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
        axes[i, 0].set_ylabel(f"z={frac:.2f}", fontsize=11)
    for col, t in [
        (0, "GT edges"),
        (1, "MAISI edges"),
        (2, "Wan edges"),
        (3, "MAISI−GT edge Δ"),
        (4, "Wan−GT edge Δ"),
    ]:
        axes[0, col].set_title(t, fontsize=12)
    fig.suptitle(
        _suptitle(sid, p)
        + "   —   edge strength |∇| (zoom); Δ blue = detail smeared away",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.colorbar(
        im, ax=axes, fraction=0.015, pad=0.01, label="|∇| deficit (recon − GT), HU/px"
    )
    fig.savefig(C.OUT_DIR / f"{sid}_detail.png", dpi=110)
    plt.close(fig)


def main() -> None:
    patients = json.loads((C.OUT_DIR / "manifest.json").read_text())
    for p in patients:
        sid = p["id"]
        sl = _load(sid)
        lung_figure(sid, p, sl)
        medi_figure(sid, p, sl)
        diff_figure(sid, p, sl)
        detail_figure(sid, p, sl)
        print(f"  montage {sid}")


if __name__ == "__main__":
    main()
