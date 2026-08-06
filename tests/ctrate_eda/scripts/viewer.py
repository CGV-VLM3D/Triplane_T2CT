"""Interactive hands-on CT-RATE viewer (Jupyter + ipympl).

Run in a notebook whose first cell is::

    %matplotlib widget
    import sys; sys.path.insert(0, '/workspace/tests/ctrate_eda/scripts')
    from viewer import *

Then, with any absolute NIfTI path::

    view_volume('/workspace/datasets/datasets/CT-RATE/dataset/valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz')
    view_recon_pair(path_recon1, path_recon2)
    load_report('valid_1000_a_1')

All controls are ipywidgets; the live figure needs the `%matplotlib widget`
backend (ipympl). Everything is path-swappable so the user can explore any
volume / mask / report interactively. Read-only on the dataset.

Analysis unit: a single .nii.gz = one *reconstruction* (volume). Reports and
the 18 abnormality labels are constant per *scan* (shared across its recons).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/workspace")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CT_RATE_ROOT = "/workspace/datasets/datasets/CT-RATE/dataset"
SENTINEL = -8192  # out-of-FOV padding in _fixed volumes; treat as air.
HU_CLIP = (-1000.0, 1000.0)

# (window level, window width) HU presets. "raw" = clip to HU_CLIP.
WINDOW_PRESETS: dict[str, tuple[float, float] | None] = {
    "lung": (-600.0, 1500.0),
    "mediastinal": (40.0, 400.0),
    "bone": (400.0, 1800.0),
    "raw": None,
}

# Displayed slicing axis per orientation, given a nibabel LPS array (X, Y, Z).
#   axial    -> slice along Z (axis 2)
#   coronal  -> slice along Y (axis 1)
#   sagittal -> slice along X (axis 0)
ORIENT_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume as float32 HU with the -8192 sentinel neutralised.

    Args:
        path: Absolute path to a `.nii.gz` reconstruction.

    Returns:
        ``(X, Y, Z)`` float32 array; sentinel voxels set to the air floor
        (``HU_CLIP[0]``). Raw HU otherwise (NOT windowed).
    """
    import nibabel as nib

    arr = np.asarray(nib.load(path).dataobj).astype(np.float32)  # (X, Y, Z)
    arr[arr <= SENTINEL] = HU_CLIP[0]
    return arr


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask aligned to a volume.

    Args:
        path: Absolute path to a `.nii.gz` label mask (e.g. TotalSegmentator).

    Returns:
        ``(X, Y, Z)`` int32 label array (0 = background).
    """
    import nibabel as nib

    return np.asarray(nib.load(path).dataobj).astype(np.int32)  # (X, Y, Z)


def _take_slice(vol: np.ndarray, orient: str, idx: int) -> np.ndarray:
    """Extract a 2D display slice, oriented head-up.

    Args:
        vol: ``(X, Y, Z)`` volume.
        orient: one of ``axial`` / ``coronal`` / ``sagittal``.
        idx: slice index along the orientation axis.

    Returns:
        2D array already transposed/flipped for radiological display.
    """
    axis = ORIENT_AXIS[orient]
    idx = int(np.clip(idx, 0, vol.shape[axis] - 1))
    sl = np.take(vol, idx, axis=axis)  # 2D
    # Orient so superior is up and the image is not mirrored.
    return np.flipud(sl.T)


def _window(sl: np.ndarray, preset: str) -> tuple[np.ndarray, float, float]:
    """Apply a HU window preset to a 2D slice for display.

    Returns the slice plus the (vmin, vmax) display range.
    """
    wl_ww = WINDOW_PRESETS.get(preset)
    if wl_ww is None:  # raw
        return np.clip(sl, *HU_CLIP), HU_CLIP[0], HU_CLIP[1]
    wl, ww = wl_ww
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip(sl, lo, hi), lo, hi


# --------------------------------------------------------------------------- #
# Report / label lookup
# --------------------------------------------------------------------------- #
_RECORD_CACHE: dict[str, dict] = {}


def _norm_volume_name(volume_name: str) -> str:
    """Normalise a volume name to the CSV key form (`..._1.nii.gz`)."""
    name = os.path.basename(volume_name)
    if not name.endswith(".nii.gz"):
        name = name + ".nii.gz"
    return name


def _load_split_records(split: str) -> dict:
    """Cache reports+labels for a split, keyed by VolumeName (with .nii.gz)."""
    if split not in _RECORD_CACHE:
        from src.data.ct_rate_datamodule import load_records

        _RECORD_CACHE[split] = {r.volume_name: r for r in load_records(split)}
    return _RECORD_CACHE[split]


def load_report(volume_name: str) -> None:
    """Print the paired Findings / Impression + positive labels for a volume.

    Args:
        volume_name: e.g. ``valid_1000_a_1`` or a full path / basename.
            Split (train/valid) is inferred from the name prefix. Report and
            labels are scan-level (identical across a scan's reconstructions).
    """
    key = _norm_volume_name(volume_name)
    split = "train" if key.startswith("train") else "valid"
    rec = _load_split_records(split).get(key)
    print("=" * 78)
    print(f"VOLUME: {key}   (split={split}; report/labels are scan-level)")
    print("=" * 78)
    if rec is None:
        print("  [no report/label row found — likely excluded or wrong name]")
        return
    pos = [k for k, v in (rec.labels or {}).items() if v]
    print(f"POSITIVE LABELS ({len(pos)}): " + (", ".join(pos) if pos else "(none)"))
    print(f"SPACING: xy={rec.spacing_xy}  z={rec.spacing_z}")
    print("-" * 78)
    print("FINDINGS:\n" + (rec.findings.strip() or "(empty)"))
    print("-" * 78)
    print("IMPRESSION:\n" + (rec.impression.strip() or "(empty)"))
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Single-volume interactive viewer
# --------------------------------------------------------------------------- #
def view_volume(path: str, mask_path: str | None = None):
    """Interactively view one volume: orientation, slice, HU hover, windows.

    Controls (ipywidgets): orientation Dropdown, slice IntSlider (live),
    window-preset Dropdown, and — when ``mask_path`` is given — a mask-overlay
    toggle. Mouse hover prints the ``(x, y, slice)`` voxel HU at the cursor.

    Args:
        path: Absolute path to the `.nii.gz` volume.
        mask_path: Optional aligned label mask; enables a translucent overlay.

    Returns:
        The ipywidgets container (also displayed in Jupyter).
    """
    import ipywidgets as widgets
    from IPython.display import display

    if not os.path.isfile(path):
        print(f"[view_volume] file not found: {path}")
        return None

    vol = load_volume(path)  # (X, Y, Z)
    mask = load_mask(mask_path) if (mask_path and os.path.isfile(mask_path)) else None
    if mask_path and mask is None:
        print(f"[view_volume] mask not found (ignored): {mask_path}")

    orient_dd = widgets.Dropdown(
        options=list(ORIENT_AXIS), value="axial", description="orient"
    )
    window_dd = widgets.Dropdown(
        options=list(WINDOW_PRESETS), value="lung", description="window"
    )
    axis0 = ORIENT_AXIS[orient_dd.value]
    slice_sl = widgets.IntSlider(
        min=0,
        max=vol.shape[axis0] - 1,
        value=vol.shape[axis0] // 2,
        description="slice",
        continuous_update=True,
    )
    overlay_tb = widgets.ToggleButton(
        value=False, description="mask overlay", disabled=mask is None
    )
    readout = widgets.Label(value="hover over the image to read HU")

    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.header_visible = False
    fig.canvas.toolbar_position = "right"
    im = ax.imshow(np.zeros((2, 2)), cmap="gray")
    mask_im = ax.imshow(np.zeros((2, 2)), cmap="autumn", alpha=0.0)
    plt.ion()

    def _current_slice_2d(vol_or_mask):
        return _take_slice(vol_or_mask, orient_dd.value, slice_sl.value)

    def redraw(*_):
        sl2d = _current_slice_2d(vol)
        disp, lo, hi = _window(sl2d, window_dd.value)
        im.set_data(disp)
        im.set_clim(lo, hi)
        im.set_extent([0, disp.shape[1], disp.shape[0], 0])
        ax.set_xlim(0, disp.shape[1])
        ax.set_ylim(disp.shape[0], 0)
        if mask is not None and overlay_tb.value:
            m2d = _current_slice_2d(mask)
            mask_im.set_data(np.where(m2d > 0, m2d, np.nan))
            mask_im.set_clim(1, max(2, int(mask.max())))
            mask_im.set_alpha(0.35)
            mask_im.set_extent([0, m2d.shape[1], m2d.shape[0], 0])
        else:
            mask_im.set_alpha(0.0)
        ax.set_title(
            f"{os.path.basename(path)}\n{orient_dd.value} "
            f"slice {slice_sl.value}/{slice_sl.max}"
            f"  |  window={window_dd.value}",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    def on_orient(_):
        axis = ORIENT_AXIS[orient_dd.value]
        slice_sl.max = vol.shape[axis] - 1
        slice_sl.value = min(slice_sl.value, slice_sl.max)
        redraw()

    def on_hover(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        rowdisp = int(round(event.ydata))
        sl2d = _current_slice_2d(vol)  # oriented (rows, cols)
        if 0 <= rowdisp < sl2d.shape[0] and 0 <= col < sl2d.shape[1]:
            hu = float(sl2d[rowdisp, col])
            readout.value = (
                f"x={col}  y={rowdisp}  slice={slice_sl.value}  "
                f"[{orient_dd.value}]   HU = {hu:.0f}"
            )

    orient_dd.observe(on_orient, "value")
    window_dd.observe(redraw, "value")
    slice_sl.observe(redraw, "value")
    overlay_tb.observe(redraw, "value")
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    redraw()
    controls = widgets.HBox([orient_dd, window_dd, overlay_tb])
    ui = widgets.VBox([controls, slice_sl, readout, fig.canvas])
    display(ui)
    load_report(os.path.basename(path))
    return ui


# --------------------------------------------------------------------------- #
# Linked recon-pair viewer (kernel-difference inspection)
# --------------------------------------------------------------------------- #
def view_recon_pair(path1: str, path2: str):
    """Two linked panels (recon_1 vs recon_2) sharing orientation + window.

    Each panel has its own slice slider (the two recons can differ in slice
    count / kernel); orientation and window are shared. Use to inspect the
    kernel / spacing difference between reconstructions of the same scan.

    Args:
        path1: Absolute path to reconstruction 1.
        path2: Absolute path to reconstruction 2.

    Returns:
        The ipywidgets container (also displayed in Jupyter).
    """
    import ipywidgets as widgets
    from IPython.display import display

    for p in (path1, path2):
        if not os.path.isfile(p):
            print(f"[view_recon_pair] file not found: {p}")
            return None

    v1, v2 = load_volume(path1), load_volume(path2)

    orient_dd = widgets.Dropdown(
        options=list(ORIENT_AXIS), value="axial", description="orient"
    )
    window_dd = widgets.Dropdown(
        options=list(WINDOW_PRESETS), value="lung", description="window"
    )

    def _mk_slider(v):
        ax = ORIENT_AXIS[orient_dd.value]
        return widgets.IntSlider(
            min=0, max=v.shape[ax] - 1, value=v.shape[ax] // 2, description="slice"
        )

    s1, s2 = _mk_slider(v1), _mk_slider(v2)
    readout = widgets.Label(value="hover over either panel to read HU")

    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.canvas.header_visible = False
    im1 = ax1.imshow(np.zeros((2, 2)), cmap="gray")
    im2 = ax2.imshow(np.zeros((2, 2)), cmap="gray")
    plt.ion()

    def _panel(vol, im, axp, slider, tag):
        sl2d = _take_slice(vol, orient_dd.value, slider.value)
        disp, lo, hi = _window(sl2d, window_dd.value)
        im.set_data(disp)
        im.set_clim(lo, hi)
        im.set_extent([0, disp.shape[1], disp.shape[0], 0])
        axp.set_xlim(0, disp.shape[1])
        axp.set_ylim(disp.shape[0], 0)
        axp.set_title(f"{tag}\n{os.path.basename(_panel.paths[tag])}", fontsize=9)

    _panel.paths = {"recon_1": path1, "recon_2": path2}

    def redraw(*_):
        _panel(v1, im1, ax1, s1, "recon_1")
        _panel(v2, im2, ax2, s2, "recon_2")
        fig.suptitle(
            f"kernel-diff  |  orient={orient_dd.value}  window={window_dd.value}",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    def on_orient(_):
        for v, s in ((v1, s1), (v2, s2)):
            ax = ORIENT_AXIS[orient_dd.value]
            s.max = v.shape[ax] - 1
            s.value = min(s.value, s.max)
        redraw()

    def on_hover(event):
        if event.xdata is None:
            return
        if event.inaxes is ax1:
            vol, slider, axp, tag = v1, s1, ax1, "recon_1"
        elif event.inaxes is ax2:
            vol, slider, axp, tag = v2, s2, ax2, "recon_2"
        else:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        sl2d = _take_slice(vol, orient_dd.value, slider.value)
        if 0 <= row < sl2d.shape[0] and 0 <= col < sl2d.shape[1]:
            readout.value = (
                f"{tag}: x={col} y={row} slice={slider.value} "
                f"[{orient_dd.value}]  HU = {float(sl2d[row, col]):.0f}"
            )

    orient_dd.observe(on_orient, "value")
    window_dd.observe(redraw, "value")
    s1.observe(redraw, "value")
    s2.observe(redraw, "value")
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    redraw()
    ui = widgets.VBox(
        [
            widgets.HBox([orient_dd, window_dd]),
            widgets.HBox(
                [
                    widgets.VBox([widgets.Label("recon_1"), s1]),
                    widgets.VBox([widgets.Label("recon_2"), s2]),
                ]
            ),
            readout,
            fig.canvas,
        ]
    )
    display(ui)
    return ui


__all__ = [
    "view_volume",
    "view_recon_pair",
    "load_report",
    "load_volume",
    "load_mask",
    "CT_RATE_ROOT",
    "WINDOW_PRESETS",
]
