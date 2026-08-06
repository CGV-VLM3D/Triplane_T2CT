"""Generate the hands-on Jupyter notebooks (valid nbformat v4 JSON).

Re-runnable: rebuilds all four notebooks + README under
/workspace/tests/ctrate_eda/hands_on/. Read-only on the dataset.
"""

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HANDS_ON = "/workspace/tests/ctrate_eda/hands_on"
BUNDLE = "/workspace/tests/ctrate_eda_bundle/files"
DSET = "/workspace/datasets/datasets/CT-RATE/dataset"
MASK_EX = (
    f"{DSET}/ts_seg/ts_total/valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz"
)

FIRST_CELL = (
    "%matplotlib widget\n"
    "import sys; sys.path.insert(0, '/workspace/tests/ctrate_eda/scripts')\n"
    "from viewer import *"
)

# Bundle primary volumes per group (5 groups x 5).
GROUPS = {
    "all-zero": [
        "valid_1000_a_1",
        "valid_1010_a_1",
        "valid_1012_a_1",
        "valid_1020_a_1",
        "valid_1025_b_1",
    ],
    "lung-nodule-only": [
        "valid_1001_a_1",
        "valid_1009_a_1",
        "valid_1061_a_1",
        "valid_1077_a_1",
        "valid_1085_a_1",
    ],
    "diffuse-low-burden": [
        "valid_1022_a_1",
        "valid_1039_a_1",
        "valid_1068_a_1",
        "valid_1073_a_1",
        "valid_1153_a_1",
    ],
    "multi-abnormality": [
        "valid_1016_b_1",
        "valid_1016_d_1",
        "valid_103_a_1",
        "valid_1041_c_1",
        "valid_1078_a_1",
    ],
    "medical-material": [
        "valid_1288_a_1",
        "valid_225_b_1",
        "valid_366_a_1",
        "valid_1103_b_1",
        "valid_114_b_1",
    ],
}
# recon pairs (primary _1 / comparison _2) available in the bundle.
RECON_PAIRS = [
    ("all-zero", "valid_1000_a"),
    ("lung-nodule-only", "valid_1001_a"),
    ("diffuse-low-burden", "valid_1022_a"),
    ("multi-abnormality", "valid_1016_b"),
    ("medical-material", "valid_1288_a"),
]


def bpath(group: str, vol: str) -> str:
    return f"{BUNDLE}/{group}/{vol}.nii.gz"


def save(nb, name: str):
    nbf.validate(nb)
    with open(f"{HANDS_ON}/{name}", "w") as f:
        nbf.write(nb, f)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# 00_explore.ipynb — free exploration
# --------------------------------------------------------------------------- #
def nb_explore():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 00 — Free exploration\n\n"
            "Swap `VOL` (and optionally `MASK`) to **any** absolute `.nii.gz` "
            "path and re-run the viewer cell.\n\n"
            "**Controls**: orientation dropdown (axial/coronal/sagittal), the "
            "slice slider (live), a window preset (lung / mediastinal / bone / "
            "raw), and a *mask overlay* toggle (enabled only when `MASK` is "
            "set). **Hover** the mouse over the image to read the "
            "`(x, y, slice)` voxel **HU** value.\n\n"
            "Analysis unit: one `.nii.gz` = one *reconstruction* (volume). The "
            "report + 18 labels printed below are *scan-level*."
        ),
        new_code_cell(FIRST_CELL),
        new_code_cell(
            "# ---- set your target here ----\n"
            f"VOL  = '{bpath('multi-abnormality', 'valid_103_a_1')}'\n"
            "MASK = None   # e.g. a TotalSegmentator ts_total mask path, or None\n"
            "\n"
            "view_volume(VOL, mask_path=MASK)"
        ),
        new_markdown_cell(
            "### Try a mask overlay\n"
            "Point `MASK` at an aligned TotalSegmentator mask, then toggle "
            "*mask overlay*."
        ),
        new_code_cell(
            f"VOL  = '{bpath('all-zero', 'valid_1000_a_1')}'\n"
            f"MASK = '{MASK_EX}'\n"
            "view_volume(VOL, mask_path=MASK)"
        ),
        new_markdown_cell(
            "### Just the text\n"
            "`load_report('<volume_name>')` prints Findings / Impression + "
            "positive labels for any volume name."
        ),
        new_code_cell("load_report('valid_103_a_1')"),
    ]
    save(nb, "00_explore.ipynb")


# --------------------------------------------------------------------------- #
# 06_voxel.ipynb — 5 bundle groups + a no_chest note
# --------------------------------------------------------------------------- #
def nb_voxel():
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# 06 — Voxel appearance across bundle groups\n\n"
            "Pre-wired cells for the 5 curated bundle groups. Compare the "
            "**all-zero** (radiologically normal) appearance against the "
            "abnormal groups. Use the **lung** window for parenchyma, "
            "**mediastinal** for soft tissue / material, **bone** for the "
            "skeleton. Hover to read HU.\n\n"
            "Unit: each volume is one reconstruction; labels are scan-level."
        ),
        new_code_cell(FIRST_CELL),
    ]
    blurbs = {
        "all-zero": "**all-zero** — no positive labels; the 'normal' reference "
        "appearance. Note clean lung fields and no focal density.",
        "lung-nodule-only": "**lung-nodule-only** — a single focal soft-tissue "
        "nodule in otherwise clear lung. Use the lung window "
        "and scroll to find it.",
        "diffuse-low-burden": "**diffuse-low-burden** — subtle diffuse change "
        "(fibrotic sequela / bronchiectasis). Reticulation / "
        "traction rather than a discrete mass.",
        "multi-abnormality": "**multi-abnormality** — several co-occurring "
        "findings (lymphadenopathy, atelectasis, "
        "calcification...). Cross-check with the printed "
        "labels.",
        "medical-material": "**medical-material** — high-HU foreign material "
        "(stent / port / wires). Switch to the **bone** or "
        "**mediastinal** window to see the bright metal.",
    }
    for group, vols in GROUPS.items():
        cells.append(new_markdown_cell(f"## {blurbs[group]}"))
        cells.append(
            new_code_cell(f"# {group}: {vols}\nview_volume('{bpath(group, vols[0])}')")
        )
    cells.append(
        new_markdown_cell(
            "## no_chest example\n\n"
            "`no_chest` reconstructions (non-thoracic FOV) are **excluded from "
            "`valid_fixed`** — they are the removed recons of otherwise-present "
            "scans, so no `no_chest` NIfTI ships in the clean population or the "
            "bundle. Example: scan **valid_109_a** had recons 1/2/4 flagged "
            "`no_chest` (only 3 & 5 remain). Below we view the surviving chest "
            "recon **valid_109_a_3** to see a scan from such a study. (If you "
            "have access to a raw non-chest volume, just pass its path.)"
        )
    )
    cells.append(
        new_code_cell(
            f"view_volume('{DSET}/valid_fixed/valid_109/valid_109_a/"
            "valid_109_a_3.nii.gz')"
        )
    )
    nb.cells = cells
    save(nb, "06_voxel.ipynb")


# --------------------------------------------------------------------------- #
# 07_multimodal.ipynb — disease appearance atlas (image beside report/labels)
# --------------------------------------------------------------------------- #
def nb_multimodal():
    atlas = [
        (
            "Normal (reference)",
            "all-zero",
            "valid_1000_a_1",
            "lung",
            "No positive labels — the baseline appearance to compare against.",
        ),
        (
            "Lung nodule",
            "lung-nodule-only",
            "valid_1001_a_1",
            "lung",
            "Find the focal rounded soft-tissue density in the lung window.",
        ),
        (
            "Pulmonary fibrotic sequela",
            "diffuse-low-burden",
            "valid_1022_a_1",
            "lung",
            "Peripheral reticulation / architectural distortion.",
        ),
        (
            "Medical material",
            "medical-material",
            "valid_1288_a_1",
            "mediastinal",
            "Bright metallic material — use mediastinal/bone window.",
        ),
        (
            "Multi-abnormality (nodal/atelectasis)",
            "multi-abnormality",
            "valid_1016_b_1",
            "mediastinal",
            "Several findings at once; read the label list carefully.",
        ),
        (
            "Cardiomegaly + coronary calcification",
            "multi-abnormality",
            "valid_1078_a_1",
            "mediastinal",
            "Enlarged cardiac silhouette; bright coronary calcification.",
        ),
    ]
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# 07 — Disease appearance atlas (image + report + labels)\n\n"
            "For each disease, a viewer cell opens the volume; `view_volume` "
            "auto-prints the paired **Findings / Impression + labels** beneath "
            "the figure, so you see appearance *and* text together. The "
            "suggested window is named in each heading.\n\n"
            "Unit: reconstruction-level image; scan-level report/labels."
        ),
        new_code_cell(FIRST_CELL),
    ]
    for title, group, vol, window, hint in atlas:
        cells.append(
            new_markdown_cell(f"## {title}\n_Suggested window: **{window}**._ {hint}")
        )
        cells.append(
            new_code_cell(
                f"view_volume('{bpath(group, vol)}')\n"
                f"# tip: set the window dropdown to '{window}'"
            )
        )
    nb.cells = cells
    save(nb, "07_multimodal.ipynb")


# --------------------------------------------------------------------------- #
# 08_recon.ipynb — recon_1 vs recon_2 kernel-difference pairs
# --------------------------------------------------------------------------- #
def nb_recon():
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# 08 — Reconstruction pairs (kernel / spacing difference)\n\n"
            "`view_recon_pair(p1, p2)` shows recon **_1** vs recon **_2** of "
            "the *same scan* side-by-side. They share orientation + window but "
            "have independent slice sliders (recon_2 is often a finer-spacing / "
            "different-kernel reconstruction, so its slice count differs). "
            "Match an anatomical level in both panels and compare **edge "
            "sharpness / noise texture** — that is the kernel difference. Report "
            "and labels are identical across reconstructions of a scan."
        ),
        new_code_cell(FIRST_CELL),
    ]
    for group, scan in RECON_PAIRS:
        p1 = bpath(group, f"{scan}_1")
        p2 = bpath(group, f"{scan}_2")
        cells.append(new_markdown_cell(f"## {group} — {scan} (recon 1 vs 2)"))
        cells.append(new_code_cell(f"view_recon_pair('{p1}',\n                '{p2}')"))
    nb.cells = cells
    save(nb, "08_recon.ipynb")


def readme():
    txt = """# Hands-on CT-RATE viewer notebooks

Interactive volume/report explorer for the CT-RATE EDA bundle. Everything is
path-swappable, so you can point the viewer at any volume, mask, or report.

## Launch

```bash
cd /workspace          # so `src` imports resolve
jupyter lab            # or: jupyter notebook
```

Open a notebook and **run the first cell** (it sets `%matplotlib widget` and
imports the viewer). Then run the cells top-to-bottom. The figures are live
ipympl widgets — sliders, dropdowns, and mouse-hover HU readout only work
inside a running Jupyter kernel (not in the static rendered `.ipynb`).

Requirements (already installed): `ipympl` 0.10, `ipywidgets` 8.1.

## Notebooks

| file | what it does |
|------|--------------|
| `00_explore.ipynb` | Free exploration — set `VOL`/`MASK` to any path and view. |
| `06_voxel.ipynb`   | The 5 bundle groups (all-zero vs abnormal) + a no_chest note. |
| `07_multimodal.ipynb` | Disease atlas: each volume beside its report + labels. |
| `08_recon.ipynb`   | recon_1 vs recon_2 linked panels — kernel/spacing diff. |

## Viewer API (`from viewer import *`)

- `view_volume(path, mask_path=None)` — orientation dropdown, live slice
  slider, HU hover readout, window presets (lung/mediastinal/bone/raw),
  optional translucent mask overlay toggle. Auto-prints the report below.
- `view_recon_pair(path1, path2)` — two linked panels (shared orientation +
  window, independent slice sliders) for kernel-difference inspection.
- `load_report(volume_name)` — print Findings / Impression + positive labels
  for any volume name (e.g. `valid_1000_a_1`).

## Windows (HU level / width)

| preset | level | width | use |
|--------|------:|------:|-----|
| lung        | -600 | 1500 | parenchyma, nodules |
| mediastinal |   40 |  400 | soft tissue, material |
| bone        |  400 | 1800 | skeleton, metal |
| raw         |   —  |   —  | clipped to [-1000, 1000] HU |

## Notes

- `_fixed` NIfTI already have HU baked in — the viewer applies **no** rescale
  slope/intercept. The `-8192` out-of-FOV sentinel is mapped to the air floor.
- One `.nii.gz` = one *reconstruction*. Reports and the 18 abnormality labels
  are *scan-level* (shared across a scan's reconstructions).
- Dataset is read-only; the viewer never writes to it.
"""
    with open(f"{HANDS_ON}/README.md", "w") as f:
        f.write(txt)
    print("wrote README.md")


if __name__ == "__main__":
    os.makedirs(HANDS_ON, exist_ok=True)
    nb_explore()
    nb_voxel()
    nb_multimodal()
    nb_recon()
    readme()
    print("done")
