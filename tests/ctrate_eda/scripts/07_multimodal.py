"""EDA v2 — Module 07: multimodal per-case sheets (GPT §7).

For each of the 5 bundle groups, builds a per-case Markdown sheet that fuses:
  - identity (patient / scan / reconstruction) + metadata join
    (age / sex / manufacturer / model / kernel / spacing / shape)
  - the 18-label positive set + free-text Findings / Impression
  - a 6-panel montage (axial / coronal / sagittal x lung / mediastinal window)
  - NegEx assertion status (AFFIRMED / NEGATED / UNCERTAIN / ABSENT) of each
    label's trigger term(s) inside the report text
  - a per-case 'human review needed' checklist (pointers only — NO medical
    adjudication)

Also assembles a report<->label MISMATCH-SUSPECTS section:
  - FP-suspect : positive silver-label whose term is not AFFIRMED in the report
  - FN-suspect : an AFFIRMED disease term whose label is negative

The NegEx here is a compact, transparent re-implementation of Chapman's
rule-based algorithm (pre-/post-/pseudo-negation + uncertainty triggers over a
char-window scope). It is a POINTER tool, not a validated clinical NLP system.

Read-only on /workspace/datasets. Outputs under /workspace/tests/ctrate_eda/.
Outputs:
  case_sheets/<group>__<scan>.md      (25 primary cases)
  case_sheets/INDEX.md                (TOC + quick-scan grid)
  case_sheets/mismatch_suspects.md    (aggregate FP/FN suspects)
  figures/montages/<group>__<scan>.png
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

BUNDLE = Path("/workspace/tests/ctrate_eda_bundle")
MANIFEST = BUNDLE / "manifest.csv"
META_CSV = Path(
    "/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv"
)
LABELS_CSV = Path(
    "/workspace/datasets/datasets/CT-RATE/dataset"
    "/multi_abnormality_labels/valid_predicted_labels.csv"
)
OUT = Path("/workspace/tests/ctrate_eda")
SHEETS = OUT / "case_sheets"
MONT = OUT / "figures" / "montages"

# ------------------------------------------------------------------ NegEx ----
# Trigger terms per label. Kept deliberately broad (synonyms/word-stems) so a
# missing affirmation is a *pointer to verify*, not a verdict. Regex, IGNORECASE.
LABEL_TERMS: dict[str, list[str]] = {
    "Medical material": [
        r"stent",
        r"catheter",
        r"\bport\b",
        r"pacemaker",
        r"surgical (?:material|clip)",
        r"\bclip\b",
        r"prosthes",
        r"\bdrain\b",
        r"sternal (?:wire|suture)",
        r"\bsuture",
        r"\bcoil\b",
        r"\bscrew\b",
        r"\bplate\b",
        r"\bgraft\b",
        r"medical material",
        r"chest tube",
        r"nasogastric",
        r"\bdevice\b",
        r"foreign (?:body|material)",
    ],
    "Arterial wall calcification": [
        r"arterial wall calcification",
        r"aortic calcif",
        r"vascular calcif",
        r"atherosclerotic calcif",
        r"calcified aort",
        r"aorta.{0,20}calcif",
        r"wall calcification",
    ],
    "Cardiomegaly": [
        r"cardiomegal",
        r"enlarged heart",
        r"cardiac enlargement",
        r"increased cardiothoracic",
        r"heart .{0,15}enlarg",
    ],
    "Pericardial effusion": [r"pericardial effusion", r"pericardial fluid"],
    "Coronary artery wall calcification": [
        r"coronary artery.{0,25}calcif",
        r"coronary calcif",
        r"calcif.{0,15}coronary",
    ],
    "Hiatal hernia": [r"hiatal hernia", r"hiatus hernia"],
    "Lymphadenopathy": [
        r"lymphadenopath",
        r"enlarged lymph node",
        r"lymph node.{0,15}enlarg",
        r"pathologic.{0,25}lymph",
        r"lymph node.{0,20}pathologic",
    ],
    "Emphysema": [r"emphysemat", r"emphysema"],
    "Atelectasis": [r"atelecta", r"\bcollapse"],
    "Lung nodule": [r"nodul"],
    "Lung opacity": [
        r"opacit",
        r"ground.?glass",
        r"\binfiltrat",
        r"consolidat",
        r"density",
    ],
    "Pulmonary fibrotic sequela": [
        r"fibroti",
        r"fibrosis",
        r"sequela",
        r"scar",
    ],
    "Pleural effusion": [r"pleural effusion", r"pleural fluid"],
    "Mosaic attenuation pattern": [
        r"mosaic attenuation",
        r"mosaic perfusion",
        r"mosaic pattern",
    ],
    "Peribronchial thickening": [
        r"peribronchial thicken",
        r"bronchial wall thicken",
        r"peribronchial",
    ],
    "Consolidation": [r"consolidat"],
    "Bronchiectasis": [r"bronchiecta"],
    "Interlobular septal thickening": [
        r"interlobular septal thicken",
        r"septal thicken",
        r"interlobular septal",
    ],
}

PSEUDO = [
    r"not excluded",
    r"cannot be excluded",
    r"can not be excluded",
    r"not entirely excluded",
    r"not necessarily",
    r"without difficulty",
    r"no significant change",
    r"no interval change",
    r"no change",
]
# Pre-negation cues (appear BEFORE the finding).
PRE_NEG = [
    r"\bno\b",
    r"\bnot\b",
    r"without",
    r"absence of",
    r"negative for",
    r"free of",
    r"neither",
    r"ruled out",
    r"no evidence",
    r"no sign",
]
# Post-negation cues (appear AFTER the finding) — Turkish-radiology-report style.
POST_NEG = [
    r"was not (?:detected|observed|seen|identified|present)",
    r"were not (?:detected|observed|seen|identified|present)",
    r"not detected",
    r"not observed",
    r"not seen",
    r"not identified",
    r"is not present",
    r"are not present",
    r"was not evident",
    r"were not evident",
    r"is absent",
    r"are absent",
]
UNCERTAIN = [
    r"suspic",
    r"suspected",
    r"possibl",
    r"probabl",
    r"likely",
    r"\bmay\b",
    r"questionable",
    r"concerning for",
    r"compatible with",
    r"consistent with",
    r"cannot be excluded",
    r"\?",
]

_PRE = re.compile("|".join(PRE_NEG), re.I)
_POST = re.compile("|".join(POST_NEG), re.I)
_PSEUDO = re.compile("|".join(PSEUDO), re.I)
_UNC = re.compile("|".join(UNCERTAIN), re.I)


def split_sentences(text: str) -> list[str]:
    """Naive clause/sentence split (period, semicolon, newline)."""
    if not isinstance(text, str):
        return []
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def assert_status(report: str, terms: list[str]) -> dict:
    """Chapman-style NegEx over `report` for any of `terms`.

    Returns dict(status, evidence): status in {AFFIRMED, NEGATED, UNCERTAIN,
    ABSENT}. AFFIRMED wins over NEGATED/UNCERTAIN if any single mention is
    affirmed (a disease affirmed once is present regardless of other negated
    mentions of unrelated structures).
    """
    pat = re.compile("|".join(terms), re.I)
    found = {"AFFIRMED": [], "NEGATED": [], "UNCERTAIN": []}
    for sent in split_sentences(report):
        for m in pat.finditer(sent):
            s, e = m.span()
            pre = sent[max(0, s - 100) : s]
            post = sent[e : e + 70]
            local = sent[max(0, s - 100) : e + 70]
            if _PSEUDO.search(local):
                # pseudo-negation → the phrase does NOT negate → treat mention
                # as uncertain rather than negated
                found["UNCERTAIN"].append(sent)
                continue
            if _PRE.search(pre) or _POST.search(post):
                found["NEGATED"].append(sent)
            elif _UNC.search(local):
                found["UNCERTAIN"].append(sent)
            else:
                found["AFFIRMED"].append(sent)
    for status in ("AFFIRMED", "UNCERTAIN", "NEGATED"):
        if found[status]:
            return {"status": status, "evidence": found[status][0]}
    return {"status": "ABSENT", "evidence": ""}


# --------------------------------------------------------------- montage -----
def window(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Apply a HU window (level/width) and scale to [0,1]."""
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def montage(path: Path, out_png: Path, title: str, spacing: tuple[float, float, float]):
    """6-panel montage: rows=[lung, mediastinal], cols=[axial, coronal, sagittal]."""
    vol = np.asarray(nib.load(str(path)).dataobj).astype(np.float32)
    vol = np.clip(vol, -1000, 1000)  # -8192 padding sentinel → air
    nx, ny, nz = vol.shape
    ax = vol[:, :, nz // 2]  # (X, Y)
    cor = vol[:, ny // 2, :]  # (X, Z)
    sag = vol[nx // 2, :, :]  # (Y, Z)
    sx, sy, sz = spacing
    # z/xy aspect so out-of-plane slices are not squashed
    asp_cor = sz / sx if sx else 1.0
    asp_sag = sz / sy if sy else 1.0

    windows = [
        ("Lung (WL-600/WW1500)", -600.0, 1500.0),
        ("Mediastinal (WL40/WW400)", 40.0, 400.0),
    ]
    views = [("Axial", ax, 1.0), ("Coronal", cor, asp_cor), ("Sagittal", sag, asp_sag)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, (wname, wl, ww) in enumerate(windows):
        for c, (vname, sl, asp) in enumerate(views):
            a = axes[r][c]
            disp = np.rot90(window(sl, wl, ww))
            a.imshow(disp, cmap="gray", aspect=asp, vmin=0, vmax=1)
            a.set_title(f"{vname} | {wname}", fontsize=8)
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main -----
def norm_age(a) -> str:
    if not isinstance(a, str) or not a.strip():
        return "n/a"
    m = re.match(r"0*(\d+)", a)
    return f"{int(m.group(1))}Y" if m else a


def main() -> None:
    SHEETS.mkdir(parents=True, exist_ok=True)
    MONT.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "primary"]
    meta = pd.read_csv(META_CSV).set_index("VolumeName")
    labels_df = pd.read_csv(LABELS_CSV).set_index("VolumeName")
    LABELS = [c for c in labels_df.columns]

    index_rows = []
    mismatch_fp = []  # positive label, term not affirmed
    mismatch_fn = []  # affirmed disease term, label negative

    for row in sorted(rows, key=lambda r: (r["group"], r["filename"])):
        group = row["group"]
        fname = row["filename"]
        vol_name = fname.replace(".nii.gz", "")
        parts = vol_name.split("_")
        patient = "_".join(parts[:2])
        scan = "_".join(parts[:3])
        recon = parts[3]
        stem = f"{group}__{scan}"

        # --- metadata join ---
        mrow = meta.loc[fname] if fname in meta.index else None
        if mrow is not None:
            man = str(mrow.get("Manufacturer", "n/a"))
            model = str(mrow.get("ManufacturerModelName", "n/a"))
            sex = str(mrow.get("PatientSex", "n/a"))
            age = norm_age(str(mrow.get("PatientAge", "")))
            kernel = str(mrow.get("ConvolutionKernel", "n/a"))
        else:
            man = model = sex = age = kernel = "n/a"

        shape = row["shape"]
        spacing_str = row["voxel_spacing"]
        sx, sy, sz = (float(x) for x in spacing_str.split("x"))
        findings = row["Findings_EN"] or ""
        impression = row["Impressions_EN"] or ""
        report_full = f"{findings}\n{impression}"

        # positive labels from CSV (authoritative)
        if fname in labels_df.index:
            pos = [lab for lab in LABELS if int(labels_df.loc[fname, lab]) == 1]
        else:
            pos = []

        # --- NegEx status for all 18 labels ---
        status_by_label = {
            lab: assert_status(report_full, LABEL_TERMS[lab]) for lab in LABELS
        }

        # --- mismatch detection (this case) ---
        case_fp, case_fn = [], []
        for lab in LABELS:
            st = status_by_label[lab]["status"]
            is_pos = lab in pos
            if is_pos and st != "AFFIRMED":
                case_fp.append((lab, st))
                mismatch_fp.append((group, scan, lab, st))
            if (not is_pos) and st == "AFFIRMED":
                case_fn.append((lab, status_by_label[lab]["evidence"]))
                mismatch_fn.append((group, scan, lab, status_by_label[lab]["evidence"]))

        # --- montage ---
        png = MONT / f"{stem}.png"
        montage(
            BUNDLE / row["bundle_path"],
            png,
            title=f"{scan} recon {recon} | {group} | {shape} @ {spacing_str}",
            spacing=(sx, sy, sz),
        )

        # --- write case sheet ---
        rel_png = f"../figures/montages/{stem}.png"
        lines: list[str] = []
        lines.append(f"# Case sheet — `{scan}` (recon {recon}) — group: **{group}**\n")
        lines.append("## Identity & acquisition\n")
        lines.append("| field | value |")
        lines.append("|---|---|")
        lines.append(f"| patient | `{patient}` |")
        lines.append(f"| scan | `{scan}` |")
        lines.append(
            f"| reconstruction | `{recon}`  (available: {row['available_reconstruction_ids']}) |"
        )
        lines.append(f"| primary file | `{fname}` |")
        lines.append(f"| age / sex | {age} / {sex} |")
        lines.append(f"| manufacturer / model | {man} / {model} |")
        lines.append(f"| kernel | {kernel} |")
        lines.append(f"| shape (RxCxSlices) | {shape} |")
        lines.append(f"| voxel spacing (x,y,z mm) | {spacing_str} |")
        lines.append(
            f"| intensity min/med/max (HU) | {row['intensity_min']} / {row['intensity_median']} / {row['intensity_max']} |"
        )
        lines.append(f"| no_chest excluded set | {row['no_chest_included']} |\n")

        lines.append("## Montage (axial / coronal / sagittal x lung / mediastinal)\n")
        lines.append(f"![montage]({rel_png})\n")
        lines.append(
            "_Analysis unit: single reconstruction (volume). "
            "Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._\n"
        )

        lines.append("## Positive silver-labels (from *_predicted_labels.csv)\n")
        lines.append(
            "(none)" if not pos else "- " + "\n- ".join(f"**{p}**" for p in pos)
        )
        lines.append("")

        lines.append("## Findings (Findings_EN)\n")
        lines.append(f"> {findings if findings else '_(empty)_'}\n")
        lines.append("## Impression (Impressions_EN)\n")
        lines.append(f"> {impression if impression else '_(empty)_'}\n")

        lines.append("## NegEx assertion status of each label's term in the report\n")
        lines.append("| label | in CSV? | NegEx status | evidence sentence |")
        lines.append("|---|:---:|:---:|---|")
        for lab in LABELS:
            st = status_by_label[lab]
            incsv = "✅" if lab in pos else "—"
            ev = st["evidence"].replace("|", "\\|")
            ev = (ev[:120] + "…") if len(ev) > 120 else ev
            lines.append(f"| {lab} | {incsv} | **{st['status']}** | {ev} |")
        lines.append("")

        lines.append(
            "## Human review needed (pointers only — NO medical adjudication)\n"
        )
        chk = []
        chk.append(
            f"- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`{stem}.png`)."
        )
        chk.append(
            f"- [ ] Verify spacing/shape are plausible for a chest CT (spacing {spacing_str}, shape {shape})."
        )
        if row["intensity_min"] and int(float(row["intensity_min"])) <= -2000:
            chk.append(
                f"- [ ] Verify HU: raw min = {row['intensity_min']} (−8192 padding sentinel expected; stats clip to [−1000,1000])."
            )
        if age == "n/a" or sex in ("n/a", "nan", ""):
            chk.append("- [ ] Verify missing demographic (age/sex absent in metadata).")
        if group == "all-zero" and pos:
            chk.append(
                "- [ ] Verify: group is 'all-zero' yet CSV lists positive labels (regroup or check)."
            )
        if group == "all-zero" and not pos:
            chk.append(
                "- [ ] Verify truly-negative: scan is label-negative; confirm no subtle finding was missed."
            )
        for lab, st in case_fp:
            chk.append(
                f"- [ ] Verify label **{lab}**: positive in CSV but NegEx='{st}' in report (possible silver-label FP or synonym miss)."
            )
        for lab, ev in case_fn:
            short = (ev[:90] + "…") if len(ev) > 90 else ev
            chk.append(
                f'- [ ] Verify **{lab}**: AFFIRMED in report but label negative (possible silver-label FN). Ev: "{short}"'
            )
        lines.extend(chk)
        lines.append("")

        (SHEETS / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")

        index_rows.append(
            {
                "group": group,
                "scan": scan,
                "recon": recon,
                "sheet": f"{stem}.md",
                "n_pos": len(pos),
                "n_fp_suspect": len(case_fp),
                "n_fn_suspect": len(case_fn),
                "age": age,
                "sex": sex,
                "manufacturer": man,
                "shape": shape,
            }
        )

    # ------------------------------------------------------------- INDEX ----
    idx = [
        "# Case-sheet INDEX — 25 primary bundle cases (module 07)\n",
        "Analysis unit: **single reconstruction (volume)**. "
        "Labels are per-scan silver labels; NegEx is a heuristic pointer tool.\n",
        "## Table of contents\n",
    ]
    for g in (
        "all-zero",
        "lung-nodule-only",
        "diffuse-low-burden",
        "multi-abnormality",
        "medical-material",
    ):
        idx.append(f"### {g}")
        for r in index_rows:
            if r["group"] == g:
                idx.append(
                    f"- [{r['scan']} (recon {r['recon']})]({r['sheet']}) "
                    f"— {r['n_pos']} pos labels, "
                    f"{r['n_fp_suspect']} FP-susp, {r['n_fn_suspect']} FN-susp"
                )
        idx.append("")
    idx.append("## Quick-scan grid\n")
    idx.append(
        "| group | scan | recon | age | sex | manufacturer | shape | #pos | #FP-susp | #FN-susp | sheet |"
    )
    idx.append("|---|---|:---:|---|:---:|---|---|:---:|:---:|:---:|---|")
    for r in index_rows:
        idx.append(
            f"| {r['group']} | `{r['scan']}` | {r['recon']} | {r['age']} | {r['sex']} "
            f"| {r['manufacturer']} | {r['shape']} | {r['n_pos']} | {r['n_fp_suspect']} "
            f"| {r['n_fn_suspect']} | [md]({r['sheet']}) |"
        )
    idx.append("")
    idx.append(
        f"**Totals:** {len(index_rows)} cases · "
        f"{sum(r['n_fp_suspect'] for r in index_rows)} FP-suspect label-cells · "
        f"{sum(r['n_fn_suspect'] for r in index_rows)} FN-suspect label-cells.\n"
    )
    (SHEETS / "INDEX.md").write_text("\n".join(idx), encoding="utf-8")

    # ------------------------------------------------- MISMATCH SUSPECTS ----
    ms = [
        "# Report <-> label mismatch suspects (module 07)\n",
        "Heuristic NegEx over the 30-volume bundle's **25 primary cases**. "
        "These are POINTERS to verify — not confirmed errors. Silver labels are "
        "RadBERT-mined; the NegEx term dictionary is broad (synonym stems), so "
        "both directions carry false alarms.\n",
        "## A. FP-suspects — positive silver-label whose term is NOT affirmed in the report\n",
        "(positive label + NegEx status ∈ {NEGATED, UNCERTAIN, ABSENT})\n",
        "| group | scan | label | NegEx status |",
        "|---|---|---|:---:|",
    ]
    for g, s, lab, st in mismatch_fp:
        ms.append(f"| {g} | `{s}` | {lab} | {st} |")
    ms.append(
        f"\n**FP-suspect count:** {len(mismatch_fp)} label-cells "
        f"across {len({(g, s) for g, s, _, _ in mismatch_fp})} scans.\n"
    )
    ms.append("## B. FN-suspects — AFFIRMED disease term whose label is negative\n")
    ms.append("| group | scan | label | evidence sentence |")
    ms.append("|---|---|---|---|")
    for g, s, lab, ev in mismatch_fn:
        ev = ev.replace("|", "\\|")
        ev = (ev[:140] + "…") if len(ev) > 140 else ev
        ms.append(f"| {g} | `{s}` | {lab} | {ev} |")
    ms.append(
        f"\n**FN-suspect count:** {len(mismatch_fn)} label-cells "
        f"across {len({(g, s) for g, s, _, _ in mismatch_fn})} scans.\n"
    )
    ms.append("### Caveats\n")
    ms.append(
        "- NegEx is char-window heuristic; long negated lists, cross-sentence "
        "scope, and Turkish→EN template phrasing cause misses.\n"
        "- Overlapping terms (Lung opacity ⊃ consolidation/infiltrate/ground-glass) "
        "inflate FN-suspects for co-defined labels.\n"
        "- Labels are per-**scan**; report is per-**scan**; montage is one **volume**.\n"
        "- 'Medical material' & 'Lung opacity' have the broadest term sets → most noise."
    )
    (SHEETS / "mismatch_suspects.md").write_text("\n".join(ms), encoding="utf-8")

    # ------------------------------------------------------------- report ---
    print(f"Wrote {len(index_rows)} case sheets to {SHEETS}")
    print(f"Montages: {len(list(MONT.glob('*.png')))} PNGs in {MONT}")
    print(
        f"FP-suspect cells: {len(mismatch_fp)} | FN-suspect cells: {len(mismatch_fn)}"
    )
    print("INDEX.md, mismatch_suspects.md written.")


if __name__ == "__main__":
    main()
