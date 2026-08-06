#!/usr/bin/env python3
"""Write a per-abnormality-folder `reports.txt` next to the figures.

For every case rendered under `figures/<abnormality>/`, dump:
  1. the positive abnormality labels,
  2. **★ the sentences relevant to THAT abnormality, cropped out at the top** (keyword match),
  3. the full FINDINGS and IMPRESSION.

So you can read a figure and immediately see what the report actually claimed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DR = "/workspace/datasets/datasets/CT-RATE/dataset"
FIGROOT = Path("tests/abnormality_fidelity/figures")

# folder -> case-insensitive REGEX used to crop the relevant sentences.
# NOTE: the short abbreviations CTO/CTI need \b — as bare substrings they falsely match
# "se(cti)ons" / "obstru(cti)ve" / "destru(cti)ve".
KEYWORDS = {
    "cardiomegaly": r"cardiomegal|cardiothoracic|cardiac|heart|ventricl|\bcto\b|\bcti\b",
    "pleural_effusion": r"pleural|effusion",
    "emphysema": r"emphysem",
    "lung_nodule": r"nodul",
    "consolidation": r"consolidat",
    "arterial_calcification": r"calcif|atheroma|atheroscler|plaque",
    "normal_all_zero": "",
}


def sentences(text: str) -> list[str]:
    """Split report prose into sentences (period/semicolon + whitespace; keeps '4.7 cm' intact)."""
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text.strip()) if s.strip()]


def crop(text: str, pattern: str) -> list[str]:
    """Sentences matching the abnormality regex (case-insensitive)."""
    if not pattern:
        return []
    rx = re.compile(pattern, re.IGNORECASE)
    return [s for s in sentences(text) if rx.search(s)]


def main() -> None:
    manifest = json.loads((FIGROOT / "abnormality_cases.json").read_text())
    lab = pd.read_csv(f"{DR}/multi_abnormality_labels/valid_predicted_labels.csv")
    lab["id"] = lab["VolumeName"].str.replace(".nii.gz", "", regex=False)
    lab = lab.set_index("id")
    label_cols = [c for c in lab.columns if c != "VolumeName"]
    rep = pd.read_csv(f"{DR}/radiology_text_reports/validation_reports.csv")
    rep["id"] = rep["VolumeName"].str.replace(".nii.gz", "", regex=False)
    rep = rep.set_index("id")

    for folder, cases in manifest.items():
        keys = KEYWORDS.get(folder, "")
        lines: list[str] = []
        lines.append("=" * 100)
        lines.append(f"ABNORMALITY FOLDER: {folder}")
        lines.append(
            f"figures: figures/{folder}/<case>.png   (rows = GT + 4 models, cols = axial/coronal/sagittal)"
        )
        if keys:
            lines.append(f"relevance regex: {keys}")
        lines.append("=" * 100)

        for c in cases:
            cid = c["case"]
            pos = [k for k in label_cols if lab.loc[cid, k] == 1]
            findings = str(rep.loc[cid, "Findings_EN"]).strip()
            impression = str(rep.loc[cid, "Impressions_EN"]).strip()

            lines.append("")
            lines.append("-" * 100)
            lines.append(f"CASE: {cid}")
            lines.append(
                f"positive labels ({len(pos)}): {', '.join(pos) if pos else 'NONE (all-zero / normal)'}"
            )
            lines.append("-" * 100)

            # ★ cropped, abnormality-relevant sentences FIRST
            hits = crop(findings, keys) + crop(impression, keys)
            lines.append(f"[★ {folder} 관련 소견 발췌]")
            if not keys:
                lines.append(
                    "  (no target abnormality — this is an all-zero / normal case)"
                )
            elif hits:
                for h in hits:
                    lines.append(f"  • {h}")
            else:
                lines.append(
                    "  (no sentence matched the keywords — check the full text below)"
                )

            lines.append("")
            lines.append("[FINDINGS]")
            lines.append(findings)
            lines.append("")
            lines.append("[IMPRESSION]")
            lines.append(impression)

        out = FIGROOT / folder / "reports.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()
