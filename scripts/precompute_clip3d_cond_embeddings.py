#!/usr/bin/env python3
"""Precompute FrozenCLIP3D conditioning embeddings for the report2ct_CLIP3D experiment.

For each scan id:
    "Findings: {f} Impression: {i}" -> Text2CTAdapter.encode_text -> (1, 768) -> <out>/<scan_id>.pt

The saved (1, 768) L2-normalized tensor is text2ct's **exact** text condition (same
FrozenCLIP3D encoder, same "Findings: ... Impression: ..." formatting), to be grafted onto
Report2CT's UNet at train time via `src/data/fvlm_cond_datamodule.py` (the condition-shape-
agnostic `.pt` loader). Mirrors `scripts/precompute_fvlm_cond_embeddings.py`, but the
conditioner is FrozenCLIP3D and the report text comes straight from the CT-RATE reports CSV
(no decomposition).

Keyed by scan-study id (e.g. 'train_10000_a'): the report — and thus the embedding — is
shared across a scan's reconstructions, matching `scripts/build_fvlm_cond_datalist.py`,
which is reused to build the datalist.

Usage (toy-train 5000):
  CUDA_VISIBLE_DEVICES=0 python scripts/precompute_clip3d_cond_embeddings.py \\
      --ids-file    /workspace/data/ctrate_toy_v2/train/ids.json \\
      --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv \\
      --out-dir     /workspace/data/text2ct_clip3d/cond_embeddings/train --device cuda:0

Usage (valid_v2 1304):
  CUDA_VISIBLE_DEVICES=0 python scripts/precompute_clip3d_cond_embeddings.py \\
      --ids-file    /workspace/data/ctrate_toy_v2/valid_v2/ids.json \\
      --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \\
      --out-dir     /workspace/data/text2ct_clip3d/cond_embeddings/valid --device cuda:0

Requires the Text2CT CLIP3D checkpoint (see docs/text2ct_runbook.md); CUDA-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.fvlm_organ_report import scan_id_from_volume_name
from src.eval.samplers.text2ct import Text2CTSampler


def _read_vol_ids(ids_file: Path) -> list[str]:
    """Volume ids (e.g. 'train_10000_a_1') from a `.json` ({"ids": [...]} or a bare list)
    or `.txt` (one per line). Kept as vol ids (not collapsed) — the reports CSV is keyed by
    VolumeName, so the per-reconstruction id is needed to look the report up."""
    text = ids_file.read_text()
    if ids_file.suffix == ".json":
        raw = json.loads(text)
        return raw["ids"] if isinstance(raw, dict) else raw
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute FrozenCLIP3D conditioning embeddings (1, 768)"
    )
    p.add_argument(
        "--ids-file",
        required=True,
        help="vol/scan ids: .json with an 'ids' key (or bare list), or .txt one-per-line",
    )
    p.add_argument(
        "--reports-csv",
        required=True,
        help="CT-RATE radiology_text_reports/{train,validation}_reports.csv",
    )
    p.add_argument(
        "--out-dir", required=True, help="output directory for <scan_id>.pt ((1, 768))"
    )
    p.add_argument(
        "--device", default="cuda:0", help="Torch device (CUDA-only encoder)"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="process at most N scans (smoke run)"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="skip scans whose output .pt already exists",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    vol_ids = _read_vol_ids(Path(args.ids_file))
    if args.limit:
        vol_ids = vol_ids[: args.limit]

    reports_df = pd.read_csv(args.reports_csv)
    # VolumeName column contains e.g. "train_10000_a_1.nii.gz" → index by the bare vol id.
    reports_df["_id"] = reports_df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    reports_df = reports_df.set_index("_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — FrozenCLIP3D (text2ct submodule / transformers) only loads on demand.
    from src.baselines.text2ct_adapter import Text2CTAdapter

    print(f"Loading FrozenCLIP3D on {args.device} ...")
    adapter = Text2CTAdapter(device_str=args.device, load_weights=True)

    seen: set[str] = set()
    processed, skipped, missing = 0, 0, 0
    for vol_id in tqdm(vol_ids, desc="clip3d cond emb"):
        sid = scan_id_from_volume_name(vol_id)  # 'train_10000_a_1' → 'train_10000_a'
        if sid in seen:
            continue  # another reconstruction of an already-handled scan
        out_path = out_dir / f"{sid}.pt"
        if args.skip_existing and out_path.exists():
            seen.add(sid)
            skipped += 1
            continue
        if vol_id not in reports_df.index:
            # This reconstruction is absent from the CSV; a later one may carry the report,
            # so do NOT mark the scan seen — just skip this id.
            missing += 1
            continue

        row = reports_df.loc[vol_id]
        findings = str(row.get("Findings_EN", "") or "")
        impression = str(row.get("Impressions_EN", "") or "")
        text = Text2CTSampler._report_text(
            findings, impression
        )  # "Findings: ... Impression: ..."

        emb = adapter.encode_text(text)  # (1, 768) cpu float32, L2-normalized
        torch.save(emb, out_path)
        seen.add(sid)
        processed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, missing_in_csv={missing}")
    if missing:
        print(
            f"  ({missing} vol ids absent from the CSV — fine if another reconstruction "
            "of the same scan was found)",
            file=sys.stderr,
        )
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
