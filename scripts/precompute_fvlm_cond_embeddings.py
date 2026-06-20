#!/usr/bin/env python3
"""Precompute per-organ fVLM conditioning embeddings for ctgen training/eval.

For each scan id:
    build_organ_text -> FVLMBackbone.encode_organ_texts -> (4, 256) -> <out>/<scan_id>.pt

The saved (4, 256) tensor (row order = `src.data.fvlm_organ_report.ORGANS` =
lung/heart/esophagus/aorta) is the **frozen** text condition consumed at train time
by `src/data/fvlm_cond_datamodule.py`. Mirrors the role of
`scripts/precompute_report2ct_text_embeddings.py` (which precomputes the 3-encoder
2560-d findings/impression vectors) — here the conditioner is fVLM instead.

Usage (toy-train 5000; released author decomposition):
  CUDA_VISIBLE_DEVICES=0 python scripts/precompute_fvlm_cond_embeddings.py \\
      --ids-file    /workspace/data/ctrate_toy_v2/train/ids.json \\
      --report-root /workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report \\
      --out-dir     /workspace/data/fVLM/cond_embeddings/train --device cuda:0

Usage (valid_v2 1304; AFTER the Qwen decomposition of scripts/decompose_reports_fvlm.py):
  CUDA_VISIBLE_DEVICES=0 python scripts/precompute_fvlm_cond_embeddings.py \\
      --ids-file    /workspace/data/ctrate_toy_v2/valid_v2/ids.json \\
      --report-root /workspace/data/fVLM/decomposed_report_valid \\
      --out-dir     /workspace/data/fVLM/cond_embeddings/valid_v2 --device cuda:0

Requires the fVLM CT-RATE checkpoint (see docs/vlm_baselines_runbook.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.fvlm_organ_report import (
    build_organ_text,
    load_decomposed_report,
    scan_id_from_volume_name,
)


def _read_ids(ids_file: Path) -> list[str]:
    """Read scan ids from a `.json` ({"ids": [...]} or a bare list) or `.txt`
    (one per line), then collapse each to its scan-study id and dedup.

    `scan_id_from_volume_name` maps both 'train_10000_a_1' and the already-collapsed
    'train_10000_a' to 'train_10000_a', matching the decomposed-report JSON keys.
    """
    text = ids_file.read_text()
    if ids_file.suffix == ".json":
        raw = json.loads(text)
        raw_ids = raw["ids"] if isinstance(raw, dict) else raw
    else:
        raw_ids = [ln.strip() for ln in text.splitlines() if ln.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for vid in raw_ids:
        sid = scan_id_from_volume_name(vid)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute per-organ fVLM conditioning embeddings (4, 256)"
    )
    p.add_argument(
        "--ids-file",
        required=True,
        help="scan/volume ids: .json with an 'ids' key (or bare list), or .txt one-per-line",
    )
    p.add_argument(
        "--report-root",
        required=True,
        help="directory containing desc_info.json + conc_info.json",
    )
    p.add_argument(
        "--out-dir", required=True, help="output directory for <scan_id>.pt ((4, 256))"
    )
    p.add_argument(
        "--ckpt", default=None, help="fVLM checkpoint (default: adapter DEFAULT_CKPT)"
    )
    p.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0)")
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

    ids = _read_ids(Path(args.ids_file))
    if args.limit:
        ids = ids[: args.limit]

    root = Path(args.report_root)
    desc_info, conc_info = load_decomposed_report(
        root / "desc_info.json", root / "conc_info.json"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — fVLM (lavis/transformers) only loads on demand.
    from src.baselines.fvlm_adapter import DEFAULT_CKPT, FVLMBackbone

    ckpt = args.ckpt or DEFAULT_CKPT
    print(f"Loading fVLM backbone on {args.device} (ckpt={ckpt}) ...")
    backbone = FVLMBackbone(ckpt_path=ckpt, device_str=args.device, load_weights=True)

    processed, skipped, missing = 0, 0, 0
    for sid in tqdm(ids, desc="fvlm cond emb"):
        out_path = out_dir / f"{sid}.pt"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        # Truly-absent scan (no decomposition at all) → skip; an all-template
        # embedding would carry no report signal. (toy-5000 are all present.)
        if sid not in desc_info and sid not in conc_info:
            print(f"WARNING: {sid} absent from decomposed report — skipping")
            missing += 1
            continue

        organ_text = build_organ_text(
            desc_info, conc_info, sid
        )  # {organ: text} (all 4)
        feat = backbone.encode_organ_texts(organ_text)  # (4, 256) on device
        torch.save(feat.cpu().float(), out_path)  # (4, 256) cpu float32
        processed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, missing={missing}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
