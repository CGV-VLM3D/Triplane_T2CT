#!/usr/bin/env python3
"""Sanity-check our Qwen report decomposition against the author's TRAIN GT.

The fVLM authors released their decomposition for CT-RATE **train** only. That
makes train a free ground-truth set to check whether OUR reproduction (default
`Qwen/Qwen2.5-7B-Instruct`, the model fixed in the plan) matches the author's
distribution BEFORE we spend compute decomposing the held-out `valid_fixed`.

This is a single-model quality report, NOT a model-selection sweep — size is
fixed at 7B per the plan; if the numbers look poor, re-run with a bigger
`--model-id` (e.g. Qwen/Qwen2.5-14B-Instruct).

It reports, on a random sample of `--n` train scans:
  (a) Mention agreement (Step 1) — our `mentioned` decision vs the author's
      per-section organ-key presence: accuracy / precision / recall / F1 +
      confusion, overall and per organ, for BOTH the LLM-only predictor and the
      LLM+substring predictor (so you can see the substring fallback's effect).
  (b) Extraction similarity (Step 2) — on agree-positives, ROUGE-L
      (torchmetrics) and token-IoU between our extracted text and the author's,
      with best/worst examples dumped.

Reference numbers (informative, not enforced): mention-F1 ~0.95, median ROUGE-L
~0.6 indicate the 7B reproduction tracks the author well.

Usage (GPU; GPU2 is full on this box — pin 0 or 1):
  CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_decompose_fvlm.py --n 150 \
      --model-id Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

from scripts.decompose_reports_fvlm import (
    build_backend,
    extract_prompt,
    load_scan_reports,
    mention_prompt,
    normalize_extraction,
    parse_yes_no,
    substring_mentioned,
)
from src.data.fvlm_organ_report import ORGANS, load_decomposed_report

_DEFAULT_GT = Path("/workspace/data/fVLM/decomposed_report")
_TRAIN_CSV = "/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv"


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


class _Confusion:
    """Accumulate TP/FP/FN/TN, overall and per organ."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_organ = {o: [0, 0, 0, 0] for o in ORGANS}  # tp,fp,fn,tn

    def add(self, organ: str, pred: bool, gt: bool) -> None:
        idx = (
            0
            if (pred and gt)
            else 1
            if (pred and not gt)
            else 2
            if (not pred and gt)
            else 3
        )
        self.per_organ[organ][idx] += 1
        if idx == 0:
            self.tp += 1
        elif idx == 1:
            self.fp += 1
        elif idx == 2:
            self.fn += 1
        else:
            self.tn += 1

    def summary(self) -> dict:
        p, r, f = _prf1(self.tp, self.fp, self.fn)
        total = self.tp + self.fp + self.fn + self.tn
        acc = (self.tp + self.tn) / total if total else 0.0
        per_organ = {}
        for o, (tp, fp, fn, tn) in self.per_organ.items():
            po, pr_, pf = _prf1(tp, fp, fn)
            per_organ[o] = {
                "precision": po,
                "recall": pr_,
                "f1": pf,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "per_organ": per_organ,
        }


def _token_iou(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate Qwen decomposition vs author TRAIN GT"
    )
    p.add_argument("--n", type=int, default=150, help="number of train scans to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gt-root", default=str(_DEFAULT_GT), help="author desc/conc JSON dir"
    )
    p.add_argument("--reports-csv", default=_TRAIN_CSV)
    p.add_argument(
        "--backend", choices=["transformers", "vllm"], default="transformers"
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--dtype", choices=["bfloat16", "float16", "auto"], default="bfloat16"
    )
    p.add_argument("--device", default="cuda:0", help="GPU2 is full — use 0 or 1.")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", default=None, help="report JSON path (auto by default)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    gt_root = Path(args.gt_root)
    desc_gt, conc_gt = load_decomposed_report(
        gt_root / "desc_info.json", gt_root / "conc_info.json"
    )
    reports = load_scan_reports(args.reports_csv)

    candidates = [s for s in reports if s in desc_gt and s in conc_gt]
    random.Random(args.seed).shuffle(candidates)
    sample = candidates[: args.n]
    print(
        f"calibrating on {len(sample)} train scans (model={args.model_id} dtype={args.dtype})"
    )

    from torchmetrics.text import ROUGEScore  # noqa: PLC0415 (heavy)

    rouge = ROUGEScore(rouge_keys=("rougeL",))
    backend = build_backend(args)

    conf_llm = _Confusion()
    conf_comb = _Confusion()
    rougeL: list[float] = []
    ious: list[float] = []
    examples: list[dict] = []

    from tqdm import tqdm  # noqa: PLC0415

    # section name -> (author GT dict, "is findings?")
    sections = [("findings", desc_gt, 0), ("impression", conc_gt, 1)]

    for scan in tqdm(sample, desc="calibrate"):
        findings, impression = reports[scan]
        texts = {"findings": findings, "impression": impression}
        for sec_name, gt_map, sec_idx in sections:
            section_text = texts[sec_name].strip()
            gt_present = {o for o in ORGANS if o in gt_map.get(scan, {})}
            if not section_text:
                # empty section: every organ is a true negative if GT also empty.
                for o in ORGANS:
                    conf_llm.add(o, False, o in gt_present)
                    conf_comb.add(o, False, o in gt_present)
                continue

            # Step 1: ask the LLM for ALL organs (so we can score llm-only AND
            # llm+substring from one pass).
            llm_resp = backend.generate(
                [mention_prompt(section_text, o) for o in ORGANS], max_new_tokens=8
            )
            llm_yes = {o: parse_yes_no(r) for o, r in zip(ORGANS, llm_resp)}
            sub = {o: substring_mentioned(section_text, o) for o in ORGANS}
            pred_comb = {o: llm_yes[o] or sub[o] for o in ORGANS}
            for o in ORGANS:
                conf_llm.add(o, llm_yes[o], o in gt_present)
                conf_comb.add(o, pred_comb[o], o in gt_present)

            # Step 2: extraction quality on agree-positives (we and GT both have it).
            agree_pos = [o for o in ORGANS if pred_comb[o] and o in gt_present]
            if agree_pos:
                ex_resp = backend.generate(
                    [extract_prompt(section_text, o) for o in agree_pos],
                    max_new_tokens=args.max_new_tokens,
                )
                for o, r in zip(agree_pos, ex_resp):
                    ours = normalize_extraction(r, o)
                    gold = str(gt_map[scan][o])
                    rl = float(rouge(ours, gold)["rougeL_fmeasure"])
                    iou = _token_iou(ours, gold)
                    rougeL.append(rl)
                    ious.append(iou)
                    examples.append(
                        {
                            "scan": scan,
                            "section": sec_name,
                            "organ": o,
                            "rougeL": rl,
                            "ours": ours[:200],
                            "author": gold[:200],
                        }
                    )

    examples.sort(key=lambda e: e["rougeL"])
    report = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "n_scans": len(sample),
        "seed": args.seed,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mention": {
            "llm_only": conf_llm.summary(),
            "llm_plus_substring": conf_comb.summary(),
        },
        "extraction": {
            "n_pairs": len(rougeL),
            "rougeL_mean": statistics.fmean(rougeL) if rougeL else None,
            "rougeL_median": statistics.median(rougeL) if rougeL else None,
            "token_iou_mean": statistics.fmean(ious) if ious else None,
            "examples_worst": examples[:5],
            "examples_best": examples[-5:][::-1],
        },
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path("/workspace/data/fVLM/calibration")
        / f"{args.model_id.split('/')[-1]}_{len(sample)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    m = report["mention"]
    e = report["extraction"]
    print(
        "\n=== mention (llm+substring) ===\n"
        f"  acc={m['llm_plus_substring']['accuracy']:.3f} "
        f"P={m['llm_plus_substring']['precision']:.3f} "
        f"R={m['llm_plus_substring']['recall']:.3f} "
        f"F1={m['llm_plus_substring']['f1']:.3f}\n"
        f"  (llm-only F1={m['llm_only']['f1']:.3f})\n"
        "=== extraction ===\n"
        f"  n={e['n_pairs']} rougeL_mean={e['rougeL_mean']} "
        f"rougeL_median={e['rougeL_median']} token_iou_mean={e['token_iou_mean']}\n"
        f"report -> {out_path}"
    )


if __name__ == "__main__":
    main()
