#!/usr/bin/env python3
"""Aggregate per-case sharpness JSONs (from blur_spectrum.py) into a report.

Reads all shard_*.json, computes per-model mean±std for every metric, per-case
rankings (how often each generated model is sharpest/blurriest), the clip_grid
resample effect, and a robustness verdict (does the ranking agree across metrics).
Emits a markdown report to stdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st

GEN = ["report2ct", "wan", "wan_mask"]
ALL = ["gt"] + GEN
CLIP_METRICS = ["hf_clip", "grad_clip", "lap_clip", "edge_clip"]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="tests/clip_grid_view/blur_results/shard_*.json")
    args = ap.parse_args()

    data: dict[str, dict] = {}
    for f in sorted(glob.glob(args.glob)):
        data.update(json.load(open(f)))
    cases = sorted(data)
    n = len(cases)

    # collect per-model metric lists
    cols = ["hf_native", "hf_clip", "grad_clip", "lap_clip", "edge_clip"]
    series: dict[str, dict[str, list]] = {m: {c: [] for c in cols} for m in ALL}
    for case in cases:
        for m in ALL:
            if m in data[case]:
                for c in cols:
                    series[m][c].append(data[case][m].get(c, float("nan")))

    out = [f"# Blur / sharpness over {n} samples (higher = sharper)\n"]
    out.append(
        "clip_grid metrics (0.75mm, same resolution for all → fair cross-model). "
        "hf_native shown for the resample-effect only.\n"
    )

    # main table: mean ± std
    out.append("| source | hf_native | hf_clip | grad_clip | lap_clip | edge_clip |")
    out.append("|---|---|---|---|---|---|")
    for m in ALL:
        cellvals = []
        for c in cols:
            mu, sd = _mean_std(series[m][c])
            if c in ("grad_clip", "lap_clip"):
                cellvals.append(f"{mu:.0f}±{sd:.0f}")
            else:
                cellvals.append(f"{mu:.4f}±{sd:.4f}")
        out.append(f"| {m} | " + " | ".join(cellvals) + " |")

    # resample effect: clip/native ratio per model
    out.append("\n## clip_grid resample effect (hf_clip / hf_native, mean)")
    out.append("| source | ratio | interpretation |")
    out.append("|---|---|---|")
    for m in ALL:
        ratios = [
            data[c][m]["hf_clip"] / data[c][m]["hf_native"]
            for c in cases
            if m in data[c]
            and data[c][m]["hf_native"] not in (0, float("nan"))
            and data[c][m]["hf_native"] == data[c][m]["hf_native"]
        ]
        mu, _ = _mean_std(ratios)
        out.append(f"| {m} | {mu:.3f} | {'HF lost' if mu < 1 else 'HF kept/gained'} |")

    # per-case ranking among generated models (blurriest = lowest metric)
    out.append(
        "\n## Among generated models — how often each is BLURRIEST (lowest) / SHARPEST (highest)"
    )
    for metric in CLIP_METRICS:
        blur = {g: 0 for g in GEN}
        sharp = {g: 0 for g in GEN}
        valid = 0
        for case in cases:
            vals = {g: data[case][g][metric] for g in GEN if g in data[case]}
            vals = {g: v for g, v in vals.items() if v == v}
            if len(vals) < len(GEN):
                continue
            valid += 1
            blur[min(vals, key=vals.get)] += 1
            sharp[max(vals, key=vals.get)] += 1
        out.append(f"\n**{metric}** (n={valid})")
        out.append("| model | blurriest count | sharpest count |")
        out.append("|---|---|---|")
        for g in GEN:
            out.append(f"| {g} | {blur[g]} | {sharp[g]} |")

    # robustness verdict: mean ranking of generated models across metrics
    out.append(
        "\n## Robustness — mean rank of generated models per clip metric (1 = sharpest of 3)"
    )
    out.append("| metric | " + " | ".join(GEN) + " |")
    out.append("|---|" + "---|" * len(GEN))
    for metric in CLIP_METRICS:
        ranksum = {g: [] for g in GEN}
        for case in cases:
            vals = {
                g: data[case][g][metric]
                for g in GEN
                if g in data[case] and data[case][g][metric] == data[case][g][metric]
            }
            if len(vals) < len(GEN):
                continue
            order = sorted(vals, key=vals.get, reverse=True)  # sharpest first
            for rank, g in enumerate(order, 1):
                ranksum[g].append(rank)
        row = [f"{st.mean(ranksum[g]):.2f}" if ranksum[g] else "nan" for g in GEN]
        out.append(f"| {metric} | " + " | ".join(row) + " |")

    print("\n".join(out))


if __name__ == "__main__":
    main()
