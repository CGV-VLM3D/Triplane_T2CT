#!/usr/bin/env python3
"""Print the spacing sweeps as full markdown tables — every point, every metric.

Optima alone are misleading here: the four metrics disagree about where the optimum is, so the
selection has to be read off the whole grid. This prints (a) the full ragged grid and (b) the
complete-coverage subset with per-metric ranks, which is the only subset you can compare across
all four at once.
"""

from __future__ import annotations

import json
import pathlib

R = pathlib.Path(__file__).parent / "results"

MODELS = [
    (
        "report2ct_wan ep299 — grid 512x512x253, content gen@(0.73, 0.73, 1.34) for ALL metrics",
        "sweep_fid__wan_ep299_300.json",
        "zsweep_clip__wan_ep299_300_n300.json",
        "fvd_sweep__wan_ep299_300.json",
    ),
    (
        "report2ct ep099-toy — grid 480x480x256, FID/CLIP content gen@0.8 but FVD content gen@0.757",
        "sweep_fid__gen_sp0.8_300.json",
        "zsweep_clip__gen_sp0.8_300_n300.json",
        "fvd_sweep__r2c_ep099_sp0.757_300.json",
    ),
]

# (label, source index, json key, lower_is_better)
COLS = [
    ("FID", 0, "FID_2p5D_Avg", True),
    ("CLIP-T2I", 1, "CLIPScore", False),
    ("CLIP-I2I", 1, "CLIPScore_I2I", False),
    ("FVD", 2, "FVD_CTCLIP", True),
]
FMT = {"FID": "{:.3f}", "CLIP-T2I": "{:.2f}", "CLIP-I2I": "{:.2f}", "FVD": "{:.4f}"}


def _load(fname: str) -> dict:
    p = R / fname
    d = json.loads(p.read_text()) if p.is_file() else {}
    return d.get("results", d)


def _sort(k: str) -> tuple[float, float]:
    sx, sz = k.split("_")
    return (float(sz), float(sx))


def _cell(srcs, k, ci):
    _, si, key, _ = COLS[ci]
    return srcs[si].get(k, {}).get(key)


def main() -> None:
    for title, *files in MODELS:
        srcs = [_load(f) for f in files]
        keys = sorted(set().union(*(set(s) for s in srcs)), key=_sort)

        print(f"\n### {title}\n")
        print("| declared (in-plane, z) | " + " | ".join(c[0] for c in COLS) + " |")
        print("|---|" + "---|" * len(COLS))
        for k in keys:
            sx, sz = k.split("_")
            cells = []
            for ci, (name, *_rest) in enumerate(COLS):
                v = _cell(srcs, k, ci)
                cells.append(FMT[name].format(v) if v is not None else "—")
            print(f"| {sx}, {sx}, {sz} | " + " | ".join(cells) + " |")

        full = [
            k
            for k in keys
            if all(_cell(srcs, k, ci) is not None for ci in range(len(COLS)))
        ]
        if not full:
            print("\n(no point has all four metrics)")
            continue

        print(f"\n**전 지표 측정된 {len(full)}점 — 지표별 순위 (1 = 최선)**\n")
        ranks = {}
        for ci, (name, _si, _key, lower) in enumerate(COLS):
            vals = {k: _cell(srcs, k, ci) for k in full}
            order = sorted(full, key=lambda k: vals[k], reverse=not lower)
            ranks[name] = {k: i + 1 for i, k in enumerate(order)}

        print(
            "| declared | "
            + " | ".join(f"{c[0]} (순위)" for c in COLS)
            + " | 평균순위 |"
        )
        print("|---|" + "---|" * (len(COLS) + 1))
        rows = []
        for k in full:
            mean_r = sum(ranks[c[0]][k] for c in COLS) / len(COLS)
            rows.append((mean_r, k))
        for mean_r, k in sorted(rows):
            sx, sz = k.split("_")
            cells = [
                f"{FMT[c[0]].format(_cell(srcs, k, ci))} ({ranks[c[0]][k]})"
                for ci, c in enumerate(COLS)
            ]
            print(
                f"| {sx}, {sx}, {sz} | " + " | ".join(cells) + f" | **{mean_r:.2f}** |"
            )


if __name__ == "__main__":
    main()
