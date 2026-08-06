#!/usr/bin/env python3
"""Join the FID / CLIP / FVD spacing sweeps into one table per model.

The three metrics were swept separately (different scripts, different days), so this reads the
result jsons and aligns them on the declared-spacing key. Coverage is ragged — FID has the
densest grid, FVD the sparsest — so missing cells print as "-" rather than being dropped.

Content provenance matters and is NOT uniform:
  wan        FID / CLIP / FVD all re-declare the SAME gen@(0.73,0.73,1.34) volumes  → comparable
  report2ct  FID / CLIP re-declare gen@0.8 ; FVD re-declares gen@(0.757,0.757,1.5)  → the FVD
             column is a different content set, so read it for TREND, not against the other two.

    python tests/spacing_fov/consolidate_sweeps.py
"""

from __future__ import annotations

import json
import pathlib

R = pathlib.Path(__file__).parent / "results"

MODELS = [
    {
        "name": "report2ct_wan ep299  (grid 512x512x253)",
        "fid": ("sweep_fid__wan_ep299_300.json", "gen@0.73/0.73/1.34"),
        "clip": ("zsweep_clip__wan_ep299_300_n300.json", "gen@0.73/0.73/1.34"),
        "fvd": ("fvd_sweep__wan_ep299_300.json", "gen@0.73/0.73/1.34"),
    },
    {
        "name": "report2ct ep099-toy  (grid 480x480x256)",
        "fid": ("sweep_fid__gen_sp0.8_300.json", "gen@0.8/0.8/1.5"),
        "clip": ("zsweep_clip__gen_sp0.8_300_n300.json", "gen@0.8/0.8/1.5"),
        "fvd": (
            "fvd_sweep__r2c_ep099_sp0.757_300.json",
            "gen@0.757/0.757/1.5  (DIFFERENT)",
        ),
    },
]


def _load(fname: str) -> dict:
    p = R / fname
    if not p.is_file():
        return {}
    d = json.loads(p.read_text())
    return d.get("results", d)  # the FID sweep nests its cells under "results"


def _key_sort(k: str) -> tuple[float, float]:
    sx, sz = k.split("_")
    return (float(sz), float(sx))


def main() -> None:
    for m in MODELS:
        fid, clip, fvd = (_load(m[k][0]) for k in ("fid", "clip", "fvd"))
        print(f"\n{'=' * 74}\n{m['name']}\n{'=' * 74}")
        for k in ("fid", "clip", "fvd"):
            print(f"  {k.upper():5s} content: {m[k][1]}")
        print()
        print(
            f"{'declared (in-plane, z)':24}{'FID':>9}{'CLIP-T2I':>10}{'CLIP-I2I':>10}{'FVD':>9}"
        )
        keys = sorted(set(fid) | set(clip) | set(fvd), key=_key_sort)
        for k in keys:
            sx, sz = k.split("_")
            cells = [
                fid.get(k, {}).get("FID_2p5D_Avg"),
                clip.get(k, {}).get("CLIPScore"),
                clip.get(k, {}).get("CLIPScore_I2I"),
                fvd.get(k, {}).get("FVD_CTCLIP"),
            ]
            widths, precs = (9, 10, 10, 9), (3, 2, 2, 4)
            row = "".join(
                f"{v:{w}.{p}f}" if v is not None else "-".rjust(w)
                for v, w, p in zip(cells, widths, precs)
            )
            print(f"{f'({sx}, {sx}, {sz})':24}{row}")

        print()
        for label, src, key, best in (
            ("FID  (lower better)", fid, "FID_2p5D_Avg", min),
            ("CLIP-T2I (higher)  ", clip, "CLIPScore", max),
            ("FVD  (lower better)", fvd, "FVD_CTCLIP", min),
        ):
            have = {k: v[key] for k, v in src.items() if v.get(key) is not None}
            if not have:
                print(f"  best {label}: (not measured)")
                continue
            k = best(have, key=have.get)
            print(
                f"  best {label}: ({k.replace('_', ', ')})  = {have[k]:.4f}   [{len(have)} pts]"
            )


if __name__ == "__main__":
    main()
