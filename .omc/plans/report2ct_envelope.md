# Report2CT Envelope (Phase A Day 5 lock target)

**Status (Day 3 draft, lock at 5/31)**: target envelope around Report2CT's headline configuration.

## Anchor: 3-TE-cfg(mid)

Picked from `paper_pdf/Report2CT.pdf` Figures 5/6 as the "best aligned + still-good-FID" configuration matching the paper's narrative ("classifier-free guidance further enhanced alignment with only a minor trade-off in FID"). It is **not** the lowest-FID variant (1-TE w/o cfg = 3.57 is, but with weak alignment); anchoring to the headline config keeps "beat Report2CT" honest.

| Metric | Anchor value | Envelope | Bounds | Source |
|---|---|---|---|---|
| **2.5D-FID Average** | **4.04** | ±15% | `[3.43, 4.65]` | Fig 6 FID Average (3-TE-cfg5 bar) |
| **CLIPScore-T2I** | **59.93** | ±10% | `[53.94, 65.92]` | Fig 5 left panel (3-TE-cfg5 bar) |
| CLIPScore-I2I (informational, no envelope) | 53.06 | — | — | Fig 5 middle panel |
| **FVD** | **self-measured 6/1 anchor** | ±25% | TBD on 6/1 | NOT reported in paper; anchored from our 6/1 1-epoch run on paper-spec skeleton |
| FID XY / XZ / YZ (informational) | 2.92 / 5.48 / 3.72 | — | — | Fig 6 right panels (3-TE-cfg5) |

> All bars read off Figure 5/6 of `paper_pdf/Report2CT.pdf` (verified Day 1 PM). The
> exact CFG-scale label of the "(mid)" bar (cfg=3 vs cfg=5) is annotated in the paper's
> Fig caption as "cfg" without an explicit numeric — we treat it as **cfg=5** based on
> the paper text saying "moderate scales (e.g. 3–7)" and the bar being the second of
> three CFG'd 3-TE bars. Cross-check the submodule's default CFG scale when we wire the
> launcher (Phase B Day 1).

## FVD anchor reasoning

The Report2CT paper does NOT report FVD. We measure FVD ourselves on 6/1 from a
1-epoch run of the paper-spec skeleton on 100 samples, then use that number as the
anchor for FVD ±25%. Documented in `report2ct_sanity.json` schema as:

```jsonc
{
  "fvd": {
    "anchor": <number>,
    "anchor_source": "self_measured_6_1_paper_spec_skeleton",
    "envelope_pct": 25,
    "note": "FVD not reported in Report2CT paper. Anchor widened to absorb baseline uncertainty."
  },
  "fid_2p5d_avg": {
    "anchor": 4.04,
    "anchor_source": "paper_fig_6_3te_cfg5",
    "envelope_pct": 15
  },
  "clip_score_t2i": {
    "anchor": 59.93,
    "anchor_source": "paper_fig_5_3te_cfg5",
    "envelope_pct": 10
  },
  "clip_score_i2i": {
    "anchor": 53.06,
    "anchor_source": "paper_fig_5_3te_cfg5",
    "envelope_pct": null,
    "informational": true
  }
}
```

## Win condition (recap)

`ours_final` beats `report2ct_our_repro` in ≥2 of {2.5D-FID, CLIPScore-T2I, FVD} **and**
`report2ct_our_repro` is within envelope on 2.5D-FID and CLIPScore-T2I.

Metric priority for the headline claim: **2.5D-FID > CLIPScore-T2I > FVD**.
A 2/3 win that excludes 2.5D-FID requires writeup to note the weaker anchor situation.
A FVD-only envelope miss is acceptable (FVD has self-anchor). A 2.5D-FID or CLIPScore-T2I
envelope miss triggers the downgrade branch (plan §3 win condition L130).

## Day-5 lock checklist

- [ ] Cite exact paper page/figure for each paper-anchored value.
- [ ] Confirm CFG-scale of the (mid) 3-TE bar (cfg=5 likely; cross-check submodule default).
- [ ] Verify VLM3D-Dockers `2.5-D FID` definition matches the paper's "FID with 2.5D feature extraction (RadImageNet ResNet-50, three planes averaged)" — see Day 5 in plan.
- [ ] Pin envelope numbers in `results/report2ct_envelope.json` (single source of truth for downstream scripts).
