# Does abnormality reproduction work only for report2ct? (wan? text2ct?)

My visual read of the 4 figures in `figures/` — **verify yourself**, each row is a model, cols = axial/coronal/sagittal.
Open: [valid_27](figures/valid_27_a_1.png) · [valid_144](figures/valid_144_a_1.png) ·
[valid_155](figures/valid_155_a_1.png) · [valid_322](figures/valid_322_a_1.png).

## Short answer
**No — it's not "only report2ct".** Big structural findings show up across *all* families; the
difference is image quality/resolution, not whether the finding appears. Small findings work for none.

## Big structural findings — reproduced by every family
`valid_27` (cardiomegaly + effusion), `valid_144` (giant right-hemithorax mass):

| family | does the finding appear? | quality note |
|---|---|---|
| **report2ct_toy_v2 / _full** (MAISI latent) | **yes, clearest** | sharpest, cleanest anatomy; enlarged heart / right-hemithorax mass rendered best |
| **report2ct_wan** (Wan latent) | **yes** | finding clearly present, but texture noticeably **noisier/grainier** — "works, lower image quality", NOT a failure |
| **report2ct_wan_mask** | **yes** | finding present + organ boundaries tighter (mask fixes geometry) |
| **text2ct** (toy_v2) | **yes, weaker** | z is coarse (3 mm / 128 slices) → **blocky**; on `valid_144` lungs stay more aerated so the mass reads weaker |

Rough fidelity/quality ranking on big findings: **report2ct ≳ wan > text2ct** (report2ct sharpest,
wan noisier, text2ct lowest-res). But wan is **not** "안 됨" — the enlarged heart / mass is visibly there.

## Small findings — reproduced by NONE
`valid_155` (lung nodule 15×13 mm), `valid_322` (coronary LAD calcification): not reliably generated
by any family, and not at a fixed location. This is a **fundamental report→CT resolution/conditioning
limit shared across all models**, not a family difference.

## Where the evidence lives
- **Figures**: `tests/abnormality_fidelity/figures/<case>.png` (5 rows incl. GT, 3 planes).
- **Reports** (what to look for): `tests/abnormality_fidelity/case_reports.txt`.
- **How slices/windows were chosen**: `tests/abnormality_fidelity/make_figures.py` + README.
- Generations are text-conditioned, **not registered** — don't expect the same voxel across a row;
  judge "does the finding appear / look right", then page through the volumes yourself to confirm.
