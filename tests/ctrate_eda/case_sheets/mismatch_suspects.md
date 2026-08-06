# Report <-> label mismatch suspects (module 07)

Heuristic NegEx over the 30-volume bundle's **25 primary cases**. These are POINTERS to verify — not confirmed errors. Silver labels are RadBERT-mined; the NegEx term dictionary is broad (synonym stems), so both directions carry false alarms.

## A. FP-suspects — positive silver-label whose term is NOT affirmed in the report

(positive label + NegEx status ∈ {NEGATED, UNCERTAIN, ABSENT})

| group | scan | label | NegEx status |
|---|---|---|:---:|
| diffuse-low-burden | `valid_1022_a` | Pulmonary fibrotic sequela | ABSENT |
| medical-material | `valid_1288_a` | Medical material | ABSENT |
| medical-material | `valid_366_a` | Medical material | ABSENT |
| multi-abnormality | `valid_1016_b` | Atelectasis | UNCERTAIN |
| multi-abnormality | `valid_1016_b` | Pleural effusion | ABSENT |
| multi-abnormality | `valid_1016_b` | Peribronchial thickening | ABSENT |
| multi-abnormality | `valid_1016_b` | Consolidation | UNCERTAIN |
| multi-abnormality | `valid_1016_d` | Arterial wall calcification | ABSENT |
| multi-abnormality | `valid_1016_d` | Coronary artery wall calcification | ABSENT |
| multi-abnormality | `valid_1016_d` | Lymphadenopathy | ABSENT |
| multi-abnormality | `valid_103_a` | Lymphadenopathy | ABSENT |
| multi-abnormality | `valid_1041_c` | Lymphadenopathy | ABSENT |
| multi-abnormality | `valid_1041_c` | Interlobular septal thickening | ABSENT |
| multi-abnormality | `valid_1078_a` | Arterial wall calcification | ABSENT |
| multi-abnormality | `valid_1078_a` | Cardiomegaly | ABSENT |
| multi-abnormality | `valid_1078_a` | Coronary artery wall calcification | ABSENT |

**FP-suspect count:** 16 label-cells across 8 scans.

## B. FN-suspects — AFFIRMED disease term whose label is negative

| group | scan | label | evidence sentence |
|---|---|---|---|
| diffuse-low-burden | `valid_1068_a` | Lung nodule | Nodular density is observed in the spleen hilum, which is considered to be compatible with the accessory spleen with a diameter of approxima… |
| diffuse-low-burden | `valid_1068_a` | Lung opacity | Nodular density is observed in the spleen hilum, which is considered to be compatible with the accessory spleen with a diameter of approxima… |
| lung-nodule-only | `valid_1009_a` | Lung opacity | Diffuse density reduction in bone structures and mild hypertrophic tapering in the end plates are observed. |

**FN-suspect count:** 3 label-cells across 2 scans.

### Caveats

- NegEx is char-window heuristic; long negated lists, cross-sentence scope, and Turkish→EN template phrasing cause misses.
- Overlapping terms (Lung opacity ⊃ consolidation/infiltrate/ground-glass) inflate FN-suspects for co-defined labels.
- Labels are per-**scan**; report is per-**scan**; montage is one **volume**.
- 'Medical material' & 'Lung opacity' have the broadest term sets → most noise.