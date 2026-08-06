# Case sheet — `valid_1085_a` (recon 1) — group: **lung-nodule-only**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1085` |
| scan | `valid_1085_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1085_a_1.nii.gz` |
| age / sex | 47Y / M |
| manufacturer / model | Philips / Brilliance Big Bore |
| kernel | A |
| shape (RxCxSlices) | 512x512x232 |
| voxel spacing (x,y,z mm) | 0.72656x0.72656x1.5 |
| intensity min/med/max (HU) | -1024 / -638.0 / 1705 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/lung-nodule-only__valid_1085_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lung nodule**

## Findings (Findings_EN)

> No lymph node in pathological size and appearance was observed in the supraclavicular fossa, axilla and mediastinum. Heart dimensions and compartments appear natural. Calibrations of mediastinal main vascular structures are natural. Pericardial effusion was not detected. No mass or nodular suspicious space-occupying lesion was detected in the lung parenchyma. A few nonspecific nodules with diameters less than 5 mm are observed. No pneumonic infiltration was detected in the parenchyma. No feature was observed in the upper abdomen sections. No lytic-destructive lesions were detected in bone structures.

## Impression (Impressions_EN)

> Not given.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion was not detected. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No lymph node in pathological size and appearance was observed in the supraclavicular fossa, axilla and mediastinum. |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | ✅ | **AFFIRMED** | A few nonspecific nodules with diameters less than 5 mm are observed. |
| Lung opacity | — | **NEGATED** | No pneumonic infiltration was detected in the parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`lung-nodule-only__valid_1085_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.72656x0.72656x1.5, shape 512x512x232).
