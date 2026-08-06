# Case sheet — `valid_1073_a` (recon 1) — group: **diffuse-low-burden**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1073` |
| scan | `valid_1073_a` |
| reconstruction | `1`  (available: 1) |
| primary file | `valid_1073_a_1.nii.gz` |
| age / sex | 24Y / F |
| manufacturer / model | PNMS / MX 16 |
| kernel | EA |
| shape (RxCxSlices) | 768x768x226 |
| voxel spacing (x,y,z mm) | 0.39062x0.39062x1.5 |
| intensity min/med/max (HU) | -1024 / -893.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/diffuse-low-burden__valid_1073_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Pulmonary fibrotic sequela**

## Findings (Findings_EN)

> CTO is normal. Calibration of mediastinal major vascular structures is natural. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No pathologically sized and configured lymph nodes were detected in the mediastinum and at both hilar levels. When examined in the lung parenchyma window; trachea and both main bronchi are open. Sequelae changes are observed at the apical level of both lungs. At the laterobasal level, parenchymal thin bands are observed. There was no finding compatible with pneumonia in both lungs. Sections passing through the upper abdomen included in the sections are natural. Surrounding soft tissue plans are natural. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> No finding compatible with pneumonia was observed.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion-thickening was not observed. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **ABSENT** |  |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | — | **ABSENT** |  |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | Sequelae changes are observed at the apical level of both lungs. |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`diffuse-low-burden__valid_1073_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.39062x0.39062x1.5, shape 768x768x226).
