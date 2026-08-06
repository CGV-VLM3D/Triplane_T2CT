# Case sheet — `valid_366_a` (recon 1) — group: **medical-material**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_366` |
| scan | `valid_366_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_366_a_1.nii.gz` |
| age / sex | 35Y / F |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x229 |
| voxel spacing (x,y,z mm) | 0.3418x0.3418x1.5 |
| intensity min/med/max (HU) | -1024 / -909.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/medical-material__valid_366_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**

## Findings (Findings_EN)

> Trachea and lumen of both main bronchi are open. No occlusive pathology was detected in the trachea and lumen of both main bronchi. Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be seen; Calibration of mediastinal major vascular structures is natural. Heart contour size is natural. Pericardial thickening-effusion was not detected. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No lymph node was detected in mediastinal and bilateral hilar pathological size and appearance. Prosthetic material was observed in both breasts. When examined in the lung parenchyma window; no mass nodule-infiltration was detected in both lung parenchyma. Bilateral pleural thickening-effusion was not detected. Upper abdominal sections entering the examination area are natural. Bilateral adrenal gland calibration was normal and no space-occupying lesion was detected. No lytic-destructive lesion was detected in bone structures.

## Impression (Impressions_EN)

> No sign of pneumonia was detected.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **ABSENT** |  |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **NEGATED** | no mass nodule-infiltration was detected in both lung parenchyma. |
| Lung opacity | — | **NEGATED** | no mass nodule-infiltration was detected in both lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`medical-material__valid_366_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.3418x0.3418x1.5, shape 1024x1024x229).
- [ ] Verify label **Medical material**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
