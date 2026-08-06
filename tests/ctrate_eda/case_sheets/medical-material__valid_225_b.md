# Case sheet — `valid_225_b` (recon 1) — group: **medical-material**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_225` |
| scan | `valid_225_b` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_225_b_1.nii.gz` |
| age / sex | 22Y / F |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x242 |
| voxel spacing (x,y,z mm) | 0.38574x0.38574x1.5 |
| intensity min/med/max (HU) | -1024 / -907.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/medical-material__valid_225_b.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**

## Findings (Findings_EN)

> CTO is normal. Calibration of mediastinal major vascular structures is natural. In the superior jugular vein, the appearance of a catheter extending towards the right atrium appendix is observed. No pathological size and configuration lymph nodes were detected in the mediastinum. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No pathological size and configuration lymph nodes were observed at both hilar levels. When examined in the lung parenchyma window; both hemithorax are symmetrical. Calibration of the trachea and main bronchi is normal. The aeration of the parenchyma of both lungs is normal, and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Right-facing scoliosis is observed in the dorsal region.

## Impression (Impressions_EN)

> Not given.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **AFFIRMED** | In the superior jugular vein, the appearance of a catheter extending towards the right atrium appendix is observed. |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **ABSENT** |  |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | — | **NEGATED** | The aeration of the parenchyma of both lungs is normal, and no infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`medical-material__valid_225_b.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.38574x0.38574x1.5, shape 1024x1024x242).
