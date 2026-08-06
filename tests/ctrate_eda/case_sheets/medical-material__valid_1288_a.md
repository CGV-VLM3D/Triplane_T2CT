# Case sheet — `valid_1288_a` (recon 1) — group: **medical-material**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1288` |
| scan | `valid_1288_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1288_a_1.nii.gz` |
| age / sex | 30Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x277 |
| voxel spacing (x,y,z mm) | 0.38379x0.38379x1.5 |
| intensity min/med/max (HU) | -1024 / -916.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/medical-material__valid_1288_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Fixation material is observed in the thoracic vertebrae included in the study area. Metallic body artifact is observed on the left anterior chest wall.

## Impression (Impressions_EN)

> Examination within normal limits

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion-thickening was not observed. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Lung opacity | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`medical-material__valid_1288_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.38379x0.38379x1.5, shape 1024x1024x277).
- [ ] Verify label **Medical material**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
