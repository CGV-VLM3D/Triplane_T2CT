# Case sheet — `valid_1077_a` (recon 1) — group: **lung-nodule-only**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1077` |
| scan | `valid_1077_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1077_a_1.nii.gz` |
| age / sex | 36Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x222 |
| voxel spacing (x,y,z mm) | 0.31543x0.31543x1.5 |
| intensity min/med/max (HU) | -1024 / -221.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/lung-nodule-only__valid_1077_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lung nodule**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; There is a millimetric subpleural nodule in the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> Thoracic CT examination within normal limits

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion-thickening was not observed. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | ✅ | **AFFIRMED** | There is a millimetric subpleural nodule in the right lung. |
| Lung opacity | — | **NEGATED** | Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`lung-nodule-only__valid_1077_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.31543x0.31543x1.5, shape 1024x1024x222).
