# Case sheet — `valid_1153_a` (recon 1) — group: **diffuse-low-burden**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1153` |
| scan | `valid_1153_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1153_a_1.nii.gz` |
| age / sex | 38Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x242 |
| voxel spacing (x,y,z mm) | 0.38574x0.38574x1.5 |
| intensity min/med/max (HU) | -1024 / -850.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/diffuse-low-burden__valid_1153_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Bronchiectasis**

## Findings (Findings_EN)

> Thymic hyperplasia was observed. There is bilateral gynecomastia. Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; There are mild bronchiectatic changes in both lungs. Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> Thymus appears hyperplastic. Mild bronchiectatic changes in both lungs.

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
| Lung nodule | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Lung opacity | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | ✅ | **AFFIRMED** | There are mild bronchiectatic changes in both lungs. |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`diffuse-low-burden__valid_1153_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.38574x0.38574x1.5, shape 1024x1024x242).
