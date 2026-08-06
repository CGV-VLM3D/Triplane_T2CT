# Case sheet — `valid_1061_a` (recon 1) — group: **lung-nodule-only**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1061` |
| scan | `valid_1061_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1061_a_1.nii.gz` |
| age / sex | 28Y / F |
| manufacturer / model | Siemens Healthineers / SOMATOM go.All |
| kernel | ['Bl56f', '3'] |
| shape (RxCxSlices) | 512x512x229 |
| voxel spacing (x,y,z mm) | 0.61434x0.61434x1.25 |
| intensity min/med/max (HU) | -2793 / -881.0 / 10286 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/lung-nodule-only__valid_1061_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lung nodule**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A nonspecific nodule measuring 5 mm in size is observed in the lower lobe of the left lung (series 2, image 156). It is recommended to compare and follow-up with previous examinations, if any. Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> If there is a nonspecific nodule measuring 5 mm in the lower lobe of the left lung (series 2, image 156), it is recommended to compare and follow-up with previous examinations.

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
| Lung nodule | ✅ | **AFFIRMED** | A nonspecific nodule measuring 5 mm in size is observed in the lower lobe of the left lung (series 2, image 156). |
| Lung opacity | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`lung-nodule-only__valid_1061_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.61434x0.61434x1.25, shape 512x512x229).
- [ ] Verify HU: raw min = -2793 (−8192 padding sentinel expected; stats clip to [−1000,1000]).
