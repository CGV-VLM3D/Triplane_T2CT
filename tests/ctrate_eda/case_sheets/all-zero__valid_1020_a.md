# Case sheet — `valid_1020_a` (recon 1) — group: **all-zero**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1020` |
| scan | `valid_1020_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1020_a_1.nii.gz` |
| age / sex | 31Y / F |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x218 |
| voxel spacing (x,y,z mm) | 0.30859x0.30859x1.5 |
| intensity min/med/max (HU) | -1024 / -228.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/all-zero__valid_1020_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

(none)

## Findings (Findings_EN)

> CTO is normal. In the anterior mediastinum, thymic tissue with trigonal configuration without mass effect is observed. Arkus oarta calibration is natural. Calibration of mediastinal other moment vascular structures is natural. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No lymph node with pathological size and configuration was detected at the mediastinal and hilar level. When examined in the lung parenchyma window; Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. No bilateral pleural effusion or pneumothorax was detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Nodular dance compatible with the accessory spleen is also observed in the spleen hilum. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Surrounding soft tissue plans are natural. Mild degenerative changes are observed in the bone structure.

## Impression (Impressions_EN)

> No finding compatible with pneumonia was detected.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No lymph node with pathological size and configuration was detected at the mediastinal and hilar level. |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **UNCERTAIN** | Nodular dance compatible with the accessory spleen is also observed in the spleen hilum. |
| Lung opacity | — | **NEGATED** | Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | No bilateral pleural effusion or pneumothorax was detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`all-zero__valid_1020_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.30859x0.30859x1.5, shape 1024x1024x218).
- [ ] Verify truly-negative: scan is label-negative; confirm no subtle finding was missed.
