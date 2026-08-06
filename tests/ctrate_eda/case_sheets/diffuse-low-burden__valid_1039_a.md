# Case sheet — `valid_1039_a` (recon 1) — group: **diffuse-low-burden**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1039` |
| scan | `valid_1039_a` |
| reconstruction | `1`  (available: 1) |
| primary file | `valid_1039_a_1.nii.gz` |
| age / sex | 39Y / F |
| manufacturer / model | PNMS / MX 16 |
| kernel | EA |
| shape (RxCxSlices) | 768x768x246 |
| voxel spacing (x,y,z mm) | 0.44531x0.44531x1.5 |
| intensity min/med/max (HU) | -1024 / -802.0 / 2015 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/diffuse-low-burden__valid_1039_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Pulmonary fibrotic sequela**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; In sections passing through the upper part of the left lung, subpleural sequelae fibrotic changes are observed in the major fissure, lower lobe laterobasal and right lung lower lobe laterobasal. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> Millimetric sequela fibrotic changes in bilateral lungs.

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
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | — | **ABSENT** |  |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | In sections passing through the upper part of the left lung, subpleural sequelae fibrotic changes are observed in the ma… |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`diffuse-low-burden__valid_1039_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.44531x0.44531x1.5, shape 768x768x246).
