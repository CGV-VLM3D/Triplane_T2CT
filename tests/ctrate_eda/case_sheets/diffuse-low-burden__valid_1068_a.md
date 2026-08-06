# Case sheet — `valid_1068_a` (recon 1) — group: **diffuse-low-burden**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1068` |
| scan | `valid_1068_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1068_a_1.nii.gz` |
| age / sex | 45Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x209 |
| voxel spacing (x,y,z mm) | 0.34961x0.34961x1.5 |
| intensity min/med/max (HU) | -1024 / -281.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/diffuse-low-burden__valid_1068_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Pulmonary fibrotic sequela**

## Findings (Findings_EN)

> CTO is within normal limits. Calibration of mediastinal major vascular structures is natural. No pathologically sized and configured lymph nodes were detected at both hilar levels in the mediastinum. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. When examined in the lung parenchyma window; Calibration of trachea, both main bronchi is natural. Lumens are clear. On the right, azygos fissure variation is observed. Density increases consistent with pleuroparenchymal sequelae are observed in the lingular segment on the right. No nodular or infiltrative lesion was detected in both lung parenchyma. Pleural effusion-thickening was not detected. In the upper abdominal organs included in the sections, there is a hypodense appearance that may be compatible with a parapelvic cyst at the level of the left kidney superior pole. Nodular density is observed in the spleen hilum, which is considered to be compatible with the accessory spleen with a diameter of approximately 8 mm. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> Mild sequelae changes in the middle lobe of the right lung, azygos fissure variation in the upper lobe on the right. Hypodense appearance that may be compatible with parapelvic cyst at the level of the left kidney superior pole.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **UNCERTAIN** | Density increases consistent with pleuroparenchymal sequelae are observed in the lingular segment on the right. |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **ABSENT** |  |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **AFFIRMED** | Nodular density is observed in the spleen hilum, which is considered to be compatible with the accessory spleen with a d… |
| Lung opacity | — | **AFFIRMED** | Nodular density is observed in the spleen hilum, which is considered to be compatible with the accessory spleen with a d… |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | Mild sequelae changes in the middle lobe of the right lung, azygos fissure variation in the upper lobe on the right. |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`diffuse-low-burden__valid_1068_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.34961x0.34961x1.5, shape 1024x1024x209).
- [ ] Verify **Lung nodule**: AFFIRMED in report but label negative (possible silver-label FN). Ev: "Nodular density is observed in the spleen hilum, which is considered to be compatible with…"
- [ ] Verify **Lung opacity**: AFFIRMED in report but label negative (possible silver-label FN). Ev: "Nodular density is observed in the spleen hilum, which is considered to be compatible with…"
