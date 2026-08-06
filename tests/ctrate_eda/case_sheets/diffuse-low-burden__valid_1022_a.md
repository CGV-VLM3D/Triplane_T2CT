# Case sheet — `valid_1022_a` (recon 1) — group: **diffuse-low-burden**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1022` |
| scan | `valid_1022_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1022_a_1.nii.gz` |
| age / sex | 28Y / M |
| manufacturer / model | Philips / Brilliance Big Bore |
| kernel | A |
| shape (RxCxSlices) | 512x512x232 |
| voxel spacing (x,y,z mm) | 0.76562x0.76562x1.5 |
| intensity min/med/max (HU) | -1024 / -802.0 / 1761 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/diffuse-low-burden__valid_1022_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Pulmonary fibrotic sequela**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. The mediastinum could not be evaluated optimally in the non-contrast examination. As far as can be observed: mediastinal main vascular structures, heart contour, size is normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Linear fibroatelactastic changes were observed in the right lung middle lobe medial and left lung upper lobe inferior lingular segment. Apart from this, both lung parenchyma aeration is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. Liver parenchymal density is diffusely decreased, consistent with hepatosteatosis. Two accessory spleens with diameters of 7 and 14 mm were observed in the upper pole anterior of the spleen. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

## Impression (Impressions_EN)

> Linear fibroatelactasis changes in right lung middle lobe medial and left lung upper lobe inferior lingular segment . Hepatosteatosis . Two accessory spleens in anterior upper pole of spleen.  Scoliosis with thoracic opening facing left

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **UNCERTAIN** | Liver parenchymal density is diffusely decreased, consistent with hepatosteatosis. |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion-thickening was not observed. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **NEGATED** | Apart from this, both lung parenchyma aeration is normal and no nodular or infiltrative lesion is detected in the lung p… |
| Lung opacity | — | **UNCERTAIN** | Liver parenchymal density is diffusely decreased, consistent with hepatosteatosis. |
| Pulmonary fibrotic sequela | ✅ | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`diffuse-low-burden__valid_1022_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.76562x0.76562x1.5, shape 512x512x232).
- [ ] Verify label **Pulmonary fibrotic sequela**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
