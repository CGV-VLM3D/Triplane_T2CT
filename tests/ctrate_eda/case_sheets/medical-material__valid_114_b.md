# Case sheet — `valid_114_b` (recon 1) — group: **medical-material**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_114` |
| scan | `valid_114_b` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_114_b_1.nii.gz` |
| age / sex | 51Y / F |
| manufacturer / model | SIEMENS / SOMATOM Force |
| kernel | ['Bl57d', '3'] |
| shape (RxCxSlices) | 512x512x206 |
| voxel spacing (x,y,z mm) | 0.54883x0.54883x1.5 |
| intensity min/med/max (HU) | -1024 / -893.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/medical-material__valid_114_b.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**
- **Lung nodule**

## Findings (Findings_EN)

> On the right, the port chamber and the image of the catheter extending to the superior vena cava are seen on the anterior chest wall. Trachea and both main bronchi were in the midline and no obstructive pathology was observed in the lumen. In the non-contrast examination, the mediastinal could not be evaluated optimally. As far as can be seen; mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific parenchymal nodules were observed in both lungs. No mass lesion-pneumonic infiltration with distinguishable borders was detected in the lung parenchyma. As far as can be seen within the sections; upper abdominal organs are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Intra-abdominal solid organs were evaluated in detail in MR examination.

## Impression (Impressions_EN)

> Millimetric stable parenchymal nodules in both lungs

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **AFFIRMED** | On the right, the port chamber and the image of the catheter extending to the superior vena cava are seen on the anterio… |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion-thickening was not observed. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | ✅ | **AFFIRMED** | A few millimetric nonspecific parenchymal nodules were observed in both lungs. |
| Lung opacity | — | **NEGATED** | No mass lesion-pneumonic infiltration with distinguishable borders was detected in the lung parenchyma. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`medical-material__valid_114_b.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.54883x0.54883x1.5, shape 512x512x206).
