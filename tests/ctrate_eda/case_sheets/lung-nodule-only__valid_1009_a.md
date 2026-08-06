# Case sheet — `valid_1009_a` (recon 1) — group: **lung-nodule-only**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1009` |
| scan | `valid_1009_a` |
| reconstruction | `1`  (available: 1) |
| primary file | `valid_1009_a_1.nii.gz` |
| age / sex | 43Y / F |
| manufacturer / model | PNMS / MX 16 |
| kernel | EA |
| shape (RxCxSlices) | 768x768x218 |
| voxel spacing (x,y,z mm) | 0.53385x0.53385x1.5 |
| intensity min/med/max (HU) | -1024 / -187.0 / 1925 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/lung-nodule-only__valid_1009_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lung nodule**

## Findings (Findings_EN)

> Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. There is a 10 mm hypodense oval-shaped finding in the lower quadrant of the left breast. lymph node? When examined in the lung parenchyma window; There are several millimetric non-specific nodules in both lungs. Both lung parenchyma aeration is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. In the upper abdominal organs included in the sections, the liver parenchyma changes in favor of steatosis. The bilateral adrenal glands are normal. No space-occupying lesions were detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Diffuse density reduction in bone structures and mild hypertrophic tapering in the end plates are observed.

## Impression (Impressions_EN)

> There is a 10 mm hypodense oval-shaped finding in the lower quadrant of the left breast. Lymph node?. There are several millimetric non-specific nodules in both lungs. Hepatosteatosis.

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
| Lung nodule | ✅ | **AFFIRMED** | There are several millimetric non-specific nodules in both lungs. |
| Lung opacity | — | **AFFIRMED** | Diffuse density reduction in bone structures and mild hypertrophic tapering in the end plates are observed. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **NEGATED** | Pleural effusion-thickening was not detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`lung-nodule-only__valid_1009_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.53385x0.53385x1.5, shape 768x768x218).
- [ ] Verify **Lung opacity**: AFFIRMED in report but label negative (possible silver-label FN). Ev: "Diffuse density reduction in bone structures and mild hypertrophic tapering in the end pla…"
