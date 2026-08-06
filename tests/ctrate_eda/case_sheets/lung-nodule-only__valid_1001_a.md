# Case sheet — `valid_1001_a` (recon 1) — group: **lung-nodule-only**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1001` |
| scan | `valid_1001_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1001_a_1.nii.gz` |
| age / sex | 40Y / M |
| manufacturer / model | Philips / Brilliance Big Bore |
| kernel | A |
| shape (RxCxSlices) | 512x512x213 |
| voxel spacing (x,y,z mm) | 0.68359x0.68359x1.5 |
| intensity min/med/max (HU) | -1024 / -841.0 / 1738 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/lung-nodule-only__valid_1001_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lung nodule**

## Findings (Findings_EN)

> Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. There are millimetric nonspecific nodules in both lungs. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No enlarged lymph nodes in pathological size and appearance were detected in the mediastinum and hilar regions. No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No pathologically enlarged lymph nodes were observed. In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections.

## Impression (Impressions_EN)

> Millimetric nodules in both lungs

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | No pleural or pericardial effusion was detected. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No enlarged lymph nodes in pathological size and appearance were detected in the mediastinum and hilar regions. |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | ✅ | **AFFIRMED** | There are millimetric nonspecific nodules in both lungs. |
| Lung opacity | — | **NEGATED** | No mass or infiltrative lesion was detected in both lungs. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`lung-nodule-only__valid_1001_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.68359x0.68359x1.5, shape 512x512x213).
