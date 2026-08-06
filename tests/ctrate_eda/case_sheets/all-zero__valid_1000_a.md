# Case sheet — `valid_1000_a` (recon 1) — group: **all-zero**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1000` |
| scan | `valid_1000_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1000_a_1.nii.gz` |
| age / sex | 42Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x194 |
| voxel spacing (x,y,z mm) | 0.39355x0.39355x1.5 |
| intensity min/med/max (HU) | -1024 / -875.0 / 2285 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/all-zero__valid_1000_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

(none)

## Findings (Findings_EN)

> Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions. No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No enlarged lymph nodes in pathological dimensions were detected. In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections.

## Impression (Impressions_EN)

> Findings within normal limits.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | No pleural or pericardial effusion was detected. |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions. |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | — | **NEGATED** | No mass or infiltrative lesion was detected in both lungs. |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | — | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`all-zero__valid_1000_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.39355x0.39355x1.5, shape 1024x1024x194).
- [ ] Verify truly-negative: scan is label-negative; confirm no subtle finding was missed.
