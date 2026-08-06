# Case sheet — `valid_1016_d` (recon 1) — group: **multi-abnormality**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1016` |
| scan | `valid_1016_d` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1016_d_1.nii.gz` |
| age / sex | 61Y / M |
| manufacturer / model | Siemens Healthineers / SOMATOM go.All |
| kernel | ['Br40f', '3'] |
| shape (RxCxSlices) | 512x512x295 |
| voxel spacing (x,y,z mm) | 0.68771x0.68771x1 |
| intensity min/med/max (HU) | -8192 / -874.0 / 1856 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/multi-abnormality__valid_1016_d.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Arterial wall calcification**
- **Coronary artery wall calcification**
- **Lymphadenopathy**
- **Emphysema**
- **Atelectasis**
- **Lung nodule**
- **Pulmonary fibrotic sequela**
- **Pleural effusion**

## Findings (Findings_EN)

> Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. Pericardial effusion was not detected. There is minimal pleural effusion on the right. No pleural effusion was detected on the left. The widths of the mediastinal main vascular structures are normal. There are atheromatous plaques in the coronary arteries. There are lymph nodes in the mediastinum and hilar regions. The largest of these lymph nodes is observed in the right hilar region and its short diameter is 9 mm. An irregularly circumscribed mass is observed adjacent to the prevascular region in the medial of the upper lobe of the left lung. The longest diameter of the mass was 48 mm. There is no pathological wall thickness increase in the esophagus within the sections. No occlusive pathology was detected in the trachea and both main bronchi. Emphysematous changes and occasional atelectasis and minimal pleuroparenchymal sequelae were observed in both lungs. There are multiple nodules in both lungs. The largest of these nodules is observed in the lower lobe of the left lung and the longest diameter was 12 mm. No appearance that can be evaluated in favor of pneumonic infiltration was observed in both lungs. There is no upper abdominal free fluid-collection within the sections. There are no fractures or lytic-destructive lesions in the bone structures within the sections.

## Impression (Impressions_EN)

> Mass in the medial part of the upper lobe of the left lung, multiple nodules in both lungs.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | ✅ | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion was not detected. |
| Coronary artery wall calcification | ✅ | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | ✅ | **ABSENT** |  |
| Emphysema | ✅ | **AFFIRMED** | Emphysematous changes and occasional atelectasis and minimal pleuroparenchymal sequelae were observed in both lungs. |
| Atelectasis | ✅ | **AFFIRMED** | Emphysematous changes and occasional atelectasis and minimal pleuroparenchymal sequelae were observed in both lungs. |
| Lung nodule | ✅ | **AFFIRMED** | There are multiple nodules in both lungs. |
| Lung opacity | — | **NEGATED** | No appearance that can be evaluated in favor of pneumonic infiltration was observed in both lungs. |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | Emphysematous changes and occasional atelectasis and minimal pleuroparenchymal sequelae were observed in both lungs. |
| Pleural effusion | ✅ | **AFFIRMED** | There is minimal pleural effusion on the right. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`multi-abnormality__valid_1016_d.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.68771x0.68771x1, shape 512x512x295).
- [ ] Verify HU: raw min = -8192 (−8192 padding sentinel expected; stats clip to [−1000,1000]).
- [ ] Verify label **Arterial wall calcification**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Coronary artery wall calcification**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Lymphadenopathy**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
