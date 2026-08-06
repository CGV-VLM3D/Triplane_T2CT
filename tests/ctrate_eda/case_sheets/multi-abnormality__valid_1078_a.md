# Case sheet — `valid_1078_a` (recon 1) — group: **multi-abnormality**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1078` |
| scan | `valid_1078_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1078_a_1.nii.gz` |
| age / sex | 73Y / F |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x226 |
| voxel spacing (x,y,z mm) | 0.37012x0.37012x1.5 |
| intensity min/med/max (HU) | -1024 / -855.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/multi-abnormality__valid_1078_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Arterial wall calcification**
- **Cardiomegaly**
- **Coronary artery wall calcification**
- **Emphysema**
- **Lung nodule**
- **Lung opacity**
- **Pleural effusion**
- **Consolidation**

## Findings (Findings_EN)

> Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: The heart is minimally larger than normal. Pericardial effusion was not detected. The widths of the mediastinal main vascular structures are normal. Atheroma plaques were observed in the aorta and coronary arteries. Pleural effusion is observed on the left. The pleural effusion measured 70 mm at its thickest point. There is no pleural effusion on the right. There are no pathologically enlarged lymph nodes in the mediastinum and hilar regions. No pathological increase in wall thickness was detected in the esophagus within the sections. Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. Consolidation and ground-glass appearances were observed in the posterior part of the lower lobe of the right lung, the lower lobe of the left lung, and the apicoposterior segment of the upper lobe. The described manifestations were primarily evaluated in favor of pneumonic infiltration. There are emphysematous changes in both aerated lungs. There are several millimeric nonspecific nodules in both lungs. No mass or infiltrative lesion was detected in both lungs. No fractures or lytic-destructive lesions were observed in the bone structures within the sections.

## Impression (Impressions_EN)

> Follow-up over ca.  Left pleural effusion.  Findings evaluated primarily in favor of pneumonic infiltration in both lungs.  Emphysematous changes in both lungs.  Millimetric nonspecific nodules in both lungs.  Atherosclerotic changes in the aorta and coronary arteries.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | ✅ | **ABSENT** |  |
| Cardiomegaly | ✅ | **ABSENT** |  |
| Pericardial effusion | — | **NEGATED** | Pericardial effusion was not detected. |
| Coronary artery wall calcification | ✅ | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | There are no pathologically enlarged lymph nodes in the mediastinum and hilar regions. |
| Emphysema | ✅ | **AFFIRMED** | There are emphysematous changes in both aerated lungs. |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | ✅ | **AFFIRMED** | There are several millimeric nonspecific nodules in both lungs. |
| Lung opacity | ✅ | **AFFIRMED** | Consolidation and ground-glass appearances were observed in the posterior part of the lower lobe of the right lung, the … |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | ✅ | **AFFIRMED** | Pleural effusion is observed on the left. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | ✅ | **AFFIRMED** | Consolidation and ground-glass appearances were observed in the posterior part of the lower lobe of the right lung, the … |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`multi-abnormality__valid_1078_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.37012x0.37012x1.5, shape 1024x1024x226).
- [ ] Verify label **Arterial wall calcification**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Cardiomegaly**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Coronary artery wall calcification**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
