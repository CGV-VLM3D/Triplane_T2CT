# Case sheet — `valid_103_a` (recon 1) — group: **multi-abnormality**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_103` |
| scan | `valid_103_a` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_103_a_1.nii.gz` |
| age / sex | 41Y / M |
| manufacturer / model | Philips / iCT 256 |
| kernel | YA |
| shape (RxCxSlices) | 1024x1024x236 |
| voxel spacing (x,y,z mm) | 0.45312x0.45312x1.5 |
| intensity min/med/max (HU) | -1024 / -938.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/multi-abnormality__valid_103_a.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**
- **Cardiomegaly**
- **Lymphadenopathy**
- **Emphysema**
- **Lung opacity**
- **Pleural effusion**
- **Consolidation**
- **Interlobular septal thickening**

## Findings (Findings_EN)

> Trachea and lumen of both main bronchi are open. No occlusive pathology was detected in the trachea and lumen of both main bronchi. Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be observed: Calibration of thoracic main vascular structures is natural. Heart size increased. Pericardial thickening-effusion was not detected. Prosthetic material was observed in the aortic valve. There is post-op suture material on the wall of the ascending aorta. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. Lymph nodes measuring 19x11 mm in size were observed in the upper-lower paratracheal, prevascular, precarinal, and subcarinal localizations. When both lung parenchyma windows were evaluated, patchy areas of consolidation extending to the periphery and accompanying ground glass density increases were observed in the perihilar area of both lungs. The appearance suggests an infectious process in the first place. Clinical and laboratory correlation is recommended. In addition, smooth interseptal thickenings were observed in the intersepta, which became prominent in the lower lobes of both lungs (secondary to cardiac pathology?). Free fluid was observed between the pleural leaves on the right, with a thickness of 24 mm, and on the left, measuring 5 mm. Both fissures are observed as thick. In both lung parenchyma, no significant mass lesion was detected in the non-enhanced examination limits. Emphysematous changes were observed in both lungs. Upper abdominal sections entering the examination area are natural. Bilateral adrenal gland calibration was normal and no space-occupying lesion was detected. Metallic suture materials of sternotomy were observed in the sternum. No lytic-destructive lesion was detected in bone structures.

## Impression (Impressions_EN)

> Cardiomegaly. Patchy areas of consolidation in both lungs extending from the diffuse perihilar area to the periphery and accompanying ground-glass density increases. The appearance was initially evaluated in favor of the infectious process. Clinical and laboratory correlation is recommended.  Bilateral diffuse uniform interlobular septal thickening (secondary to cardiac pathology?) . Bilateral pleural effusion . Mild emphysematous changes in both lungs

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **AFFIRMED** | There is post-op suture material on the wall of the ascending aorta. |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | ✅ | **AFFIRMED** | Cardiomegaly. |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | ✅ | **ABSENT** |  |
| Emphysema | ✅ | **AFFIRMED** | Emphysematous changes were observed in both lungs. |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | ✅ | **AFFIRMED** | When both lung parenchyma windows were evaluated, patchy areas of consolidation extending to the periphery and accompany… |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | ✅ | **AFFIRMED** | Bilateral pleural effusion . |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | ✅ | **AFFIRMED** | When both lung parenchyma windows were evaluated, patchy areas of consolidation extending to the periphery and accompany… |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | ✅ | **AFFIRMED** | In addition, smooth interseptal thickenings were observed in the intersepta, which became prominent in the lower lobes o… |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`multi-abnormality__valid_103_a.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.45312x0.45312x1.5, shape 1024x1024x236).
- [ ] Verify label **Lymphadenopathy**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
