# Case sheet — `valid_1016_b` (recon 1) — group: **multi-abnormality**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1016` |
| scan | `valid_1016_b` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1016_b_1.nii.gz` |
| age / sex | 61Y / M |
| manufacturer / model | SIEMENS / SOMATOM Force |
| kernel | ['Bl57d', '3'] |
| shape (RxCxSlices) | 512x512x209 |
| voxel spacing (x,y,z mm) | 0.65625x0.65625x1.5 |
| intensity min/med/max (HU) | -1024 / -757.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/multi-abnormality__valid_1016_b.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lymphadenopathy**
- **Atelectasis**
- **Lung nodule**
- **Lung opacity**
- **Pleural effusion**
- **Peribronchial thickening**
- **Consolidation**
- **Interlobular septal thickening**

## Findings (Findings_EN)

> Minimal effusion was observed in both pleural spaces. Measured 20 mm on the right at its deepest point. In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated in favor of compressive atelectasis adjacent to the effusion. In the mediastinum, a lesion of soft tissue density is observed in the prevascular area, which is evaluated primarily in favor of lymphadenopathy, in which calcified foci in millimeter sizes are also observed. Although no change was found in the craniocaudal dimension in the current examination, an increase in the mediolateral dimension was noted. It was measured as 25 mm in the previous CT examination, and it was measured as 31 mm in the current examination. In addition, there are lymph nodes in the mediastinum that are stable in number and size, short in diameter less than 1 cm, have a fusiform configuration, and are not pathological in size and appearance. There are nodules in both lungs, the largest of which is in the posterobasal segment of the left lung lower lobe, some with irregular borders and some with a ground-glass halo in the periphery. No change was detected in their number and size. In addition, thickening in the peribronchovascular area and smooth interlobular septal thickness increases are observed in the anterior segment of the left lung upper lobe. The findings were also observed in the previous CT examination and no change was detected.

## Impression (Impressions_EN)

> Not given.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **UNCERTAIN** | In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated … |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | ✅ | **AFFIRMED** | In the mediastinum, a lesion of soft tissue density is observed in the prevascular area, which is evaluated primarily in… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | ✅ | **UNCERTAIN** | In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated … |
| Lung nodule | ✅ | **AFFIRMED** | There are nodules in both lungs, the largest of which is in the posterobasal segment of the left lung lower lobe, some w… |
| Lung opacity | ✅ | **AFFIRMED** | In the mediastinum, a lesion of soft tissue density is observed in the prevascular area, which is evaluated primarily in… |
| Pulmonary fibrotic sequela | — | **ABSENT** |  |
| Pleural effusion | ✅ | **ABSENT** |  |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | ✅ | **ABSENT** |  |
| Consolidation | ✅ | **UNCERTAIN** | In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated … |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | ✅ | **AFFIRMED** | In addition, thickening in the peribronchovascular area and smooth interlobular septal thickness increases are observed … |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`multi-abnormality__valid_1016_b.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.65625x0.65625x1.5, shape 512x512x209).
- [ ] Verify label **Atelectasis**: positive in CSV but NegEx='UNCERTAIN' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Pleural effusion**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Peribronchial thickening**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Consolidation**: positive in CSV but NegEx='UNCERTAIN' in report (possible silver-label FP or synonym miss).
