# Case sheet — `valid_1041_c` (recon 1) — group: **multi-abnormality**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1041` |
| scan | `valid_1041_c` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1041_c_1.nii.gz` |
| age / sex | 68Y / M |
| manufacturer / model | Siemens Healthineers / SOMATOM go.All |
| kernel | ['Bl56f', '3'] |
| shape (RxCxSlices) | 512x512x225 |
| voxel spacing (x,y,z mm) | 0.67188x0.67188x1.25 |
| intensity min/med/max (HU) | -2152 / -794.0 / 3952 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/multi-abnormality__valid_1041_c.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Lymphadenopathy**
- **Atelectasis**
- **Lung opacity**
- **Pulmonary fibrotic sequela**
- **Pleural effusion**
- **Peribronchial thickening**
- **Bronchiectasis**
- **Interlobular septal thickening**

## Findings (Findings_EN)

> CTO is within the normal range. In the thyroid gland, hypertrophy and mild parenchymal heterogeneity are observed in both lobes. The pulmonary arterial system calibration of the ascending-descending aorta in the mediastinum is normal. The arcus aorta calibration was measured as 29 mm and it was in the maximal physiological limit. Atherosclerotic changes are observed in mediastinal vascular structures. Multiple millimetric lymph nodes are observed in the mediastinum. The largest of the lymph nodes in the mediastinum is in the paraesophageal-subcarinal area, with dimensions of approximately 25x11 mm, although it cannot be clearly distinguished from the esophagus on non-contrast examination. According to his previous review, a progression is observed in his dimensions. When examined in the lung parenchyma window; both hemithorax are symmetrical. Calibration of the trachea and main bronchi is normal. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. Thickening of the peribronchial sheath is more prominent, especially in the mid-lower zones. It is also observed in his previous review. On the right, sequela pleuroparenchymal density increases and tractional bronchiectasis are observed at the apical level. Amorphous calcification is observed in the anterior segment caudal of the upper lobe of the right lung, and it has a stable appearance according to the previous examination. In the right lung, there is a pleural effusion reaching 20 mm in its thickest part at the base and mild atelectasis adjacent to it. It was not detected in the previous review. Sequelae changes in both lungs and thickening of peripheral interlobular septa are present at this level, and there are slight ground-glass-like density increases at this level. It is recommended to be evaluated together with the clinic in terms of interstitial fibrosis. In the evaluation of upper abdominal sections in the study area; The left lobe of the liver and the caudate lobe are prominent. Sequelae changes in the liver (especially at the apical level of the right lung) are observed and there is an accompanying tractional bronchiectasis appearance. Perihepatic level effusion is present. Millimetric calculus is observed at the neck level of the gallbladder. It was not clearly identified in the previous review. The spleen is larger than normal. The pancreas is natural. Right and left adrenals are normal. Both kidneys are reduced in size and their contours are lobulated (CVI?). Mesenteric fatty planes are contaminated. At the anterior diaphragmatic level, there are lymph nodes on both sides, the largest on the right and measuring 21x13 mm. Surrounding soft tissue plans are natural. Dorsal kyphosis was evident in the evaluation of the bone structure. Square vertebra appearance and thickening of the paravertebral longitudinal ligaments and increases in density are observed (spondyloarthropathy?).

## Impression (Impressions_EN)

> Thickening of the peribronchial sheath, thickening of the interlobular and subpleural septa, occasional accompanying faint ground-glass-like density increases. It is recommended to be evaluated together with clinical and laboratory findings in terms of interstitial fibrosis.  Effusion in the right pleural space and a thin atelectatic lung segment adjacent to it were not observed in the previous examination.  It is recommended to evaluate the liver in terms of prominence in the left lobe and caudate lobe, full appearance in the spleen, perisplenic effusion, chronic liver parenchyma disease. Perihepatic effusion was evident according to his previous examination.  Reduction in the size of both kidneys, lobulation in the contours (CRF?).  There are findings suggestive of spondyloarthropathy in the bone structure.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | — | **ABSENT** |  |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | ✅ | **ABSENT** |  |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | ✅ | **AFFIRMED** | In the right lung, there is a pleural effusion reaching 20 mm in its thickest part at the base and mild atelectasis adja… |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | ✅ | **AFFIRMED** | On the right, sequela pleuroparenchymal density increases and tractional bronchiectasis are observed at the apical level… |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | On the right, sequela pleuroparenchymal density increases and tractional bronchiectasis are observed at the apical level… |
| Pleural effusion | ✅ | **AFFIRMED** | In the right lung, there is a pleural effusion reaching 20 mm in its thickest part at the base and mild atelectasis adja… |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | ✅ | **AFFIRMED** | Thickening of the peribronchial sheath is more prominent, especially in the mid-lower zones. |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | ✅ | **AFFIRMED** | On the right, sequela pleuroparenchymal density increases and tractional bronchiectasis are observed at the apical level… |
| Interlobular septal thickening | ✅ | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`multi-abnormality__valid_1041_c.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.67188x0.67188x1.25, shape 512x512x225).
- [ ] Verify HU: raw min = -2152 (−8192 padding sentinel expected; stats clip to [−1000,1000]).
- [ ] Verify label **Lymphadenopathy**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
- [ ] Verify label **Interlobular septal thickening**: positive in CSV but NegEx='ABSENT' in report (possible silver-label FP or synonym miss).
