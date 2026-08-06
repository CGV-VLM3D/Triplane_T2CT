# Case sheet — `valid_1103_b` (recon 1) — group: **medical-material**

## Identity & acquisition

| field | value |
|---|---|
| patient | `valid_1103` |
| scan | `valid_1103_b` |
| reconstruction | `1`  (available: 1,2) |
| primary file | `valid_1103_b_1.nii.gz` |
| age / sex | 35Y / F |
| manufacturer / model | SIEMENS / SOMATOM Force |
| kernel | ['Bl57d', '3'] |
| shape (RxCxSlices) | 512x512x200 |
| voxel spacing (x,y,z mm) | 0.74805x0.74805x1.5 |
| intensity min/med/max (HU) | -1024 / -907.0 / 3071 |
| no_chest excluded set | no |

## Montage (axial / coronal / sagittal x lung / mediastinal)

![montage](../figures/montages/medical-material__valid_1103_b.png)

_Analysis unit: single reconstruction (volume). Orientation LPS; out-of-plane panels aspect-corrected by z/xy spacing._

## Positive silver-labels (from *_predicted_labels.csv)

- **Medical material**
- **Pulmonary fibrotic sequela**

## Findings (Findings_EN)

> The mediastinal main vascular structures and the heart could not be evaluated optimally due to the lack of IV contrast, and as far as can be observed, the calibration of the vascular structures, the heart contour and size are natural. No pericardial, pleural effusion or thickening was detected. Trachea, both main bronchi are open and no occlusive pathology is detected. No pathological increase in thoracic esophagus wall thickness is observed. In the mediastinum, in both axillary regions and in the supraclavicular fossa, no lymph nodes are observed in pathological size and appearance. When examined in the lung parenchyma window; No active infiltrative or mass lesion was detected in both lung parenchyma. The middle lobe of the right lung is not observed secondary to the operation, and its bronchus ends in a stump, and surgical suture materials are observed around the stump. In the right lung upper lobe posterior segment, there are suture materials and fibrotic recessions in the vicinity of the suture material, extending along the major fissure, causing structural distortion and minimal volume loss in the parenchyma. In the upper abdominal sections within the image, no pathology was detected as far as can be observed within the borders of non-contrast CT. No lytic-destructive lesion was observed in the bone structures within the image.

## Impression (Impressions_EN)

> Surgical suture materials extending along the major fissure in the upper lobe posterior segment, and fibrotic recessions in the vicinity of the suture materials, structural distortion in the patient who was found to have undergone right lung middle lobectomy; findings are also present in the previous CT examination. No newly developed pathology was detected in the current examination.

## NegEx assertion status of each label's term in the report

| label | in CSV? | NegEx status | evidence sentence |
|---|:---:|:---:|---|
| Medical material | ✅ | **AFFIRMED** | In the right lung upper lobe posterior segment, there are suture materials and fibrotic recessions in the vicinity of th… |
| Arterial wall calcification | — | **ABSENT** |  |
| Cardiomegaly | — | **ABSENT** |  |
| Pericardial effusion | — | **ABSENT** |  |
| Coronary artery wall calcification | — | **ABSENT** |  |
| Hiatal hernia | — | **ABSENT** |  |
| Lymphadenopathy | — | **NEGATED** | In the mediastinum, in both axillary regions and in the supraclavicular fossa, no lymph nodes are observed in pathologic… |
| Emphysema | — | **ABSENT** |  |
| Atelectasis | — | **ABSENT** |  |
| Lung nodule | — | **ABSENT** |  |
| Lung opacity | — | **NEGATED** | No active infiltrative or mass lesion was detected in both lung parenchyma. |
| Pulmonary fibrotic sequela | ✅ | **AFFIRMED** | In the right lung upper lobe posterior segment, there are suture materials and fibrotic recessions in the vicinity of th… |
| Pleural effusion | — | **NEGATED** | No pericardial, pleural effusion or thickening was detected. |
| Mosaic attenuation pattern | — | **ABSENT** |  |
| Peribronchial thickening | — | **ABSENT** |  |
| Consolidation | — | **ABSENT** |  |
| Bronchiectasis | — | **ABSENT** |  |
| Interlobular septal thickening | — | **ABSENT** |  |

## Human review needed (pointers only — NO medical adjudication)

- [ ] Verify montage orientation & that lung fields / mediastinum are visible (`medical-material__valid_1103_b.png`).
- [ ] Verify spacing/shape are plausible for a chest CT (spacing 0.74805x0.74805x1.5, shape 512x512x200).
