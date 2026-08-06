# CT-RATE hands-on — 로컬 3D 뷰어 버전 (ipynb 대체)

ipympl 노트북이 브라우저 리드로우 때문에 느려서, **볼륨을 로컬로 받아 네이티브 3D 뷰어로
보는 절차**입니다. 각 케이스마다: 어떤 파일을, 어떤 윈도우로, 무엇을 볼지 + 짝이 되는
리포트/라벨을 함께 적었습니다. (노트북 `06/07/08`의 내용을 뷰어 무관 형태로 옮긴 것)

---

## 0. 뷰어 선택

| 방식 | 추천 | 이유 |
|------|------|------|
| **로컬 MITK / 3D Slicer / ITK-SNAP** | ✅ **1순위** | `.nii.gz` 네이티브, GPU 슬라이스 스크롤 즉시, L/W 조절 편함. 이미 MITK 있으면 그대로. |
| 서버 웹뷰어 (NiiVue / itk-vtk-viewer) | △ 대안 | 다운로드 못 할 때. 브라우저 WebGL이라 ipympl보단 빠름. 서버에 정적 http 서빙 필요. |
| SimpleITK `sitk.Show()` | ✗ | 내부적으로 ImageJ/Fiji를 띄움 → ImageJ 설치 + X 디스플레이 필요. 독립 GUI 아님. |

> `num_workers`/MONAI dataloader는 **여기에 안 맞습니다** — 단일 파일 조회라 배치가 없고,
> 하나의 gzip 스트림은 순차 압축해제라 워커로 못 쪼갭니다. 병목은 파일당 ~3초 압축해제.

## 1. 파일 받기

서버 경로: `/workspace/tests/ctrate_eda_bundle/files/<group>/<name>.nii.gz`

- **VS Code Remote**: Explorer에서 파일 우클릭 → **Download**.
- **scp**: `scp <user>@<host>:/workspace/tests/ctrate_eda_bundle/files/lung-nodule-only/valid_1001_a_1.nii.gz .`
- **여러 개 묶어서**: 원하면 서버에서 tar로 묶어 드립니다(요청하세요) → 한 파일로 다운로드.

**빠른 시작 세트(작은 512³ 3개, ~212MB)**: 케이스 2·3·5 (`valid_1001_a_1`, `valid_1022_a_1`,
`valid_1016_b_1`). 이것만 먼저 받아 감을 잡고, 나머지는 필요할 때 받으세요.

## 2. 윈도우(Level / Window) — MITK 기준

**개념**: CT는 복셀을 HU(−1000~+3000+)로 저장하는데 화면은 회색조가 256단계뿐 → **"어느 HU
구간만 검정→흰색으로 펼칠지"** 정하는 게 Level/Window. **Level(WL)** = 그 구간의 중심 HU(밝기),
**Window(WW)** = 구간의 폭(대비). 보이는 범위 = `[Level − Window/2, Level + Window/2]`, 그 아래는
전부 검정·위는 전부 흰색. (자세히는 §6.0)

로컬 뷰어에서 폐 볼 땐 lung, 심장/물질 볼 땐 mediastinal, 뼈/금속 볼 땐 bone으로 **수동 전환**:

| 이름 | Level | Window | 보이는 HU | 용도 |
|------|------:|-------:|-----------|------|
| lung        | −600 | 1500 | −1350 ~ +150 | 폐 실질·결절 |
| mediastinal |   40 |  400 | −160 ~ +240  | 연부조직·물질·심장 |
| bone        |  400 | 1800 | −500 ~ +1300 | 골격·금속 |

**MITK에서 조절** (MITK는 이 용어를 그대로 "Level/Window"로 씀):
- **Level/Window 위젯** — 렌더 창 오른쪽 가장자리의 세로 막대(핸들 2개). 막대 전체를 위아래로
  끌면 **Level**, 핸들 간격을 늘리고 줄이면 **Window**.
- **마우스** — 2D 뷰에서 **오른쪽 버튼 드래그**: 보통 **좌우 = Window, 상하 = Level**(제일 빠름).
- **정확한 값 입력** — Level/Window 위젯의 **숫자를 클릭/더블클릭**해 위 표 값을 직접 타이핑
  (예: `Level −600, Window 1500`).

> ⚠ **센티널 파일 2개** — `all-zero/valid_1025_b_1`, `multi-abnormality/valid_1016_d_1`
> 은 FOV 밖이 `−8192`로 채워져 있어 MITK가 **자동으로 잡는 L/W가 새까맣게 깨집니다.** 반드시 위
> 값을 **수동 입력**하세요. (나머지 28개는 min −1024라 MITK 자동값도 대체로 OK)

## 3. 케이스 워크스루 (노트북 07 아틀라스 = 헤드라인 6개)

각 케이스: **파일 → 윈도우 → 볼 것 → 라벨/결론**. 뷰어에서 축상면(axial)을 스크롤하며 확인.

### 1) 정상 (기준)
- 파일: `all-zero/valid_1000_a_1.nii.gz` (246MB, 1024³) · *가벼운 대안 recon_2: `valid_1000_a_2.nii.gz` (112MB, 512³)*
- 윈도우: **lung**
- 볼 것: 깨끗한 폐야, 국소 음영·결절 없음. **이후 모든 케이스의 비교 기준.**
- 라벨: (없음) · IMPRESSION: *"Findings within normal limits."*

### 2) 폐결절 (Lung nodule)
- 파일: `lung-nodule-only/valid_1001_a_1.nii.gz` (66MB, 512³) ✅ 작음
- 윈도우: **lung**
- 볼 것: 양쪽 폐의 millimetric 비특이 결절 — 둥근 국소 연부조직 음영. lung 윈도우로 스크롤하며 찾기.
- 라벨: **Lung nodule** · IMPRESSION: *"Millimetric nodules in both lungs"*

### 3) 폐 섬유화 후유증 (Pulmonary fibrotic sequela)
- 파일: `diffuse-low-burden/valid_1022_a_1.nii.gz` (73MB, 512³) ✅ 작음
- 윈도우: **lung**
- 볼 것: 우중엽 내측 + 좌상엽 하설상분절의 **선상 섬유무기폐(fibroatelectasis)**. 뚜렷한 종괴가 아니라 말초 망상 음영(reticulation)/구조 견인.
- 라벨: **Pulmonary fibrotic sequela**

### 4) 의료 물질 (Medical material)
- 파일: `medical-material/valid_1288_a_1.nii.gz` (356MB, 1024³) — 큼
- 윈도우: **mediastinal** 또는 **bone**
- 볼 것: 고-HU 금속성 물질(스텐트/포트/와이어). lung 윈도우에선 잘 안 보임 → bone/mediastinal로 밝은 금속 확인.
- 라벨: **Medical material**

### 5) 다발성 이상 (림프절/무기폐)
- 파일: `multi-abnormality/valid_1016_b_1.nii.gz` (73MB, 512³) ✅ 작음
- 윈도우: **mediastinal** ↔ **lung** 번갈아
- 볼 것: 양측 흉막 삼출(우측 최대 20mm), 삼출 인접 **압박성 무기폐/경화**, prevascular 연부조직 종괴(림프절병증). 라벨과 하나씩 대조.
- 라벨(8): **Lymphadenopathy, Atelectasis, Lung nodule, Lung opacity, Pleural effusion, Peribronchial thickening, Consolidation, Interlobular septal thickening**

### 6) 심비대 + 관상동맥 석회화
- 파일: `multi-abnormality/valid_1078_a_1.nii.gz` (293MB, 1024³) — 큼
- 윈도우: **mediastinal**
- 볼 것: 확대된 심장 음영, 대동맥·관상동맥 **석회화(밝은 반점)**, 좌측 흉막 삼출, 폐기종/결절.
- 라벨(8): **Arterial wall calcification, Cardiomegaly, Coronary artery wall calcification, Emphysema, Lung nodule, Lung opacity, Pleural effusion, Consolidation**

## 4. 부록 — 더 볼거리

**(06) 그룹별 확장** — 각 그룹 5개씩, 같은 소견의 변주를 비교:
- all-zero: `valid_1000_a_1, 1010_a_1, 1012_a_1, 1020_a_1, 1025_b_1`(⚠센티널)
- lung-nodule-only: `valid_1001_a_1, 1009_a_1, 1061_a_1, 1077_a_1, 1085_a_1`
- diffuse-low-burden: `valid_1022_a_1, 1039_a_1, 1068_a_1, 1073_a_1, 1153_a_1`
- multi-abnormality: `valid_1016_b_1, 1016_d_1`(⚠센티널)`, 103_a_1, 1041_c_1, 1078_a_1`
- medical-material: `valid_1288_a_1, 225_b_1, 366_a_1, 1103_b_1, 114_b_1`

**(08) reconstruction 쌍** — 같은 scan의 `_1` vs `_2`를 두 창에 띄우고 같은 해부 레벨에서
**엣지 선명도/노이즈 질감** 비교 = 커널 차이. 번들에 쌍이 있는 것:
`valid_1000_a`, `valid_1001_a`, `valid_1022_a`, `valid_1016_b`, `valid_1288_a` (각 `_1.nii.gz` / `_2.nii.gz`).

## 5. 리포트/라벨을 나중에 또 보려면

서버에서 임의 볼륨의 소견/결론/라벨만 텍스트로 뽑기:

```bash
cd /workspace
python -c "
from src.data.ct_rate_datamodule import load_records
r = {x.volume_name: x for x in load_records('valid')}['valid_1078_a_1.nii.gz']
print('LABELS:', [k for k,v in r.labels.items() if v])
print('FINDINGS:', r.findings)
print('IMPRESSION:', r.impression)
"
```

> 단위: `.nii.gz` 하나 = reconstruction(볼륨). 리포트+18 라벨은 scan 단위(한 scan의 recon들이 공유).
> `_fixed`는 HU가 이미 반영됨 → rescale slope/intercept 적용 금지.

---

# 6. 초보자용 — CT 읽기 기초 + 카테고리별 실제 리포트 (각 3개)

> "뭐가 뭔지 하나도 모르겠다"에서 출발하는 사람용. 먼저 **6.0 기초**를 읽고,
> 그다음 카테고리별 3개 케이스를 위에서부터 뷰어로 열어 보세요. 각 케이스는
> **영문 원본 리포트** + **한글 이상소견 정리** + **CT에서 어디를 어떻게 볼지**로 구성.

## 6.0 CT 읽기 기초 (이것부터)

**세 방향(면)**
- **axial(축상, 가로 단면)** — 실전에서 90% 이걸로 봅니다. 몸을 발쪽에서 올려다본 단면.
- **coronal(관상, 정면 슬라이스)** — 좌우·상하 한눈에. 흉수/섬유화 선형 병변에 유용.
- **sagittal(시상, 옆면 슬라이스)** — 앞뒤·상하. 척추/흉골 볼 때.
- **좌우 규칙**: 영상의 **왼쪽 = 환자의 오른쪽**(radiological convention). "right lung"은 화면 왼쪽.

**스크롤 순서** — axial을 **위(폐첨, apex) → 아래(횡격막/상복부)**로 천천히 훑습니다.

**밝기 = HU(Hounsfield) 눈금** (검정 ↔ 흰색)
- 공기 −1000 (새까맘, 정상 폐 내부) · 지방 −90 · 물 0 · 연부조직/근육 +40 · 급성 출혈 +60 · **뼈·금속·석회화 +400~+3000 (새하얌)**

**윈도우(Window) = 어느 HU 구간을 회색조로 펼칠지** (한 번에 다 못 봄)
- **lung**(L−600/W1500): 폐 안의 결절·음영·섬유화 — *폐 볼 땐 무조건 이거*
- **mediastinal**(L40/W400): 심장·종격동·림프절·흉수·연부조직 물질
- **bone**(L400/W1800): 뼈·금속·석회화

**읽는 루틴(순서를 정해두면 안 놓침)**
1. **기도**: 기관(trachea)·양 main bronchus 열려 있나 → 2. **폐야**(lung 창): 흰 점(결절)/뿌연 부분(음영·경화·간유리)/가는 선·그물(섬유화)/검은 구멍(폐기종) → 3. **종격동**(med 창): 심장 크기, 림프절 혹, 대동맥 → 4. **흉막**: 폐 바깥에 물 고임(흉수, 보통 등쪽 아래) → 5. **골격/물질**(bone 창): 척추·흉골, 금속(나사·판막·포트) → 6. **상복부 가장자리**: 간(지방간이면 어둡게) 등.

**용어 5개만 먼저**
- **nodule(결절)**: 작고 둥근 흰 점. 혈관 단면과 헷갈리면 스크롤 — 혈관은 관(선)으로 이어지고 결절은 점으로 고립.
- **consolidation(경화)**: 폐포가 액체/세포로 채워져 하얗게 됨(폐렴 등). **ground-glass(간유리)**: 그보다 옅은 뿌연 음영(혈관이 비쳐 보임).
- **effusion(삼출/흉수)**: 흉막강에 물 고임 → med 창에서 초승달 물음영, 중력방향(등쪽 아래).
- **fibrosis/sequela(섬유화 후유증)**: 흉터. 덩어리 아님 — 가는 선/그물, 폐가 쭈그러듦, 주로 폐 가장자리(subpleural).
- **calcification/metal(석회화·금속)**: 새하얀 점/선. 금속은 주변에 방사형 줄무늬 artifact.

---

## 6.1 all-zero (정상 기준) — 먼저 "정상 폐"를 눈에 익히기

### valid_1000_a  (n_labels=0; +labels: (none))
· 파일: `all-zero/valid_1000_a_1.nii.gz`

**Clinical:** Burning sensation in the body, weakness, fatigue, back pain.

**Technique:** Sections were taken without contrast medium and there were no reconstructions at the workstation.

**Findings:** Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions. No pathological wall thickness increase was observed in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No enlarged lymph nodes in pathological dimensions were detected. In the upper abdominal organs within the sections, there is no mass with distinguishable borders as far as it can be observed within the borders of non-enhanced CT. Thoracic vertebral corpus heights, alignments and densities are normal. Intervertebral disc distances are preserved. The neural foramina are open. No lytic-destructive lesions were detected in the bone structures within the sections.

**Impression:** Findings within normal limits.

**이상소견(한글):** 양성 라벨 **0개**, 리포트도 "정상 범위". 기도 개방, 폐 종괴·침윤 없음, 심장 크기 정상, 흉수·심낭 삼출 없음, 림프절 비대 없음, 뼈 정상 → **이상 없음.** 다른 케이스의 비교 기준.

**CT 볼 때:** lung 창으로 axial 스크롤 → 양쪽 폐야가 **고르게 새까맣고**(공기) 흰 점·뿌연 부분이 **없는** 상태를 눈에 각인. med 창으로 심장이 흉곽 가로폭의 절반 이하, 종격동에 혹 없음 확인. 이 "깨끗함"이 기준선.

### valid_1010_a  (n_labels=0; +labels: (none))
· 파일: `all-zero/valid_1010_a_1.nii.gz`

**Clinical:** covid?

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

**Impression:** Thorax CT examination within normal limits

**이상소견(한글):** 라벨 **0개**. COVID 의심으로 촬영했지만 폐 실질 정상, 결절·침윤·삼출 없음 → 정상. (감염이 의심돼 찍어도 정상일 수 있다는 예)

**CT 볼 때:** lung 창으로 **간유리(뿌연) 음영이 없는** 깨끗한 폐 확인 — 코로나 폐렴이면 여기 뿌연 부분이 생깁니다. 지금은 없으니 "정상 대조군"으로 익히기.

### valid_1012_a  (n_labels=0; +labels: (none))
· 파일: `all-zero/valid_1012_a_1.nii.gz`

**Clinical:** cough, sputum

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

**Impression:** Thoracic CT examination within normal limits

**이상소견(한글):** 라벨 **0개**. 기침·가래 증상이나 영상은 정상.

**CT 볼 때:** 위 두 개와 동일. **정상 3개를 연달아 보며** 검은 폐의 균질함을 기준으로 삼으면, 6.2부터 나오는 흰 점/뿌연 부분이 확 눈에 들어옵니다.

---

## 6.2 lung-nodule-only (폐결절만) — "작고 둥근 흰 점" 찾기 훈련

### valid_1001_a  (n_labels=1; +labels: Lung nodule)
· 파일: `lung-nodule-only/valid_1001_a_1.nii.gz`

**Clinical:** Cough, joint pain and chest pain

**Technique:** Sections were taken without contrast medium and reconstruction was performed at the workstation.

**Findings:** Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. There are millimetric nonspecific nodules in both lungs. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. No pleural or pericardial effusion was detected. The widths of the mediastinal main vascular structures are normal. No enlarged lymph nodes in pathological size and appearance were detected in the mediastinum and hilar regions. (…)

**Impression:** Millimetric nodules in both lungs

**이상소견(한글):** **폐결절** 1개 라벨. 양쪽 폐에 수 mm의 비특이 소결절(작고 둥근 흰 점). 나머지(심장·종격동·뼈)는 정상.

**CT 볼 때:** **lung 창 필수**(med 창에선 결절이 안 보임). axial을 천천히 스크롤하며 폐야 속 **작고 둥근 흰 점**을 찾기. 혈관 단면도 흰 점처럼 보이는데 — **위아래 슬라이스에서 관(선)으로 이어지면 혈관, 한두 장에서 점으로 고립되면 결절.**

### valid_1009_a  (n_labels=1; +labels: Lung nodule)
· 파일: `lung-nodule-only/valid_1009_a_1.nii.gz`

**Clinical:** Cough, fatigue.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** (…) There is a 10 mm hypodense oval-shaped finding in the lower quadrant of the left breast. lymph node? When examined in the lung parenchyma window; There are several millimetric non-specific nodules in both lungs. (…) In the upper abdominal organs included in the sections, the liver parenchyma changes in favor of steatosis. (…)

**Impression:** There is a 10 mm hypodense oval-shaped finding in the lower quadrant of the left breast. Lymph node?. There are several millimetric non-specific nodules in both lungs. Hepatosteatosis.

**이상소견(한글):** 라벨은 **폐결절**만. 부수 소견으로 좌측 유방 하부 10mm 저음영(림프절?)과 **간 지방증(hepatosteatosis)**. 라벨(18종)은 폐만 다뤄서 유방/간은 라벨엔 안 잡힘.

**CT 볼 때:** lung 창으로 소결절 점 찾기. **med/soft tissue 창**으로 화면 오른쪽(=환자 좌측) 유방 부위의 둥근 저음영 확인. **맨 아래 슬라이스**에서 간이 정상보다 **어둡게(지방간)** 보이는지 — 정상 간은 비장보다 약간 밝은데, 지방간은 더 어둡습니다.

### valid_1085_a  (n_labels=1; +labels: Lung nodule)
· 파일: `lung-nodule-only/valid_1085_a_1.nii.gz`

**Clinical:** Not given.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** No lymph node in pathological size and appearance was observed in the supraclavicular fossa, axilla and mediastinum. Heart dimensions and compartments appear natural. (…) No mass or nodular suspicious space-occupying lesion was detected in the lung parenchyma. A few nonspecific nodules with diameters less than 5 mm are observed. No pneumonic infiltration was detected in the parenchyma. (…)

**Impression:** Not given.

**이상소견(한글):** **폐결절**만. 5mm 미만 소결절 몇 개. 임상/결론은 "Not given"(제공 안 됨).

**CT 볼 때:** lung 창으로 5mm 미만의 작은 점 찾기. 앞의 두 결절 케이스와 **같은 패턴을 반복**해 눈에 익히면 됩니다 — 결절 그룹은 "폐 외 다른 이상은 없고 점만 있는" 게 공통.

---

## 6.3 diffuse-low-burden (폐 섬유화 후유증) — "점"이 아니라 "선/그물/흉터"

### valid_1022_a  (n_labels=1; +labels: Pulmonary fibrotic sequela)
· 파일: `diffuse-low-burden/valid_1022_a_1.nii.gz`

**Clinical:** Not given.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** (…) When examined in the lung parenchyma window; Linear fibroatelactastic changes were observed in the right lung middle lobe medial and left lung upper lobe inferior lingular segment. Apart from this, both lung parenchyma aeration is normal (…) Liver parenchymal density is diffusely decreased, consistent with hepatosteatosis. Two accessory spleens with diameters of 7 and 14 mm were observed (…)

**Impression:** Linear fibroatelactasis changes in right lung middle lobe medial and left lung upper lobe inferior lingular segment . Hepatosteatosis . Two accessory spleens in anterior upper pole of spleen. Scoliosis with thoracic opening facing left

**이상소견(한글):** **폐 섬유화 후유증.** 결절(덩어리)이 아니라 **선상 섬유무기폐** — 폐가 쭈그러들며 생긴 선/띠 모양 흉터. 위치: 우중엽 내측 + 좌상엽 하설상분절. 부수: 지방간, 부비장 2개, 척추측만(scoliosis).

**CT 볼 때:** **lung 창.** 결절(점)과 달리 **가느다란 선/그물(reticulation)·띠**와 그 주변 폐가 살짝 찌그러진(volume loss) 걸 찾기. 주로 **폐 가장자리(subpleural)**. **coronal**로 보면 선형 흉터가 더 잘 드러납니다.

### valid_1039_a  (n_labels=1; +labels: Pulmonary fibrotic sequela)
· 파일: `diffuse-low-burden/valid_1039_a_1.nii.gz`

**Clinical:** Not given.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** (…) When examined in the lung parenchyma window; In sections passing through the upper part of the left lung, subpleural sequelae fibrotic changes are observed in the major fissure, lower lobe laterobasal and right lung lower lobe laterobasal. (…)

**Impression:** Millimetric sequela fibrotic changes in bilateral lungs.

**이상소견(한글):** **폐 섬유화 후유증.** 양쪽 폐의 **아래·바깥쪽(subpleural, laterobasal)**에 mm 단위 섬유화 흉터.

**CT 볼 때:** lung 창, **폐 바깥 껍질쪽(subpleural)**의 가는 선/그물. axial **아래쪽 슬라이스(하엽)** 위주로 스크롤. 1022보다 더 미세하니 정상과 비교하며 "가장자리가 지저분한지" 보기.

### valid_1073_a  (n_labels=1; +labels: Pulmonary fibrotic sequela)
· 파일: `diffuse-low-burden/valid_1073_a_1.nii.gz`

**Clinical:** Not given.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of **3 mm**.

**Findings:** (…) When examined in the lung parenchyma window; trachea and both main bronchi are open. Sequelae changes are observed at the apical level of both lungs. At the laterobasal level, parenchymal thin bands are observed. There was no finding compatible with pneumonia in both lungs. (…)

**Impression:** No finding compatible with pneumonia was observed.

**이상소견(한글):** **폐 섬유화 후유증.** 양쪽 **폐첨(apical, 맨 위)**의 흉터성 변화 + laterobasal의 **얇은 띠(band)**. 폐렴 소견은 없음.

**CT 볼 때:** lung 창. **맨 위 슬라이스(폐첨)**의 흉터와 아래쪽의 얇은 띠. 이 스캔은 **3mm 두께**라(다른 건 1.5mm) 슬라이스가 성겨 약간 거칠게 보일 수 있음 — 정상적인 차이입니다.

---

## 6.4 multi-abnormality (복합 이상) — 창을 번갈아 쓰며 라벨과 대조

### valid_1016_b  (n_labels=8; +labels: Lymphadenopathy, Atelectasis, Lung nodule, Lung opacity, Pleural effusion, Peribronchial thickening, Consolidation, Interlobular septal thickening)
· 파일: `multi-abnormality/valid_1016_b_1.nii.gz`

**Clinical:** myelodysplastic syndrome

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** Minimal effusion was observed in both pleural spaces. Measured 20 mm on the right at its deepest point. In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated in favor of compressive atelectasis adjacent to the effusion. In the mediastinum, a lesion of soft tissue density is observed in the prevascular area, which is evaluated primarily in favor of lymphadenopathy, in which calcified foci in millimeter sizes are also observed. (…) There are nodules in both lungs (…) thickening in the peribronchovascular area and smooth interlobular septal thickness increases are observed (…)

**Impression:** Not given.

**이상소견(한글):** 라벨 **8개** — 흉수(양측, 우측 최대 20mm), 삼출에 눌린 **압박성 무기폐+경화**, 종격동 앞쪽(prevascular) **림프절병증**(안에 mm 석회화, 31mm), 양측 **폐결절**, **기관지 주위 비후**, **소엽간 중격 비후**. 골수형성이상증후군 환자의 복합 소견.

**CT 볼 때:** **창을 번갈아** 쓰세요. **med 창**: 폐 아래 초승달 물음영(**흉수**, 등쪽 아래 중력방향)·종격동 앞쪽 혹(**림프절**, 안에 흰 석회화 점). **lung 창**: 삼출 옆 눌려 뿌옇게 된 폐(**무기폐/경화**), **결절** 점, 기관지 벽 두꺼워짐, 소엽 사이 격자 선. **coronal**로 좌우 흉수 양 비교.

### valid_103_a  (n_labels=8; +labels: Medical material, Cardiomegaly, Lymphadenopathy, Emphysema, Lung opacity, Pleural effusion, Consolidation, Interlobular septal thickening)
· 파일: `multi-abnormality/valid_103_a_1.nii.gz`

**Clinical:** Not given.

**Technique:** 1.5 mm thick non-contrast sections were taken in the axial plane.

**Findings:** (…) Heart size increased. (…) Prosthetic material was observed in the aortic valve. There is post-op suture material on the wall of the ascending aorta. Lymph nodes measuring 19x11 mm (…) patchy areas of consolidation extending to the periphery and accompanying ground glass density increases were observed in the perihilar area of both lungs. (…) smooth interseptal thickenings (…) Free fluid (…) right (…) 24 mm (…) left (…) 5 mm. (…) Emphysematous changes (…) Metallic suture materials of sternotomy were observed in the sternum. (…)

**Impression:** Cardiomegaly. Patchy areas of consolidation in both lungs (…) infectious process (…). Bilateral diffuse uniform interlobular septal thickening (secondary to cardiac pathology?). Bilateral pleural effusion. Mild emphysematous changes in both lungs

**이상소견(한글):** 라벨 **8개** — **의료 물질**(대동맥판막 인공판막 + 흉골정중절개 금속 봉합사), **심비대**, **림프절병증**, **폐기종**, **폐 음영**, **흉수**(우 24/좌 5mm), **경화**, **소엽간 중격 비후**. 심장 수술을 받은 환자에 감염성 폐렴 의심.

**CT 볼 때:** **bone/med 창**으로 심장 부위 **밝은 금속**(인공판막)과 흉골의 **금속 봉합사**(아주 흰 점 + 별모양 streak artifact) 확인. **med 창**으로 심장 큼·양측 흉수. **lung 창**으로 **폐문 주위(perihilar)** 뿌연 간유리+경화(감염), 소엽간 격자 선, **폐기종**(정상 폐 안의 검은 구멍들).

### valid_1078_a  (n_labels=8; +labels: Arterial wall calcification, Cardiomegaly, Coronary artery wall calcification, Emphysema, Lung nodule, Lung opacity, Pleural effusion, Consolidation)
· 파일: `multi-abnormality/valid_1078_a_1.nii.gz`

**Clinical:** over ca.

**Technique:** Sections were taken without contrast medium and reconstruction was performed at the workstation.

**Findings:** (…) The heart is minimally larger than normal. (…) Atheroma plaques were observed in the aorta and coronary arteries. Pleural effusion is observed on the left. The pleural effusion measured 70 mm at its thickest point. (…) Consolidation and ground-glass appearances were observed in the posterior part of the lower lobe of the right lung, the lower lobe of the left lung, and the apicoposterior segment of the upper lobe (…) pneumonic infiltration. There are emphysematous changes in both aerated lungs. There are several millimeric nonspecific nodules in both lungs. (…)

**Impression:** Follow-up over ca. Left pleural effusion. Findings evaluated primarily in favor of pneumonic infiltration in both lungs. Emphysematous changes in both lungs. Millimetric nonspecific nodules in both lungs. Atherosclerotic changes in the aorta and coronary arteries.

**이상소견(한글):** 라벨 **8개** — **대동맥벽·관상동맥벽 석회화**, **심비대**, **폐기종**, **폐결절**, **폐 음영**, **좌측 흉수(70mm, 큼)**, **경화**. 폐렴 침윤 + 동맥경화가 함께 있는 환자.

**CT 볼 때:** **med 창**으로 심장 큼 + 대동맥·심장 관상동맥 벽을 따라 **흰 석회화 반점/선**. 화면 오른쪽(=환자 좌측) 폐 아래 **큰 물고임(70mm 흉수)**. **lung 창**으로 뿌연 경화·간유리(폐렴), **폐기종**(검은 공기주머니), 소결절 점. 1016_b·1078처럼 라벨 8개짜리는 "하나씩 창 바꿔가며 체크리스트로" 접근.

---

## 6.5 medical-material (인공물/의료 물질) — 새하얀 금속 + 그 주변

### valid_1288_a  (n_labels=1; +labels: Medical material)
· 파일: `medical-material/valid_1288_a_1.nii.gz`

**Clinical:** Coronavirus?

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** (…) Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected (…) Fixation material is observed in the thoracic vertebrae included in the study area. Metallic body artifact is observed on the left anterior chest wall.

**Impression:** Examination within normal limits

**이상소견(한글):** 라벨 = **의료 물질**뿐. **흉추 고정물**(척추 나사/막대) + 좌측 앞가슴벽 **금속 이물 artifact**. 폐 자체는 정상 → "물질만 있고 폐 병변은 없는" 순수 예.

**CT 볼 때:** **bone 창**으로 척추의 **새하얀 금속 고정물**(나사·막대) 확인. **med 창**으로 좌측 앞가슴벽 금속과 그 주위 **방사형 줄무늬(metal streak artifact)**. **lung 창**의 폐는 깨끗 — 물질과 폐 병변을 분리해 보는 연습.

### valid_1103_b  (n_labels=2; +labels: Medical material, Pulmonary fibrotic sequela)
· 파일: `medical-material/valid_1103_b_1.nii.gz`

**Clinical:** Operated carcinoid tumor, control.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** (…) The middle lobe of the right lung is not observed secondary to the operation, and its bronchus ends in a stump, and surgical suture materials are observed around the stump. In the right lung upper lobe posterior segment, there are suture materials and fibrotic recessions in the vicinity of the suture material, extending along the major fissure, causing structural distortion and minimal volume loss (…)

**Impression:** Surgical suture materials extending along the major fissure (…) right lung middle lobectomy; findings are also present in the previous CT examination. No newly developed pathology (…)

**이상소견(한글):** 라벨 **2개** — **의료 물질(수술 봉합사)** + **폐 섬유화 후유증.** 우중엽 절제술(lobectomy) 후 상태: 봉합사 + 그 주변 흉터성 수축·구조 왜곡. (카르시노이드 종양 수술 후 추적)

**CT 볼 때:** **med/bone 창**으로 우측 폐의 **봉합사(작은 밝은 점들)** 찾기. **lung 창**으로 그 주변 폐가 흉터로 당겨져(fibrotic recession) **구조가 뒤틀리고 부피가 준** 것 확인. 좌우 비교로 **우중엽이 "없는"(절제됨)** 것도 관찰.

### valid_114_b  (n_labels=2; +labels: Medical material, Lung nodule)
· 파일: `medical-material/valid_114_b_1.nii.gz`

**Clinical:** Not given.

**Technique:** Non-contrast images were taken in the axial plane with a section thickness of 1.5 mm.

**Findings:** On the right, the port chamber and the image of the catheter extending to the superior vena cava are seen on the anterior chest wall. (…) When examined in the lung parenchyma window; A few millimetric nonspecific parenchymal nodules were observed in both lungs. (…)

**Impression:** Millimetric stable parenchymal nodules in both lungs

**이상소견(한글):** 라벨 **2개** — **의료 물질**(우측 앞가슴벽 **케모포트** + 상대정맥으로 가는 **카테터**) + **폐결절.**

**CT 볼 때:** **med 창**으로 화면 왼쪽(=환자 우측) 앞가슴벽의 **둥근 금속 포트 챔버**와 거기서 나온 **카테터 선**이 상대정맥(SVC)으로 들어가는 경로를 슬라이스로 추적. **lung 창**으로 폐 소결절 점. **포트/카테터는 피부 밑 연부조직·혈관 안**, **결절은 폐야 안** — 위치로 구분하는 훈련.

---

> **정리**: 6.1 정상으로 기준을 잡고 → 6.2 결절(점) → 6.3 섬유화(선/그물) → 6.4 복합(창 번갈아) →
> 6.5 물질(금속)의 순서로 난이도가 올라갑니다. 각 케이스에서 **먼저 라벨을 가리고** 스스로
> 찾아본 뒤 한글 정리와 대조하면 훨씬 빨리 늡니다.
