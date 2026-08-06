# CT-RATE 데이터셋 EDA 종합 보고서 (한국어)

VLM3D 2026 (ctgen / reportgen / abnclass) 파이프라인을 위한 CT-RATE 탐색적 데이터 분석 종합.
모든 수치는 `/workspace/tests/ctrate_eda/tables/` · `figures/` 산출물에서 직접 읽어온 값이며, 이전 EDA(GPT) 주장과 독립 재계산으로 교차검증되었다.

- 기본 인구(population)는 **CLEAN**: `no_chest`(reconstruction/volume 단위) + unencodable 제외.
- 분석 단위(scan / volume / patient)는 각 절에 명시. 리포트·18개 라벨은 **scan 단위 상수**(같은 scan의 여러 reconstruction이 공유).
- discrepancy 전체 표: [`discrepancy_report.md`](discrepancy_report.md).

---

## 1. Executive Summary

- **규모(검증)**: 매니페스트 총 **50,188 volume / 25,692 scan / 21,304 patient** (train 47,149 vol / 24,128 scan / 20,000 patient; valid 3,039 / 1,564 / 1,304). CSV 3종(report·metadata·label) 조인은 **orphan 0**로 완벽.
- **온디스크 CLEAN**: train **46,393 vol**, valid **3,001 vol** — `ctrate_full` census와 정확히 일치. no_chest가 volume-level에서 752/37 제거, unencodable 4/1 추가 제거.
- **핵심 구조**: 한 scan당 평균 ~2 reconstruction(90.5%가 2개). reconstruction은 **kernel/spacing/matrix만** 다르고 리포트·라벨은 동일 → 텍스트·라벨 통계는 반드시 scan으로 dedup해야 한다. reconstruction을 독립 샘플로 세면 텍스트/라벨이 ~2배 부풀려진다.
- **geometry 무결성**: `_fixed` NIfTI는 원본 geometry를 그대로 보존(metadata↔header spacing 최대차 4.67e-8 mm, shape 차 0), HU만 baked. 3,031개 header 스캔에서 read error·QC flag **0**.
- **이전 EDA 대비 2건의 Major 수치 정정**: no_chest scan/patient (260/258 → **280/278**, reconstruction 단위 오카운트) 와 Z-min (0.035 → **0.3 mm clean**, raw metadata 오류). 구조적 정정 1건: **raw-vs-fixed 비교 불가**(raw NIfTI 부재).
- **모델링 함의 요약**: (a) CT-CLIP text encoder는 BertTokenizer **max_length 512**라 Findings(median 226 tok)가 **512에 사실상 전부 fit(0.99%만 잘림)** → truncation은 비이슈; Findings vs Impression 선택은 truncation이 아니라 **boilerplate 43.9% 잡음**이 관건(77-token 심각 잘림은 CLIP-BPE 인코더 경로에만 해당); (b) scanner/vendor가 라벨·결측 패턴과 강하게 얽혀 shortcut 위험 → vendor-fingerprint metadata를 조건으로 쓰지 말 것; (c) MAISI 잠재공간은 스펙트럼이 거의 white(α<1)라 pixel-space의 coarse-to-fine 가정이 약함.

---

## 2. 검증된 데이터 규모

표: [`tables/dataset_counts.csv`](tables/dataset_counts.csv), [`tables/dataset_counts_raw.json`](tables/dataset_counts_raw.json). 그림: [`figures/metadata_overview.png`](figures/metadata_overview.png).

| 단위 | train | valid | 합계 |
|---|---|---|---|
| manifest volume | 47,149 | 3,039 | 50,188 |
| manifest scan | 24,128 | 1,564 | 25,692 |
| manifest patient | 20,000 | 1,304 | 21,304 |
| **온디스크 CLEAN volume** | **46,393** | **3,001** | 49,394 |
| no_chest volume 제거 | 752 | 37 | 789 |
| unencodable volume 제거 | 4 | 1 | 5 |

- reconstruction 분포(train): recon-id 2가 21,827 scan으로 지배적, 90.46% scan이 2개 recon 보유.
- 조인 무결성: reports/labels/metadata 각 47,149행(train)·3,039행(valid), 상호 orphan 0, scan 내부 report/label 불일치 0.
- unencodable 5종: `train_14384_a_2`, `valid_251_a_2` + z-spacing 결측 `train_1267_a_4`, `train_11755_a_3`, `train_11755_a_4`.

---

## 3. 이전 EDA(GPT) discrepancy 요약

전체는 [`discrepancy_report.md`](discrepancy_report.md). 핵심만:

- **Major (3)**: ① no_chest scan/patient **260/258 → 280/278** (no_chest는 reconstruction 단위이므로 scan/patient로 접으면 train 267 scan/265 pt + valid 13/13 = 280/278). ② Z-min **0.035 → 0.3 mm** (0.035 mm는 no_chest 처리된 patient train_9792 2개 volume의 metadata 오류, clean에서 배제). ③ recon-index → kernel 매핑이 **scanner마다 반대** — index로 lung/mediastinal read를 판별 불가.
- **Minor (adopt ours)**: 제조사 % (paper 61.5/30.1/8.4는 full-cohort volume; 우리 clean scan은 train 58.9/33.0/8.0), slice median 259→261, Technique 16→15 word, Findings 문장 15→16, calc joint count 5020→5120.
- **나머지 Verified (대부분 exact)**: 나이(median 46), 성별(정확 일치), 18라벨 prevalence(≤0.3pp), all-zero 11.32/11.57, calc phi 0.742, Findings word 185 등.
- **구조 정정**: raw NIfTI가 디스크에 없으므로 raw-vs-fixed 비교는 원천적으로 불가.

---

## 4. Cohort (인구통계) — scan 단위(나이·성별은 scan 상수)

표: [`tables/demographics.csv`](tables/demographics.csv).

- 나이: median **46세**, IQR 35–62(train)/35–61(valid), range 18–102(train, >100 3 scan)/18–96(valid).
- 성별(scan): train **M 14,098 / F 10,027 / 결측 3**; valid **M 910 / F 654 / 결측 0**. (GPT와 정확 일치)
- scan/patient: 평균 1.21(train). 20,000명 중 2,638명이 >1 scan(최대 17). 재촬영 간격 median 74일(mean 177일, 4,128 scan-pair) → 종단 추적 구조. split은 patient-disjoint로 구성되어 leakage 없음.

---

## 5. Acquisition · Scanner

표: [`tables/scanner_protocol.csv`](tables/scanner_protocol.csv), [`tables/spacing_shape.csv`](tables/spacing_shape.csv), [`tables/metadata_missingness.csv`](tables/metadata_missingness.csv), [`tables/kernel_table.csv`](tables/kernel_table.csv). 그림: [`figures/metadata_overview.png`](figures/metadata_overview.png).

- 제조사(scan, canonical): train Philips 58.9% / Siemens 33.0% / PNMS 8.0%; valid 61.0/31.1/8.0. (raw: SIEMENS와 Siemens Healthineers 분리 표기)
- spacing(volume): XY median 0.680/0.683 mm(sub-mm), Z median 1.25 mm(IQR 0.75–1.5). Z 이산값 {1.5: 18,606, 0.75: 11,852, 1.0: 8,431, 1.25: 5,371, 3.0: 1,234}. clean Z-min 0.3 mm.
- matrix(volume, train): 512² 30,438 / 1024² 13,985 / 768² 1,957 + 소수 비정방. slice median 261(train)/257(valid), 최대 2,062/1,617. >1000 slice: 11/3 volume.
- RescaleIntercept 이중분포 -1024 vs -8192(train 32,608 / 13,785). **-8192는 FOV 바깥 padding sentinel** — `_fixed`에 metadata rescale 재적용 금지(전부 air로 오염됨).
- **결측 지문(vendor fingerprint)**: RescaleType 65.96%, FocalSpots/GeneratorPower/DataCollection*/ReconstructionTarget* 65.65%, collimation 56.13% 등. **15개 컬럼이 vendor간 >50pp 결측 스윙** → 존재/부재만으로 제조사 식별 → 생성/VLM conditioning에 **부적합**.
- condition-safe metadata(결측 <1% + non-fingerprint): Manufacturer, ManufacturerModelName, PatientSex, PatientAge, ReconstructionDiameter, ConvolutionKernel, XY/ZSpacing, NumberofSlices, Rows/Columns, RescaleIntercept/Slope, CTDIvol, XRayTubeCurrent, StudyDate.

---

## 6. Reconstruction

표: [`tables/recon_compare.csv`](tables/recon_compare.csv). 그림: [`figures/recon_compare.png`](figures/recon_compare.png).

- **핵심**: reconstruction index는 kernel family에 일관 매핑되지 **않음**. Philips iCT 256: recon-1=YA(HRCT/sharp/lung), recon-2=B(soft). Philips Big Bore: recon-1=A(MEDIASTEN/soft), recon-2=L(PARANKIM/sharp) — **반대**. lung vs mediastinal read를 알려면 `ConvolutionKernel`/`SeriesDescription`을 읽어야 한다.
- sharpness 프록시(soft-tissue noise HU, Laplacian variance) 5쌍 전부 일치. sharp kernel은 soft 대비 ~2배 noise, ~7–15배 edge energy.
- `valid_1016_b`는 동일 geometry(512×512×209 @0.656/0.656/1.5) 자연실험: Bl57d(lung) vs Br36d(body), 복셀 mean|diff|=55.5 HU, max 1365 HU — 동일 아님. kernel 효과를 고정 geometry에서 분리.
- sharp≠고해상 geometry: Philips L은 1024² but 106–116 slice @3.0mm; soft B는 0.75mm 388–554 slice. slice/z-spacing은 kernel sharpness와 직교하는 프로토콜 선택.

---

## 7. Label (18 silver abnormality)

표: [`tables/label_prevalence.csv`](tables/label_prevalence.csv), [`tables/label_cooccurrence.csv`](tables/label_cooccurrence.csv), [`tables/label_by_scanner.csv`](tables/label_by_scanner.csv). 그림: [`figures/label_prevalence.png`](figures/label_prevalence.png), [`figures/label_cooccurrence.png`](figures/label_cooccurrence.png).

- prevalence(train scan): Lung nodule 45.48, Lung opacity 36.66, Arterial calc 27.77, Pulm fibrotic 26.72, Atelectasis 25.57, Lymphadenopathy 25.34, Coronary calc 24.75 … 최희소 Pericardial effusion 7.05, Mosaic 7.44, Interlobular 7.88. imbalance ratio **6.45**.
- all-zero(정상) 11.32%(train)/11.57%(valid); scan당 평균 양성 3.41/3.43개.
- **no_chest 제외는 scan-level prevalence를 바꾸지 않음**: 라벨은 scan 상수(24,128 scan에서 위반 0), 모든 no_chest volume은 chest recon도 있는 scan에 속함 → clean vs incl 표가 byte-identical. no_chest는 volume-level에서만 의미(752/37 drop).
- 동반발생: Arterial+Coronary calc phi **0.742**(joint 5,120, Jaccard 0.678, lift 3.09); Lung opacity+Consolidation phi 0.334. 계층군집이 두 calcification, 그리고 diffuse-parenchymal(opacity/consolidation/interlobular/mosaic)을 각각 묶음.
- **scanner shortcut**: all-zero율 SIEMENS 7.58% / SH 8.56% / Philips 12.09% / **PNMS 17.55%**; Lung nodule SIEMENS 52.0% vs Philips 43.1%. 제조사가 라벨 prevalence와 강상관 → classifier가 vendor를 shortcut으로 악용 위험.

---

## 8. Report (text)

표: [`tables/report_statistics.csv`](tables/report_statistics.csv), [`tables/report_boilerplate.csv`](tables/report_boilerplate.csv), [`tables/term_label_agreement.csv`](tables/term_label_agreement.csv). 그림: [`figures/report_length.png`](figures/report_length.png), [`figures/report_language_patterns.png`](figures/report_language_patterns.png). 샘플: [`samples/`](samples/).

- 필드별(scan): Clinical 결측 50.4/51.1%; Technique word median 15(거의 상수 boilerplate); Findings word median 185/183, sentence 16; Impression word median 28/29, 결측 3.24/2.81%. Impression/Findings 비율 0.142. exact-unique Findings 94.34/96.10%, Impression 82.36/87.04%.
- **tokenizer / max_len**: CT-CLIP의 실제 text encoder는 **BertTokenizer(CXR-BERT) `max_length=512, truncation=True`** (코드 확인 `ct_clip.py:600`, `CTCLIPTrainer.py:251`). Findings token median **226** → **512에서 0.99%만 잘림 = 사실상 전부 fit**(256 34.1%, 128 94.0% 잘림). Impression median 35 tok(전부 fit). Findings+Impression concat median 259 tok(512에서 4.1%만 잘림). ⚠️ **"77-token 99.3% 잘림"은 CT-CLIP이 아니라 CLIP-BPE 계열 인코더**(GenerateCT/FrozenCLIP3D)의 context length일 때만 해당 — CT-CLIP 경로에선 truncation 비이슈.
- **boilerplate 43.9%**: Findings token의 43.9%가 >1% 리포트에 공유되는 음성 체크리스트 문장(156 type). 최다: "When examined in the lung parenchyma window;" 14,029, "Pericardial effusion-thickening was not observed." 9,201.
- NegEx assertion(18 라벨 term): 155,875 affirmed / 64,121 negated / 27,173 uncertain — **음성 언급이 전체의 ~29%**. raw term-matching은 boilerplate 때문에 양성을 대폭 과대추정.
- term↔label: Pericardial-effusion raw 언급의 **91.9%가 라벨 음성**(→ NegEx로 precision 0.849 회복). Atelectasis affirmed precision 0.896/recall 0.913. 대부분 라벨은 NegEx 후 고정밀(Hiatal/Emphysema/Bronchiectasis 0.99).
- 라벨 부담↔길이: Spearman(#labels, Findings words)=0.669. band median(Findings/Impression): all-zero 130/6, 1–3 168/20, 4–7 220/41, 8+ 289/62.
- all-zero vs 양성 언어(Findings): 음성표현·정상표현은 양쪽 ~99%(구조화된 음성 체크리스트). 판별 신호는 uncertainty(22.2% vs 51.1%), comparison(2.1% vs 16.4%), recommendation(3.4% vs 26.2%).

---

## 9. NIfTI QC

표: [`tables/nifti_qc.csv`](tables/nifti_qc.csv), [`tables/nifti_qc_flags.csv`](tables/nifti_qc_flags.csv). 그림: [`figures/nifti_spacing_shape.png`](figures/nifti_spacing_shape.png).

- valid_fixed 3,001 + bundle 30 = **3,031 header 스캔, read error 0, QC flag 0**. 전부 int16 · ndim 3 · **LPS** · qform=sform=1.
- metadata↔header geometry 일치: spacing 최대차 4.67e-8 mm(>1e-2 건수 0), shape 차 0 → `_fixed`는 원본 geometry 보존(HU만 baked, resampling 없음). CSV가 디스크에 충실.
- no_chest 온디스크 count 0: valid_fixed는 이미 clean 3,001. affine det 전부 양수, 비정상 orientation/비3D/비양수 spacing 없음.

**Voxel-level QC** (모듈 06, 번들 30 + no_chest 20; 표 [`tables/nifti_voxel.csv`](tables/nifti_voxel.csv), [`tables/no_chest_coverage.csv`](tables/no_chest_coverage.csv)):
- HU 타당성 실증: `_fixed`에 metadata RescaleIntercept를 재적용하면 정상 median −875 HU → **−1899 HU, 83.5%가 −1000 이하(전부 공기)** 로 붕괴 → rescale 재적용 절대 금지 확인.
- **-8192 sentinel**은 FOV 바깥 padding(valid_1025_b 등)이며 실제 HU 아님 → clip[-1000,1000] 후 분석. air-peak는 전 볼륨 −1000 근처(정상).
- **no_chest 실체 = 대부분 뇌/두경부 CT** (montage로 확인; 예 `no_chest_valid_109_a_1.png`는 명백한 두개골+뇌). 자동 `coverage_proxy`(head-neck 8 / neck-to-chest 11 / abdomen 1)는 **크루드 라벨이라 montage가 authoritative** — FOV 대비 body 비율·bbox 내 공기로 삼분류. no_chest는 volume-level 제외가 정답임을 시각 재확인.

---

## 10. raw-vs-fixed 비교 (불가 명시)

- 디스크에는 `_fixed` NIfTI만 존재하고 **raw v1 NIfTI가 없다**. bundle도 fixed 30개뿐. 따라서 raw 대비 fixed의 HU 변환/재샘플 효과를 복셀 수준에서 직접 비교하는 것은 **원천적으로 불가능**하다.
- 대신 간접 확인: (a) metadata↔header geometry가 일치(§9)하므로 fixed는 **geometry를 바꾸지 않았고**, (b) HU만 baked in(RescaleIntercept -8192/-1024 재적용 금지). 즉 fixed 파이프라인은 "HU 정규화 + sentinel padding" 수준으로 이해하면 되고, 모든 분석은 **fixed-only**로 수행했다.

---

## 11. 멀티모달 CT–Report–Label

case sheet: [`case_sheets/INDEX.md`](case_sheets/INDEX.md), montage: [`figures/montages/`](figures/montages/), 불일치: [`case_sheets/mismatch_suspects.md`](case_sheets/mismatch_suspects.md).

- 25개 per-case sheet(5 그룹×5) = identity + metadata join + 양성 silver label + Findings/Impression + 6-panel montage(axial/coronal/sagittal × lung/mediastinal window, HU clip [-1000,1000]) + per-label NegEx + 사람 리뷰 체크리스트.
- report↔label 불일치 포인터: **FP-suspect 16 cell / 8 scan**(양성 라벨인데 term이 affirmed 아님 — 주로 calcification/lymphadenopathy처럼 영상유래·비서술 소견), **FN-suspect 3 cell / 2 scan**(affirmed term인데 라벨 음성 — 예: 비장 hilum 'nodular density'가 lung 라벨에 오매칭).
- 전부 포인터(verify X)이며 의학적 판정 없음. silver 라벨(scan)과 montage(volume) 단위 혼합은 각 sheet에 명시.

---

## 12. MAISI 잠재공간

표: [`tables/latent_stats.csv`](tables/latent_stats.csv). 그림: [`figures/latent_channels.png`](figures/latent_channels.png), [`figures/latent_umap_label.png`](figures/latent_umap_label.png), [`figures/latent_umap_vendor.png`](figures/latent_umap_vendor.png), [`figures/latent_decode_example.png`](figures/latent_decode_example.png).

- scale 확인: 정본 `_emb` latent 전체 std **0.957**(채널 0.928/0.993/0.939/0.961) — canonical ~0.98과 일치. `mu.pt`(std~0.67)는 valid_v2에 부재 → 직접 비교 불가, 스케일 불일치는 기지식으로만 명시(mu.pt를 cross-split에 쓰지 말 것).
- 채널: near-zero mean(-0.145/-0.088/0.005/-0.027), near-unit std, 대략 Gaussian, 꼬리 ±5.8~8.3. 공간 평균맵은 저주파 해부 구조 보유(white noise 아님).
- 구조: pooled-latent UMAP에서 abnormality silhouette **-0.262**(군집 없음), vendor -0.074(약한 confound). PCA top-5 var 0.247/0.096/0.077/0.053/0.043 — 소수 성분에 전역 구조(전체 강도/크기 추정) 집중. → pooled latent는 병리로 선형 조직화되지 않음.
- decode: canonical tiled SlidingWindowInferer(roi 80³, overlap 0.4)로 GPU decode 성공(full-volume 480³는 OOM). 복원 슬라이스 해부학적으로 일관.

---

## 13. 파워스펙트럼

표: [`tables/rapsd.csv`](tables/rapsd.csv). 그림: [`figures/rapsd.png`](figures/rapsd.png).

- CT 축상 픽셀 RAPSD log-log slope **α=2.71**(600 slice) — natural image ~2.0, Imagenette ~2.45보다 **가파름**. chest CT는 사진보다 더 저주파 지배(균질 soft-tissue/air 영역, 강한 coarse-to-fine prior).
- MAISI latent 채널은 거의 **spectrally white**: α ch0..3 = 0.94/0.54/0.56/0.42(모두 <1, pixel 2.71보다 훨씬 평탄). VAE가 공간 구조를 대부분 decorrelate. ch0만 tilt(coarse/구조), ch1-3은 flat·저파워(noise-like).
- 함의: pixel-space의 1/f^α(coarse-to-fine) 가정이 **latent에서는 약함** → per-frequency SNR이 균일, band-wise schedule/frequency-weighted loss의 레버리지 제한. 굳이 준다면 ch0에서만 의미.

---

## 14. valid_v2 대표성

- valid_v2 = valid_fixed **1304 one-scan-per-patient** clean set(= 헤드라인 FID/CLIP 및 학습 val 정본). image+text sidecar 1304/1304 완비.
- 대표성 근거: valid 전체(3,001 vol / 1,564 scan / 1,304 patient)와 **동일 patient 모수(1,304)**를 커버하되 patient당 1 scan으로 dedup → 인구통계·spacing·라벨 분포가 valid 모집단과 정합(나이 median 46, XY 0.683 mm, all-zero 11.57%). train과도 분포 일치(제조사 58.9 vs 61.0%, 라벨 prevalence ~동형).
- 주의: valid_v2는 patient당 1 recon만 보므로 **kernel/spacing 다양성은 축소**(scan 내부 recon variety 미포함). FID/CLIP 헤드라인은 이 축소를 감안. 실제 test set은 챌린지 주최측 보유.

---

## 15. Biases & Limitations

- **silver label**: 18라벨은 RadBERT 마이닝(≈1000건만 수동주석 후 자동확장). 특히 Pericardial effusion은 언급의 92%가 음성(boilerplate)라 노이즈가 큼 → abnclass 상한을 제약할 수 있음.
- **scanner/vendor shortcut**: 제조사가 라벨 prevalence·결측 패턴 양쪽과 얽힘. vendor-fingerprint metadata(15컬럼)를 조건/특징으로 쓰면 지름길 학습.
- **텍스트 boilerplate 43.9%**: conditioning 신호를 희석하고 token budget을 낭비.
- **단위 혼동 위험**: recon을 독립 샘플로 세면 텍스트/라벨 ~2배 부풀림. 항상 scan dedup.
- **bundle 표본 한계**: recon/멀티모달/스펙트럼 결론 일부는 30-volume bundle 기반(분포 주장 아님, 예시).
- **mu.pt 부재**: latent scale 불일치는 직접 재현 못함(기지식 인용).
- **FVD CT-Net**: 로컬 계산 불가(shuttle된 ctnet 체크포인트 손상 stub) — 헤드라인 지표는 서버측만.
- **raw-vs-fixed 불가**(§10).

---

## 16. VLM · Text-to-CT 모델링 권고 (구체 5+)

1. **Text conditioning**: CT-CLIP(BertTokenizer 512)에선 Findings가 거의 다 fit(512에서 0.99% 잘림)이므로 **Findings 그대로 또는 Findings+Impression concat(512, 4.1% 잘림) 사용 가능** — truncation 걱정 불필요. 단 **CLIP-BPE(77-token) 인코더를 쓰는 경로**(GenerateCT/FrozenCLIP3D)라면 Findings가 99% 잘리므로 Impression-only(median 35 tok) 필수. CLIPScore-T2I 헤드라인에 직결.
2. **Boilerplate de-emphasis**: 음성 체크리스트 문장(43.9% token)을 conditioning 전 제거/축약하거나 Impression 중심 신호 사용 → 조건 신호 대비 잡음비 개선. reportgen 레퍼런스와의 정합은 별개로 유지.
3. **vendor-fingerprint metadata를 조건에서 배제**. condition-safe 집합(§5: PatientSex/Age, spacing, ConvolutionKernel, CTDIvol 등)만 사용. voxel-spacing은 유지(생성 geometry 정의), 그러나 collimation/FocalSpots/RescaleType류 15컬럼은 금지.
4. **scanner-stratified 평가 + 잠재적 재가중**: PNMS(all-zero 17.55%, 저prevalence)와 Philips/Siemens 간 라벨 skew를 보고. abnclass는 vendor-balanced sampling 또는 domain 라벨을 adversarial하게 제거해 shortcut 억제.
5. **latent-space schedule은 flat-spectrum 전제로**: MAISI latent α<1이므로 band-wise noise schedule/frequency-weighted loss는 레버리지 작음. 표준 스케줄 유지, coarse-to-fine 튜닝이 필요하면 ch0에만. latent std ~0.98(`_emb`) 스케일로 정규화하고 mu.pt(0.67) 혼용 금지.
6. **train manifest·sampling(권장)**: scan 단위로 dedup 후 학습(volume 중복 제거) 하되, ctgen 생성 다양성이 필요하면 recon을 **augmentation**으로만 활용. no_chest는 volume-level에서 제거(scan은 유지). datalist 정본은 `datalist_v2.json`(brain 0, no_chest/unencodable 보정). valid는 valid_v2 1304 고정.
7. **abnclass 라벨 신뢰도 보정**: Pericardial effusion·Lymphadenopathy 등 negation-heavy 라벨은 NegEx/assertion 모델로 재검하거나 손실 가중을 낮춰 silver 노이즈 상한을 완화.

---

## 17. 사람이 직접 확인할 case 20

멀티모달 sheet([`case_sheets/`](case_sheets/))·불일치 리스트([`case_sheets/mismatch_suspects.md`](case_sheets/mismatch_suspects.md)) 기반 우선 리뷰 대상.

FP/FN 불일치 우선(라벨↔리포트 검증):
1. `valid_1068_a` — 비장 hilum 'nodular density'가 Lung nodule/opacity로 오매칭(FN-suspect)
2. `valid_103_a` — multi-abnormality 8라벨, 다수 FP-suspect cell
3. `valid_1078_a` — multi-abnormality, calcification 라벨 서술 부재 확인
4. `valid_1041_c` — multi-abnormality recon-c, 라벨↔소견 정합
5. `valid_1016_b` / 6. `valid_1016_d` — 동일 scan 다른 recon(kernel 효과) 라벨 일관성

Reconstruction/kernel 자연실험(§6):
7. `valid_1000_a` (YA vs B) 8. `valid_1001_a` (A vs L) 9. `valid_1022_a` (A vs L) 10. `valid_1288_a` (YA vs B) — index↔kernel 반전 육안 확인

medical-material(금속/디바이스 아티팩트):
11. `valid_1103_b` 12. `valid_114_b` 13. `valid_225_b` 14. `valid_366_a`

diffuse-low-burden(미묘 소견 민감도):
15. `valid_1022_a` 16. `valid_1039_a` 17. `valid_1073_a` 18. `valid_1153_a`

all-zero(정상 레퍼런스 품질):
19. `valid_1010_a` 20. `valid_1020_a`

---

## 18. 12 연구질문 답

### Q1. 독립 pair 수 50,188 vs 25,692 — 어느 것이 맞나?
**A.** 둘 다 맞되 단위가 다르다. **50,188은 volume(reconstruction)** 총수, **25,692는 scan** 총수(21,304 patient). 리포트·라벨은 scan 상수이므로 **텍스트/라벨의 독립 관측은 25,692 scan**이 맞고, 영상 픽셀/geometry 관측은 50,188 volume이 맞다. (`tables/dataset_counts.csv`)

### Q2. reconstruction 중복의 영향은?
**A.** 한 scan당 평균 ~2 recon(90.5%가 2개). recon은 kernel/spacing/matrix만 다르고 리포트·라벨 동일 → recon을 독립 샘플로 세면 텍스트/라벨 통계가 ~2배 부풀고 model이 vendor/kernel 프로토콜을 과대학습한다. 반드시 scan dedup(첫 recon 유지; 라벨 상수라 pick 무관, joint count만 미세 변동). 생성 다양성엔 augmentation으로만.

### Q3. all-zero(정상)를 '정상'으로 사용해도 되나?
**A.** 조건부 가능. all-zero는 train 11.32% / valid 11.57%(scan). 단 라벨은 **silver**라 일부 all-zero가 실제 미세소견을 놓쳤을 수 있고(§8 boilerplate·NegEx 한계), all-zero 리포트도 uncertainty 22%·recommendation 3%를 포함. 정상 레퍼런스로 쓰되 silver 한계를 감안하고, 필요시 case sheet(§17 19–20)로 spot-check.

### Q4. Findings vs Impression — 어느 것을 conditioning에?
**A.** CT-CLIP(512)에선 **둘 다 fit**하므로 truncation이 아니라 **정보/잡음**으로 선택. **fidelity가 필요하면 Findings**(해부 상세, median 226 tok, 512에 0.99%만 잘림), **간결·저비용이면 Impression**(median 35 tok). Findings는 boilerplate 43.9%가 신호를 희석하니 권장: Impression 전체 + Findings 핵심문장, boilerplate 제거 후 결합(§16-1,2). (CLIP-BPE 77-token 경로라면 Findings는 못 쓰고 Impression-only.)

### Q5. 음성 문장 제거는?
**A.** conditioning 관점에서 **권장**. Findings token의 43.9%가 음성 체크리스트 boilerplate이고 term 언급의 29%가 negated. 제거하면 조건 신호가 선명해지고 token budget 확보. 단 **reportgen 학습/평가에서는 유지**(레퍼런스 분포가 음성 서술 포함). abnclass 라벨링은 NegEx로 negated를 걸러야 정확도 상한이 오른다.

### Q6. Technique 필드의 가치는?
**A.** 낮음. Technique word median 15로 거의 상수 boilerplate(exact-unique train 1.48%). 프로토콜 텍스트라 conditioning 신호 미미. 대신 구조화 metadata(spacing/kernel/CTDIvol)를 쓰는 게 낫다. 결측은 거의 없음(0.004%).

### Q7. tokenizer truncation의 최대 손실은?
**A.** 손실은 **어느 인코더를 쓰느냐**에 달림. CT-CLIP의 실제 인코더는 **BertTokenizer 512**라 Findings(median 226 tok)가 **512에서 0.99%만 잘림 = 최소**. 손실이 커지는 건 짧은 context: 256에서 34.1%, 128에서 94.0%, **77(CLIP-BPE 인코더의 값)에서 99.33%**. 즉 최대 손실 지점은 **CLIP-BPE(77) 경로에서의 Findings**이며, CT-CLIP(512) 경로에선 Findings·concat 모두 실질 손실 없음. (`tables/report_raw.json`)

### Q8. scanner / kernel / spacing shortcut 위험은?
**A.** 실재. (a) 제조사가 라벨 prevalence와 강상관(all-zero PNMS 17.55% vs SIEMENS 7.58%; Lung nodule 52 vs 43%). (b) 15개 metadata 컬럼이 vendor간 >50pp 결측 스윙 → 존재만으로 vendor 식별. (c) kernel은 noise/edge를 2–15배 바꿈. → vendor-fingerprint 조건 배제 + scanner-stratified 평가 + 재가중(§16-3,4).

### Q9. raw vs fixed — 무엇을 쓰나?
**A.** **fixed only**. 디스크에 raw NIfTI가 없어 raw-vs-fixed 복셀 비교는 불가(§10). fixed는 geometry를 보존(metadata↔header 차 ~0)하고 HU만 baked. metadata RescaleSlope/Intercept를 **재적용 금지**(-8192는 FOV padding sentinel, clip [-1000,1000]).

### Q10. no_chest 제거는 어느 시점에?
**A.** **volume(reconstruction)-level**에서. no_chest는 per-reconstruction 플래그(train 752 / valid 37 volume; scan 280 / patient 278). scan-level 라벨/텍스트 prevalence는 no_chest 제거로 **불변**(모든 no_chest volume이 chest recon도 있는 scan에 속함). 따라서 라벨/텍스트 통계는 scan dedup만으로 충분하고, **영상 학습 datalist에서만 volume-level로 no_chest를 배제**하면 된다.

### Q11. CT-CLIP score의 순환 편향은?
**A.** 존재한다. CLIPScore-T2I는 CT-CLIP(우리 backbone 후보)로 계산되고, 생성 모델도 같은 CT-CLIP 텍스트 임베딩으로 conditioning하면 **평가자와 생성자가 같은 표현공간을 공유** → 실제 화질보다 alignment를 과대평가할 수 있다. 완화: (a) 헤드라인 우선순위를 2.5D-FID > CLIPScore > FVD로(FID는 CT-CLIP 비의존), (b) conditioning encoder와 평가 encoder를 분리, (c) I2I/T2I 대각 검증(gt-gt=100)으로 sanity, (d) 사람 case 리뷰 병행.

### Q12. 권장 train manifest · sampling은?
**A.** **manifest**: scan-level dedup 정본 `datalist_v2.json`(brain 0, no_chest+unencodable 보정; 5k) / `datalist_full_v2.json`(47k). no_chest는 volume-level 제거, valid는 **valid_v2 1304 고정**. **sampling**: (1) scan을 기본 단위로, recon은 optional augmentation. (2) label-rare(Pericardial/Mosaic/Interlobular ~7%)·all-zero(11%) balance를 위해 class-aware 또는 vendor-balanced sampling. (3) latent은 `_emb`(std~0.98) 스케일 통일, mu.pt 혼용 금지. (4) 3D latent I/O가 병목이므로 CPU+worker 기본, 내부 연산이 정당화될 때만 GPU.

---

## 19. 결론

- CT-RATE의 규모·구조·geometry는 독립 재계산으로 **검증**되었다(50,188 vol / 25,692 scan / 21,304 patient; clean 46,393 / 3,001). 이전 EDA는 대체로 정확했고, **2건의 Major 수치 정정**(no_chest 280/278, Z-min 0.3mm)과 **1건의 구조 정정**(raw-vs-fixed 불가), **1건의 방법론 정정**(recon-index≠kernel)이 핵심 차이다.
- VLM3D 모델링의 3대 실무 결론: (1) **텍스트 조건은 CT-CLIP 512에 Findings가 사실상 전부 fit(0.99%만 잘림)** — truncation 비이슈, 선택 기준은 boilerplate 43.9% 잡음(77-token 심각 잘림은 CLIP-BPE 인코더 경로 한정), (2) **scanner/vendor shortcut을 설계에서 배제** — fingerprint metadata 금지 + stratified 평가, (3) **MAISI latent은 near-white 스펙트럼** — pixel-space coarse-to-fine 가정과 CT-CLIP 순환편향을 경계.
- 산출물은 재현 가능(`run_all_eda.py`)하며, 사람 검증용 20 case와 포인터형 불일치 리스트를 함께 제공한다.
