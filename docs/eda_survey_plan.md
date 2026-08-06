# CT-RATE EDA — 표준 데이터셋 서베이 (proper redo)

## Context
기존 `figs/eda/` 4개 figure는 Phase A Day 4에 시간제한(Critic "deflated scope")으로 만든 얇은 조각이다.
메타데이터 44컬럼 중 spacing 2개, 리포트는 길이만 사용했고 구조·도메인·라벨노이즈·잠재공간은 미분석.
이번엔 **표준 데이터셋 서베이**로 제대로 다시 한다. 결정: **valid 먼저**, 심화 분석(잠재공간·대표성·템플릿화) **포함**.
각 단계는 실측 → 같이 확인 → 게이트에서 다음 결정. 한 번에 다 끝내지 않는다.

## 용어 (CT-RATE 공식 규약 — HF 데이터셋 카드)
파일명 = `split_patientID_scanID_reconstructionID` (예 `valid_1_a_1` = validation, **patient** `valid_1`, **scan** `valid_1_a`, **reconstruction** 1).
공식 계층 용어는 **patient · scan · reconstruction**뿐이며 study/series는 쓰지 않는다. 각 `.nii.gz`(=volume)는 한 reconstruction.

## 이미 확정된 실측 사실 (valid, 조사 단계에서 확인)
- 3,039 volume = **1,304 patient = 1,564 scan** (reconstruction 최대 6/scan, volume 최대 16/patient) — 공식 카운트와 일치
- **Findings의 50%가 완전 중복** (unique 1,505 / 3,039) — 한 scan의 reconstruction끼리 같은 리포트 공유
- reports/labels/metadata 조인 mismatch = 0
- 라벨 = predicted(silver, RadBERT 마이닝), 불균형(Lung nodule 45% … Pericardial effusion 7%), scan당 3.47개, 11.4% 소견 0개
- 리포트 강한 템플릿화(터키어→영어 병원 보일러플레이트)
- vendor 이름 불일치(`Siemens Healthineers`/`SIEMENS`), kernel stringified list(`['Br40f','3']`), HFS 자세, 나이 median 46

## 서베이 축 (단계)
| # | 단계 | 핵심 산출 | 게이트 질문 |
|---|------|-----------|-------------|
| S1 | 구조·출처·정합성 | patient/scan/reconstruction 계층, 리포트 중복, split·no_chest·error, 누수 | 이후 분석의 기본 단위(volume/scan/patient)는? |
| S2 | 이미지 acquisition 메타데이터 | vendor/kernel/dose/FOV/인구통계, split 분포 shift | FID/생성에 영향 큰 acquisition 변수는? |
| S3 | HU·복셀 강도 | 조직별 HU 피크, windowing, MAISI 입력 (볼륨 I/O 샘플) | 클립/정규화 가정 유효한가? |
| S4 | 텍스트·리포트 | 섹션 사용/공백, 길이, **템플릿화 정량화**, 어휘·negation | reportgen 메트릭·ctgen 조건 다양성 함의? |
| S5 | 라벨 & 라벨↔텍스트/이미지 일관성 | 불균형, co-occurrence 심화, silver-label 노이즈 probe | abnclass 지도 노이즈 바닥은? |
| S6 | MAISI 잠재공간 (심화) | 채널별 통계, 라벨별 분리도, scale 이슈 실측 | 잠재공간이 라벨을 분리하나? |
| S7 | valid_v2 대표성 + 종합 (심화) | valid_v2(1304) vs valid vs train 분포 비교 → 통합 리포트 | eval proxy가 대표적인가? |

말미에 **태스크 함의 부록**(ctgen/reportgen/abnclass/backbone) 별도 정리.

## 역할 분담
- **Claude**: 단계별 분석 코드 한 유닛씩 작성 → CPU 분석 실행 → figure/digest 생성 → 발견과 함의 설명 → 다음 단계 제안
- **User**: 각 게이트에서 우선순위·임상적 해석·유닛 승인·(있으면) 무거운 GPU 실행 인가·발견이 모델 계획에 미치는 최종 판단

## 산출물 위치
- 코드: `scripts/eda/stageN_*.py` (재사용 로직은 `src/data/ct_rate_eda.py` 기존 헬퍼 활용)
- figure/digest: `figs/eda_survey/` (기존 `figs/eda/` 4개는 보존)
- 통합 리포트: `docs/eda_report.md` (단계마다 증분 추가)

## 선행 사례 (조사)
CT-RATE/GenerateCT(multi-recon·silver-label), RadGenome-Chest CT(organ-grounded),
MIMIC-CXR/CheXpert EDA 관행(라벨 분포·불균형·stratified sampling·NLP 라벨 노이즈·findings/impression 섹션)을 위 축에 매핑.
