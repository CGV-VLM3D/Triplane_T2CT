# VLM3D 2026 Project Guidebook

> Last updated: 2026-06-09 (Phase B 진행 중). 코드/디렉토리 변경 시 이 문서도 같이 갱신해 주세요.

이 문서는 **(1) 프로젝트 코드 구조 투어**, **(2) 읽어야 할 파일 우선순위**, **(3) 데이터셋 통계 요약 + 그림**, **(4) GenerateCT 출력 보는 법**, **(5) 자주 쓰는 명령**을 한 곳에 모은 가이드북입니다. baseline 코드를 어떤 순서로 읽을지는 별도 문서 [docs/baseline_reading_order.md](docs/baseline_reading_order.md)에 정리돼 있습니다.

---

## 0. TL;DR — 무엇을 하는 프로젝트인가

**목표**: MICCAI VLM3D 2026 Task 4 (Text-Conditional CT Generation) 챌린지에 제출하고, 작년 1위 **Report2CT**를 metric에서 이긴다.

**Pipeline (목표 형태)**

```
radiology report (findings + impression) + voxel spacing
  → 3 text encoders (HF) → 2 × 2560 cond tensor
  → DiffusionModelUNetMaisi (cross-attn at last 2 levels, dim 2560)
  → MAISI VAE decoder (frozen)
  → CT volume [1, 480, 480, 256]
```

**제출 마감**: 2026-08-20. Phase A 완료. **Phase B 진행 중** — Report2CT 학습 완료(`data/report2ct_work_dir/checkpoints/epoch_079*.ckpt`), 베이스라인 어댑터(CT-CLIP / fVLM / GenerateCT / Text2CT) 추가, VLM3D eval 하네스 구축.

**Win condition**: VLM3D-Dockers로 측정한 CT-RATE valid 1000-split에서 `ours_final`이 `report2ct_our_repro` 대비 `{FID_2p5D_Avg, CLIPScore, FVD_CTNet}` 중 2개 이상에서 이긴다. 우선 metric: **2.5D-FID > CLIPScore-T2I > FVD**.

전체 의사결정 흐름은 `.omc/specs/`의 deep-interview 기록과 `.omc/plans/vlm3d-pivot-plan.md`(3-iteration consensus)에 정돈돼 있습니다.

---

## 1. 처음 30분 권장 코스 (읽기만)

| # | 파일 | 왜 | 분량 |
|---|---|---|---|
| 1 | `CLAUDE.md` | 프로젝트 한 페이지 요약 (목표 / 환경 / repo layout / win condition / non-goals) | 5분 |
| 2 | `GUIDE.md` (이 파일) | 코드와 데이터셋을 어디서부터 봐야 할지 | 10분 |
| 3 | [docs/baseline_reading_order.md](docs/baseline_reading_order.md) | baseline(MAISI/CT-CLIP/Report2CT/Text2CT) 코드 읽는 순서 | 5분 |
| 4 | `figs/eda/*.png` | 데이터셋이 어떻게 생겼는지 (4 figures + summary in §3) | 5분 |

---

## 2. Repo 구조 투어

```
.
├── CLAUDE.md                       # 프로젝트 한 페이지 요약 (claude/사람 모두 읽음)
├── GUIDE.md                        # 이 가이드
├── Makefile, pyproject.toml, environment.yaml, requirements.txt
│
├── src/                            # 우리 코드 (lightning-hydra-template base + 우리 추가)
│   ├── train.py, eval.py, inference.py   # Hydra @main entrypoints
│   ├── vlm3d_runner.py             # ★ VLM3D-Dockers ctgen 평가 subprocess wrapper
│   ├── data/                       # ct_rate_datamodule, ct_rate_eda, report2ct_datamodule, fvlm_report
│   ├── models/                     # report2ct_module.py (LIVE LightningModule) + components/
│   ├── baselines/                  # MAISI loader + 베이스라인 어댑터 (상세: §4 / baseline_reading_order.md)
│   │   ├── maisi.py                # ★ MAISI VAE 동결 loader (bundle config 그대로)
│   │   ├── ctclip_adapter.py, fvlm_adapter.py, fvlm_preprocess.py   # VLM 백본 + fVLM 입력 전처리
│   │   ├── generatect_adapter.py   # ★ GenerateCT wrapper
│   │   ├── report2ct_{image,text}_encoder.py
│   │   └── text2ct_adapter.py, rflow.py, _vendored/   # Text2CT + RFlow 스케줄러
│   ├── eval/                       # samplers/ + tasks/ + ct_rate_cases.py (VLM3D ctgen 평가)
│   ├── diagnostics/                # (placeholder — __init__.py only)
│   ├── callbacks/                  # discord_notifier.py
│   └── utils/                      # gpu.py + template 보일러플레이트 (logging/rich/instantiators)
│
├── configs/                        # Hydra 계층
│   ├── train.yaml, eval.yaml, inference.yaml
│   ├── data/, trainer/, logger/, callbacks/, paths/, ...
│   ├── model/                      # report2ct.yaml, generatect.yaml, text2ct.yaml, vlm_backbone/{ctclip,fvlm}.yaml
│   ├── eval/                       # default.yaml + model/{report2ct,text2ct,generatect}.yaml + task/ctgen.yaml
│   └── experiment/report2ct_repro.yaml
│
├── tests/                          # pytest 53 passed / 4 skipped (가중치·데이터 게이트)
│   ├── test_maisi_frozen_load.py   # ★ MAISI 동결 검증
│   ├── test_{ctclip,fvlm}_adapter.py, test_fvlm_report.py   # VLM 백본 + fVLM 리포트 분해
│   ├── test_report2ct_module.py, test_report2ct_parity.py   # UNet forward + upstream parity
│   ├── test_{generatect,text2ct}_adapter.py + *_spacing 테스트
│   └── test_hydra_compose.py, test_data_module.py, test_lightning_fit_smoke.py
│
├── third_party/                    # READ-ONLY 외부 코드 (Principle P2; 핀은 docs/submodule_pins.md)
│   ├── report2ct/, generatect/, vlm3d_dockers/, ct_clip/, text2ct/   # submodules
│   ├── fvlm/                          # 일반 디렉토리로 전환됨 (더 이상 submodule 아님)
│   └── maisi_bundle/                  # MONAI MAISI bundle (vendored, FROZEN VAE)
│
├── docs/                           # baseline_reading_order, *_runbook (vlm/text2ct/report2ct),
│                                   #   submodule_pins, fvlm_report2ct_guidebook, report2ct_external_components
│
├── .omc/                           # 의사결정 + 계획 artifacts
│   ├── specs/                      # deep-interview 기록
│   └── plans/                      # vlm3d-pivot-plan.md (consensus) + report2ct_impl_spec.md 등
│
├── scripts/                        # run_eda, run_eval, precompute_report2ct_{text,image}_embeddings,
│                                   #   build_report2ct_datalist, decompose/calibrate_*_fvlm, generate_text2ct_valid
│
├── notebooks/                      # eda.ipynb, 3D_viewer.ipynb, test_generateCT.ipynb
│
├── figs/eda/                       # EDA 산출물 (4 PNGs — §3)
│
├── results/                        # upper_bound.json (MAISI round-trip PSNR 30.94) + report2ct_envelope.json
│
├── data/                           # 새 artifacts (read-write, gitignored)
│   ├── checkpoints/                # generatect, ctclip, fvlm, text2ct, hf_cache
│   └── report2ct_work_dir/         # Report2CT 학습 임베딩 + checkpoints (epoch_079 등)
│
├── datasets/                       # ☝️ READ-ONLY collaborator 데이터 (절대 쓰지 말 것)
│   └── datasets/CT-RATE/dataset/{train_fixed,valid_fixed,metadata,radiology_text_reports,ts_seg,...}
│
├── paper_pdf/                      # 참고 논문 (Report2CT, GenerateCT, MAISI, ...)
│
└── deprecated/                     # 모든 triplane-era 작업 (import 금지)
```

★ 표시 = 우리가 직접 짠 핵심 파일.

---

## 3. 데이터셋 통계 (CT-RATE valid)

`figs/eda/`에 4개 figure로 시각화돼 있어요. 핵심 발견을 인용해 둡니다 (`scripts/run_eda.py` 출력).

### 3.1 라벨 co-occurrence (`figs/eda/label_cooccurrence.png`)

![label co-occurrence](figs/eda/label_cooccurrence.png)

- 18개 abnormality 라벨 사이의 조건부확률 P(row | column) heatmap.
- 대각선이 1 (자기 자신).
- 활용 신호: Lung opacity ↔ Consolidation, Cardiomegaly ↔ Pleural effusion 처럼 임상적으로 co-occur하는 짝들이 보임. counterfactual diagnostic에서 단일-라벨 perturbation 효과를 검증할 때 이 분포를 참고.

### 3.2 Report 단어 수 분포 (`figs/eda/report_token_len.png`)

![report token len](figs/eda/report_token_len.png)

| Section | 중위수 | p95 | 최대 |
|---|---|---|---|
| Findings | 184 단어 | 329 | 626 |
| Impressions | 28 단어 | 88 | 210 |

**의미**: Findings가 Impressions보다 약 6.5배 길다. Report2CT가 두 섹션을 **별도로** encode하고 cross-attn에 concat하는 설계가 정당화됨 — 길이 차이가 커서 단순 concat 후 single-pool은 짧은 Impression 정보가 묻힐 위험.

### 3.3 Spacing 분포 (`figs/eda/spacing_violin.png`)

![spacing](figs/eda/spacing_violin.png)

| 축 | 중위수 | p5–p95 |
|---|---|---|
| XY | 0.68 mm | [0.34, 0.83] |
| Z | 1.25 mm | [0.75, 1.50] |

**의미**: anisotropic (XY ≠ Z). Report2CT가 voxel spacing을 conditioning input으로 받는 이유. 우리 모델도 spacing-conditioning을 keep해야 함.

### 3.4 HU intensity (`figs/eda/hu_histogram.png`)

![hu](figs/eda/hu_histogram.png)

50개 volume에서 ~50,000 voxel씩 샘플링한 분포.

| 통계 | 값 |
|---|---|
| 중위수 voxel HU | -825 |
| p5–p95 | [-1024, 146] |

**의미**: lung CT답게 air-dominated. Report2CT/우리 모델 모두 HU clip `[-1000, 1000]` → `[0, 1]` 정규화. Dynamic range 대부분이 air-tissue interface에 집중돼 있어 FID 계산 시 정규화 일관성이 중요.

### 3.5 데이터 위치 (절대로 쓰지 말 것 — read-only)

| Path | 용도 |
|---|---|
| `/workspace/datasets/datasets/CT-RATE/dataset/valid_fixed/<patient>/<study>/<volume>.nii.gz` | NIfTI 볼륨 (1,304 patients, 3,038 valid scans) |
| `…/train_fixed/…` | 학습용 (20,000 patients, 47,148 scans) |
| `…/radiology_text_reports/validation_reports.csv` | VolumeName + Findings_EN + Impressions_EN |
| `…/metadata/validation_metadata.csv` | spacing, manufacturer, kernel, ... |
| `…/multi_abnormality_labels/valid_predicted_labels.csv` | 18 binary labels per scan |
| `…/ts_seg/ts_total/…` | TotalSegmentator 장기 마스크 (fVLM 입력용) |

새 캐시/derivative artifact는 모두 `/workspace/data/` 아래로.

---

## 4. 코드 모듈별 해설

> 베이스라인(CT-CLIP / Report2CT / Text2CT / MAISI)을 어떤 순서로 읽을지는
> [docs/baseline_reading_order.md](docs/baseline_reading_order.md)에 정리. 아래는 핵심 모듈 요약.

### 4.1 `src/baselines/maisi.py` — MAISI VAE frozen loader

**한 줄 요약**: bundle config 그대로 읽어서 frozen autoencoder 반환. 모든 곳에서 이걸로 MAISI 로드.

```python
from src.baselines.maisi import load_frozen
vae = load_frozen(device="cuda:0")  # 모든 param requires_grad=False, .eval() 상태
```

- `monai.bundle.ConfigParser`로 `third_party/maisi_bundle/configs/inference.json`의 `autoencoder_def`를 그대로 instantiate. **architecture kwargs를 한 줄도 안 적음** (중복 0).
- `is_fully_frozen(model)` helper로 동결 검증. `tests/test_maisi_frozen_load.py`가 이를 검증.

### 4.2 `src/models/report2ct_module.py` — Report2CT (LIVE)

**한 줄 요약**: Report2CT의 **실 학습/추론 LightningModule**. config `configs/model/report2ct.yaml`의 `_target_`가 가리키는 진짜 경로. UNet은 `DiffusionModelUNetMaisi`(233M), 스케줄러는 `src/baselines/rflow.RFlowScheduler`, 텍스트 인코딩은 `report2ct_{text,image}_encoder.py`. UNet/scheduler 정의는 코드로 새로 안 짜고 YAML `_target_`로 MONAI 클래스를 instantiate하며, `config_maisi_2560.json`과의 1:1 패리티 + bit-exact forward는 [tests/test_report2ct_parity.py](tests/test_report2ct_parity.py)가 강제한다. *(옛 `report2ct_adapter.py` skeleton은 2026-06-09 삭제 — parity 테스트와 완전 중복이었음.)*

학습은 [[report2ct-training-is-user-owned]] 정책에 따라 사용자가 직접 수행(완료, `epoch_079` ckpt 존재). 절차는 [docs/report2ct_training_runbook.md](docs/report2ct_training_runbook.md).

### 4.3 `src/baselines/generatect_adapter.py` — GenerateCT 텍스트→볼륨

**한 줄 요약**: `sys.path` 추가로 transformer_maskgit 가져오고 CTViT + MaskGITTransformer를 paper kwargs로 빌드. 3 pretrained ckpt 로드.

```python
from src.baselines.generatect_adapter import GenerateCTAdapter
adapter = GenerateCTAdapter(device_str="cuda:0")
volume = adapter.text_to_volume("Findings consistent with viral pneumonia in both lungs.")
```

- `sys.path.insert`로 setup.py 실행 없이 import. **DUPLICATION INTENTIONAL** annotation이 `_CTVIT_KWARGS`/`_MASKGIT_KWARGS` 위에 있음 (submodule이 JSON/YAML 없이 Python literal로만 설정 보관).

### 4.4 `src/baselines/text2ct_adapter.py` — Text2CT (MAISI 잠재 rectified-flow)

**한 줄 요약**: Report2CT와 같은 MAISI 잠재 family의 또 다른 생성기. FrozenCLIP3D 텍스트 조건 + RFlow 샘플러. inference-only (upstream `scripts.diff_model_demo.run_inference` 재사용). 절차는 [docs/text2ct_runbook.md](docs/text2ct_runbook.md).
- 스케줄러: `src/baselines/_vendored/rectified_flow.py`(MONAI ≥1.5 전용이라 vendor). `src/baselines/rflow.py`는 MAISI 번들 스케줄러 re-export shim — 혼동 주의.

### 4.5 `src/data/ct_rate_datamodule.py` — CT-RATE LightningDataModule

**한 줄 요약**: 3개 CSV (reports + metadata + labels)를 VolumeName 키로 join해서 `CTRateRecord` dataclass 리스트로 반환.

```python
from src.data.ct_rate_datamodule import CTRateDataModule, load_records
records = load_records("valid")  # 메타데이터만 (NIfTI 안 읽음)
```

- `_parse_spacing()`가 stringified-list spacing 셀 처리. `mode="metadata"`(현재)는 NIfTI 안 읽음; `mode="volume"`은 Phase C(`NotImplementedError`).

### 4.6 `src/vlm3d_runner.py` + `src/eval/` — VLM3D 평가

**한 줄 요약**: `vlm3d_runner.py`는 ctgen_evaluation docker를 subprocess로 invoke(없으면 NaN placeholder로 schema validate). `src/eval/`는 우리 측 샘플러(`samplers/{report2ct,text2ct,generatect}.py`) + 메트릭 태스크(`tasks/ctgen.py`, 2.5D-FID/CLIPScore/FVD)로 baseline 결과를 생성·평가.

```bash
python -m src.vlm3d_runner --dry-run --out /tmp/smoke.json   # docker 없을 때
```

### 4.7 `tests/` — pytest 가이드

전체: **53 passed / 4 skipped** (skip은 실가중치·parity-reference 데이터 게이트). 핵심:

| 파일 | 검증 |
|---|---|
| `test_maisi_frozen_load.py` | MAISI ckpt 로드 + 모든 param `requires_grad=False` |
| `test_report2ct_module.py` / `test_report2ct_parity.py` | UNet 1-step forward + upstream parity |
| `test_{ctclip,fvlm}_adapter.py`, `test_fvlm_report.py` | VLM 백본 빌드/forward + fVLM 리포트 분해 |
| `test_{generatect,text2ct}_adapter.py` (+ spacing) | 어댑터 빌드/추론 + spacing 규약 |
| `test_hydra_compose.py`, `test_data_module.py`, `test_lightning_fit_smoke.py` | config compose / DM smoke / Trainer.fit 1-step |

```bash
pytest tests/ -q     # 전체
```

---

## 5. GenerateCT 출력 보기

**현재 상태**: 3개 pretrained ckpt(~1.5GB)가 `data/checkpoints/generatect/`에 있고 adapter도 작동. **단 MaskGITTransformer가 `.cuda()`를 강제**하므로 dev container 안 inference는 GPU 필요.

### 옵션 A — GPU 한 장으로 1샘플 inference (~5분, 저해상도 128³)

```python
import torch, nibabel as nib, numpy as np
from src.baselines.generatect_adapter import GenerateCTAdapter

adapter = GenerateCTAdapter(device_str="cuda:0", load_super_resolution=False)
volume = adapter.text_to_volume("Findings consistent with viral pneumonia in both lungs.")
vol = volume.squeeze().cpu().numpy()
nib.save(nib.Nifti1Image(vol, np.eye(4)), "/workspace/data/generatect_smoke.nii.gz")
# ITK-SNAP / 3D Slicer / nibabel로 보기
```

### 옵션 B — VLM3D-Dockers example_docker 그대로 (full pipeline 512×512)

```bash
cd /workspace/third_party/vlm3d_dockers/ctgen_example_docker
bash test.sh   # 빌드 + 5-prompt 샘플 inference → exported_images/
```

docker daemon 필요 + full super-res까지 도는 무거운 경로.

---

## 6. 자주 쓰는 명령어 모음

```bash
# 환경 확인
python -c "import lightning, hydra, monai, diffusers, transformers; print('OK')"

# Hydra config compose 검증
python src/train.py --cfg job --resolve experiment=report2ct_repro
python src/eval.py  --cfg job --resolve ckpt_path=/tmp/dummy.ckpt

# VLM 백본 compose (ctclip / fvlm)
python -c "from hydra import initialize, compose; \
  initialize(version_base='1.3', config_path='configs'); \
  [print(v, compose('train', overrides=[f'model=vlm_backbone/{v}']).model._target_) for v in ('ctclip','fvlm')]"

# EDA 다시 돌리기
python scripts/run_eda.py --split valid --hu-sample 50

# 전체 pytest
pytest tests/ -q

# VLM3D dry-run
python -m src.vlm3d_runner --dry-run --out /tmp/smoke.json

# MAISI VAE 로드 (reuse 패턴)
python -c "from src.baselines.maisi import load_frozen; print(type(load_frozen(device='cpu')))"
```

---

## 7. Phase plan 요약

| Phase | 기간 | 주요 산출물 | 상태 |
|---|---|---|---|
| A | 5/26 → 5/31 | repo restructure, EDA, GenerateCT inference 준비, Report2CT 1-step gate, envelope lock | ✅ 완료 |
| B | 6/1 → 6/30 | Report2CT 학습(완료), 베이스라인 어댑터 4종(CT-CLIP/fVLM/GenerateCT/Text2CT), VLM3D eval 하네스, 우리 v1 설계 | 🔄 진행 중 |
| C | 7/1 → 7/31 | 우리 모델 v1 학습 + ablation + 진단 | — |
| D | 8/1 → 8/20 | final 학습 + submission docker + writeup + 제출 | — |

상세 계획·의사결정은 `.omc/plans/vlm3d-pivot-plan.md` 참조. (Phase B 초기 6/1 작업 목록 — precompute 스크립트, report2ct_repro experiment, launcher — 은 모두 완료됨.)

---

## 8. 약속한 컨벤션 (메모리에도 저장됨)

| 항목 | 출처 |
|---|---|
| **품질 > 일정 슬립** — 깨끗한 코드가 default | `[[user-prefers-quality-over-schedule]]` |
| **코드 이해 ↔ 구현 정렬** — 사용자가 step별로 직접 읽고 지시 | `[[user-drives-code-stepwise]]` |
| **Report2CT 학습은 [U]가 직접** | `[[report2ct-training-is-user-owned]]` |
| **Triplane 잠정 폐기** — `deprecated/`에서만 참조 | `[[triplane-deprecated-2026-05]]` |
| **ViSD-Boost 제거** — 코드+서브모듈 삭제(당분간 미사용) | `[[visd-boost-removed]]` |
| **third_party/ 는 read-only** (Principle P2) | plan §2 |
| **외부 코드 reuse 우선** — adapter로 감싸고 새로 안 짬 | user 지시 2026-05-26 |
| **항상 main 브랜치에서 작업** — worktree 금지 | CLAUDE.md |
| **데이터 위치**: 새 artifact → `/workspace/data/`; collaborator → `/workspace/datasets/` (read-only) | CLAUDE.md |

---

## 부록: 참고용 외부 링크

- Report2CT 논문: `paper_pdf/Report2CT.pdf`
- GenerateCT 논문: `paper_pdf/GenerateCT.pdf` (ECCV 2024)
- MAISI 논문: `paper_pdf/MAISI.pdf` (WACV 2025)
- VLM3D 챌린지 사이트: ctgen.vlm3dchallenge.com
- lightning-hydra-template: github.com/ashleve/lightning-hydra-template

---

질문이 더 있으면 — 이 가이드에서 다루지 않은 모듈/결정/명령이 있으면 알려주세요.
