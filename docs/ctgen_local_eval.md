# ctgen 로컬 평가 방법 (VLM3D Task 4)

> 우리 생성물(`.mha`/`.nii.gz`)을 **VLM3D-Dockers**의 ctgen 평가 스크립트로 로컬에서 채점하는 방법.
> 도커 데몬 없이 평가 스크립트를 직접 subprocess로 호출한다. 제출(돈 결제) 방법은
> [ctgen_challenge_submission.md](ctgen_challenge_submission.md) 참고.

## TL;DR

```bash
# 전체 파이프라인: 생성 → proxy GT 준비 → 채점
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
  task=ctgen model=<text2ct|report2ct|generatect> \
  task.n_samples=100 out_dir=/workspace/data/vlm3d_eval/<run_name>
```

- 채점기: [src/eval/tasks/ctgen.py](../src/eval/tasks/ctgen.py) `CTGenEvaluator`
- 평가 스크립트 위치는 코드가 자동 해석 → [src/eval/_vlm3d_paths.py](../src/eval/_vlm3d_paths.py)
  `ctgen_eval_dir()` (a945900 이후 `ct_challenges/ctgen_evaluation/`, 구경로 fallback).
- GT 셋: `ctrate_toy_v2/valid_v2` = valid_fixed 1304 one-scan-per-patient
  (`task.gt_dir` 기본 `/workspace/data/vlm3d_eval/_valid_v2_1304`).

## 3개 지표

| 키 | 백본 | 로컬 실행 | 비고 |
|---|---|---|---|
| `FVD_CTNet` | CT-Net (3D) | **❌ 불가 (현재)** | 리더보드 **1차** 지표. 아래 "FVD 주의" 참고 |
| `CLIPScore` / `_I2I` / `_mean` | CT-CLIP_v2 + BiomedVLP-CXR-BERT | ✅ | I2T(텍스트↔영상) + I2I(영상↔영상). I2T엔 prompt xlsx 필요 |
| `FID_2p5D_{Avg,XY,XZ,YZ}` | RadImageNet ResNet50 | ✅ | GPU로 feature 추출 → **CPU로 FID 집계**(GPU OOM 회피) |

> **win_condition 우선순위(우리 내부)**: `2.5D-FID > CLIPScore-T2I > FVD`.
> 단 **forithmus 리더보드의 1차 랭킹 지표는 `FVDCTNet`** 이다(공지 갱신). 즉 우리가 로컬에서
> 가장 신뢰하는 2.5D-FID/CLIPScore와, 서버가 1차로 보는 FVD가 다르다는 점을 항상 의식할 것.

## 동작 구조 (각 지표가 어떻게 도는가)

`CTGenEvaluator.evaluate(pred_dir, out_dir)` → 8키 dict + `out_dir/metrics.json`.

- **FVD** (`_run_fvd`): `evaluate_fvd.py --generated_dir --gt_root --out_json`.
  - `FVD/fvd_pytorch.py`가 `import ctnet`(namespace 패키지) + `torch.load(/opt/app/FVD/ctnet/trained_params/CTNet28_ctclip_whole_data_18classes)`를 한다.
  - 우리 코드가 (1) `ctnet` → `/opt/app/FVD/ctnet` 심링크(스테일 심링크는 자동 재지정),
    (2) `FVD/` 를 `PYTHONPATH`에 얹어 editable-install 깨짐과 무관하게 `ctnet`을 해석.
- **CLIPScore** (`_run_clip`): `evaluate_clip.py --generated_dir --gt_root [--prompt_xlsx] --out_json`.
  - 스크립트가 `/opt/app/models/{CT-CLIP_v2.pt,BiomedVLP-CXR-BERT-specialized}` 하드코딩 → 우리가 심링크.
  - `transformer_maskgit`/`ct_clip` 를 `PYTHONPATH`로 얹음(`ctclip_pkg_parents()`; reorg로 editable-install 깨져도 동작).
  - `prompt_xlsx` 없으면 I2T 생략(I2I만). proxy 모드에선 `run_eval.py`가 자동 생성.
- **FID-2.5D** (`_run_fid`): `torchrun _fid_runner.py ...` 로 per-volume feature 추출 →
  [src/eval/tasks/ctgen.py](../src/eval/tasks/ctgen.py) `_fid_from_cached_features()`가 CPU에서 FID 집계.
  - **shared GT-feature 캐시**: GT feature는 모델과 무관 → `_shared_gt_fidfeat/<key>`에 1회 계산 후
    모델마다 하드링크 재사용(공정 비교 + GPU 절약).

## ⚠️ FVD 주의 — 현재 로컬에서 계산 불가

VLM3D-Dockers에 커밋된 CT-Net 체크포인트
`ct_challenges/ctgen_evaluation/FVD/ctnet/trained_params/CTNet28_ctclip_whole_data_18classes`
는 **잘림/손상된 820 KB stub**(zip central directory 없음 → `torch.load` 실패:
`PytorchStreamReader ... failed finding central directory`). 이 블롭은 구핀(`c73fe07`)·신핀(`a945900`)에서
**동일 SHA**라 이번 업데이트와 무관하며, 과거 실행에서도 `FVD_CTNet`은 항상 `NaN`이었다
(예: `data/vlm3d_eval/ctgen/text2ct/metrics.json` → `"FVD_CTNet": NaN`).

- 즉 **FVD는 reorg 때문이 아니라 "진짜 가중치 부재" 때문에 막혀 있다.** 실제 가중치는 forithmus
  평가 컨테이너 내부(비공개)에 baked-in 되어 서버 채점 시에만 쓰인다.
- 로컬 FVD가 필요하면 진짜 `CTNet28_ctclip_whole_data_18classes`(수십 MB, 정상 zip) 확보가 선행과제.
  (CT-Net = Draelos *et al.*; CT-RATE 18-class 학습본은 챌린지 주최/CT-CLIP 측 산출물.)
- 그 전까지 로컬 신뢰 지표는 **2.5D-FID + CLIPScore** 두 가지. FVD는 실제 제출 후 리더보드로만 확인.

## 검증 기록 (2026-06-09, a945900 마이그레이션 직후)

8개 Report2CT 예측 × `_shared_gt_1000` 으로 스모크:

| 지표 | 결과 |
|---|---|
| FID-2.5D Avg | **6.77** (XY 4.29 / YZ 8.09 / XZ 7.94) ✅ |
| CLIPScore / I2I / mean | **42.72 / 41.29 / 42.01** ✅ |
| FVD_CTNet | `NaN` (손상 가중치 — 위 주의 참고) |

## 자주 나는 함정
- **스테일 심링크**: `/opt/app/FVD/ctnet` 이 구경로를 가리켜 dangling이면 `link.exists()`가 False라
  `symlink_to`가 `FileExistsError`. → `_setup_fvd_paths()`가 재지정. CLIP `/opt/app/models/*` 도 동일류.
- **editable-install 깨짐**: `ctnet`/`transformer_maskgit`/`ct_clip` 은 `pip install -e`로 구 서브모듈
  경로에 묶여 있어 reorg 후 `ModuleNotFoundError`. → 우리는 `PYTHONPATH` 주입으로 우회(재설치 불필요).
- **HU range/축순서**: `evaluate_fvd.py`는 HU `[-1000,1000]`→`[-1,1]`, `(D,H,W)` 가정. 생성물 spacing은
  모델별로 truthful하게 stamp([[ctgen_eval_spacing_convention]]).
- **출력 스키마**: a945900부터 평가 컨테이너 `metrics.json`은 `{"metrics": {...8키...}}`로 한 번 감싼다.
  우리 로컬 경로는 스크립트를 직접 호출해 자체 dict를 만들므로 영향 없음. 도커 컨테이너를 직접 돌리는
  [src/vlm3d_runner.py](../src/vlm3d_runner.py)만 `payload.get("metrics", payload)`로 언랩.
