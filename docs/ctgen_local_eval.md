# ctgen 로컬 평가 방법 (VLM3D Task 4)

> 우리 생성물(`.mha`/`.nii.gz`)을 **VLM3D-Dockers**의 ctgen 평가 스크립트로 로컬에서 채점하는 방법.
> 도커 데몬 없이 평가 스크립트를 직접 subprocess로 호출한다. 제출(돈 결제) 방법은
> [ctgen_challenge_submission.md](ctgen_challenge_submission.md) 참고.

## TL;DR

```bash
# 전체 파이프라인: 생성 → proxy GT 준비 → 채점
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
  task=ctgen model=<text2ct|report2ct|generatect> \
  task.n_samples=100 out_dir=/workspace/outputs/<model>/eval_<run_name>
```

- 채점기: [src/eval/tasks/ctgen.py](../src/eval/tasks/ctgen.py) `CTGenEvaluator`
- 평가 스크립트 위치는 코드가 자동 해석 → [src/eval/_vlm3d_paths.py](../src/eval/_vlm3d_paths.py)
  `ctgen_eval_dir()` (a945900 이후 `ct_challenges/ctgen_evaluation/`, 구경로 fallback).
- GT 셋(FID reference): **full clean valid census 3001** = valid_fixed 3038 − no_chest 37 −
  unencodable 1 (`data/ctrate_full/valid/ids.json`; `task.gt_dir` 기본
  `/workspace/data/vlm3d_eval/_valid_full_3001`). FID/FVD/CLIP은 비대칭 — GT 분포는
  모델과 무관하므로 real 3001 전부로 reference를 잡고, 모델은 여전히 valid_v2 1304
  (one-scan-per-patient, `load_eval_cases` 기본)만 **생성**한다. FVD/CLIP은 각 생성물을
  stem으로 자기 GT에 페어링(여분 GT 무시)하고, FID만 3001 전체를 reference로 쓴다.
  2.5D-FID는 이 3001의 평면별 (μ, Σ)를 **미리 계산해 재사용**([_fid_refstats.py](../src/eval/tasks/_fid_refstats.py)).
  (구 `_valid_v2_1304` = 1304-only GT는 superseded.)
  ⚠ **이 "3001 reference" 서술은 `fid_profile=research`에 한정된다.** 기본값인 `docker`
  프로파일에서 FID의 reference는 **채점되는 100개 예측과 동일한 stem의 GT 100개**다
  (CLIP/FVD는 프로파일과 무관하게 종전대로 stem 페어링). 아래 프로파일 절 참고.

## 3개 지표

| 키 | 백본 | 로컬 실행 | 비고 |
|---|---|---|---|
| `FVD_CTNet` | CT-Net (3D) | **❌ 불가 (현재)** | 리더보드 **1차** 지표. 아래 "FVD 주의" 참고 |
| `CLIPScore` / `_I2I` / `_mean` | CT-CLIP_v2 + BiomedVLP-CXR-BERT | ✅ | I2T(텍스트↔영상) + I2I(영상↔영상). I2T엔 prompt xlsx 필요 |
| `FID_2p5D_{Avg,XY,XZ,YZ}` | **프로파일에 따라 다름** (아래) | ✅ | GPU로 feature 추출 → **CPU로 FID 집계**(GPU OOM 회피) |

### 2.5D-FID 프로파일 (2026-07-29 도입, 기본값 = `[docker, docker_n300]`, 2026-08-01 개정)

공식 채점 컨테이너가 FID 스크립트에 넘기는 인자와 우리 harness가 넘기던 인자가 두 곳에서
달랐다. `task.fid_profile`이 고른다 — **이름 하나 또는 리스트**를 받고, 기본값은
`[docker, docker_n300]`(둘 다 채점)이다. **FID에만 영향**을 주며 CLIP/FVD는 동일하다.
`research`는 2026-08-01부로 기본에서 빠졌다 — `task.fid_profile=research` 또는
`task.fid_profile=[docker,docker_n300,research]`로 여전히 켤 수 있다.

| | `docker` (기본) | `docker_n300` (기본, 2026-08-01~) | `research` (opt-in, 2026-08-01~) |
|---|---|---|---|
| feature net | `squeezenet1_1` | `squeezenet1_1` (동일) | `radimagenet_resnet50` |
| feature 차원 | **1000** (ImageNet 클래스 로짓, ReLU 후) | 1000 | **2048** (pooled 임베딩) |
| 표본 | 앞 **100개** (양쪽 filelist 절단) | 앞 **300개** | 예측 **전량** vs GT **3001** |
| feature 경로 | `fid_features_squeezenet1_1/` | 동일 (네트워크가 같아 공유 안전) | `fid_features/` |
| 용도 | 앞으로의 모든 평가 (컨테이너 재현) | subgroup 기본 프로파일(100이 너무 적었음), 앞으로의 모든 평가 | 논문 envelope·2026-07-29 이전 수치 재현 (명시적으로 켜야 함) |

`docker`와 `docker_n300`은 **같은 네트워크**라 스케일이 비교 가능하지만 볼륨 수가 다르므로
(FID는 n에 민감) 한 표에 섞기 전 `fid_num_images`를 확인할 것.

**프로파일별 채점 결과는 각자의 프로파일 폴더에 들어간다** (2026-07-31).
`run_eval.py`로 냈든 `rescore_predictions.py`로 냈든 산출물은 `<eval_dir>/fid_<profile>/`
(`fid_docker` / `fid_docker_n300` / `fid_research`)에 쓴다. 최상위 지표 파일은 **통합
`summary.json` 하나뿐**이며, 여기선 FID가 프로파일별로 라벨링돼 있다. 아래 예시는
2026-08-01 기본값 변경 이전(`research`가 기본이던 시절)의 실측 run이라 `fid_research/`가
보인다 — 지금 기본값으로 새로 낸 run은 그 자리에 `fid_docker_n300/`이 대신 생긴다:

```
eval_ep299_n300_sp0.75_1.3_cfg1/
├── fid_research/     metrics.json  summary.json  clip.json  fid.json  fvd_ctclip.json  fid_features/
├── fid_docker/       metrics.json  summary.json  fid.json  fid_features_squeezenet1_1/
├── _shared_pred_fidfeat/  <model>/<hash>/   ← docker/docker_n300 pred-feature cache (2026-08-01)
├── summary.json      ← 통합 (아래 스키마)
└── predictions/  latents/  analysis/  prompts.xlsx  .hydra/  run_eval.log   ← 프로파일 무관
```
`_shared_pred_fidfeat/`는 `docker`와 `docker_n300`이 같은 run 안에서 예측 feature를 서로
재사용하게 해주는 캐시다(docker의 100개는 항상 docker_n300 300개의 정렬 기준 앞 100개와
동일). 키는 feature 모델 + `pred_dir` 경로 해시 — 부모 디렉토리만으로 키를 잡으면
`tests/orientation_quant/`, `tests/spacing_fov/`처럼 한 부모 아래 서로 다른 예측 세트를
같은 파일명으로 채점하는 하네스가 캐시를 잘못 공유할 위험이 있어서다.

통합 `summary.json`은 공통 지표를 한 번, FID는 프로파일별로 **중첩**해 담는다. 최상위 파일이
위험했던 이유는 "어느 프로파일인지 모르는 FID"였는데, `fid.<profile>` 아래에선 그런 수치가
없다:

```json
{"task": "ctgen", "model": "report2ct_wan_mask_v2",
 "fid_profiles": ["docker", "research"],
 "metrics": {"FVD_CTCLIP": 0.269, "CLIPScore_T2I": 59.42},
 "fid": {"docker":   {"FID_2p5D_Avg": 47.19, "fid_profile": "docker",   "fid_num_images": 100},
         "research": {"FID_2p5D_Avg": 1.455, "fid_profile": "research", "fid_num_images": 300}}}
```

`fid_<profile>/summary.json`은 예전 스키마(그 프로파일 지표만 평평하게) 그대로다.

### 두 프로파일을 함께 재는 비용 (n=300, GT 캐시 warm — 실측)

| 단계 | 시간 | 비고 |
|---|---|---|
| 생성 (Wan gen+decode / report2ct 샘플러) | **2h35m / ~11h** | 압도적으로 지배적 |
| FVD_CTCLIP | ~12분 | 프로파일 무관 → **첫 패스에서 1회만** |
| CLIPScore | ~12분 | 〃 |
| FID `docker` (100 vol, squeezenet) | **~7분** | |
| FID `research` (300 vol, radimagenet) | **~31분** | pred feature 3.6 GB |

두 프로파일은 feature net이 달라 캐시를 공유할 수 없어 시간이 그대로 더해진다. 그래도
두 번째 프로파일 추가분(+7 또는 +31분)은 생성 단계에 비하면 작아서 기본값을 둘 다로 뒀다.
CLIP/FVD를 두 번 돌리지 않는 것이 핵심이다 (`_score_profiles`가 강제).

프로파일이 폴더를 정하므로 `docker` 채점이 `research` 채점에 **닿을 수 없다** — 아래 2026-07-29
사고를 런타임 가드가 아니라 레이아웃이 막는다. 그래서 `_refuse_cross_profile_overwrite`와
`task.allow_overwrite`는 2026-07-31에 삭제했다. 집계기는 프로파일을 추측하지 않고
`--profile`로 받는다 (한 dir이 여러 프로파일을 정상적으로 갖기 때문).

**`metrics.json`은 덮어쓰지 않고 병합한다.** 채점 패스는 자기가 켠 지표만 계산하는데, 예전에는
그 dict를 통째로 써서 **이미 기록된 나머지를 날렸다** (CLIP+FID가 있던 곳에 FVD만 재면 1-키 파일이
됐다 — 별도 `fvd/` 하위 디렉터리를 뒀던 이유가 이것뿐이었다). 지금은 기존 파일을 읽어 갱신하고
`_history`에 패스별로 날짜 · 실행한 metric set · 추가된 키 · 교체된 키(이전 값 포함)를 남긴다.
지표 키 자체는 **최상위 평평**하게 유지한다 — 모든 집계기가 `metrics.get("FID_2p5D_Avg")`로 읽는다.

> ⚠ **덮어쓰기 사고는 `run_eval.py`에서 났다.** 2026-07-29에
> `wan_mask_v2` 두 런의 research FID가 날아간 원인은 재채점 스크립트가 아니라
> `run_eval.py … out_dir=<기존 eval dir> task.fid_profile=docker` 였다
> (`outputs/report2ct_wan_mask_v2/eval_2026-07-29_{2,3}/.hydra/overrides.yaml`).
> 샘플러가 기존 `.mha`를 스킵하므로 재생성 없이 몇 분 만에 끝나면서 최상위 `metrics.json`과
> `summary.json`을 다른 지표군으로 다시 썼다. 당시 배선(`out_dir: ${hydra:runtime.output_dir}`)에선
> `out_dir`만 오버라이드하면 Hydra가 자기 날짜 디렉터리에 `.hydra`를 써서 **eval dir의 `.hydra`가
> 원래 런 그대로** 남았고, 그래서 하루 동안 원인을 재채점 스크립트로 오귀속했다.
> 2026-07-30에 배선을 반전해(`hydra.run.dir: ${out_dir}`) `out_dir=`이 로그까지 함께 옮기고,
> 갈라지는 지정은 `_refuse_split_run_dir`이 거부한다. ⚠ 그 이전 날짜의 `.hydra`는 여전히
> 자기 디렉터리의 증거가 못 된다.
> 복구는 캐시된 `fid_features/`에서 정확히 재계산했다(1.339036 / 1.455624). 이때 `metrics.json`만
> 되돌리고 `summary.json`을 빠뜨리기 쉬우니 **둘 다** 확인할 것.
>
> **지금은 이 경로가 구조적으로 막혀 있다** — 프로파일마다 폴더가 다르고, 같은 폴더에 다시 채점해도
> 병합이라 기존 키가 살아남는다.

- 근거: [evaluation.py:229-230](../third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/evaluation.py)이
  `--num_images 100 --model_name squeezenet1_1`을 넘긴다. `--model_name`은 재핀(a945900) 때
  상류가 추가한 것이라 우리가 놓쳤고(**버전 drift**), `--num_images 100`은 상류 최초 커밋부터
  있었는데 우리는 처음부터 전량을 썼다(**최초부터 불일치**).
- SqueezeNet은 ImageNet(자연영상) 학습이고 feature도 penultimate 임베딩이 아니라 1000-클래스
  로짓이다. 도메인 적합성은 RadImageNet이 낫지만, 상류가 그렇게 채점하므로 그대로 따른다.
  (상류가 바꾼 이유는 명시돼 있지 않으나, `_chunked_forward_avg` docstring의 *"512+ slices of
  512x512 fit on a **24 GB L4** without OOM"* 과 Dockerfile:62의 빌드타임 squeezenet 캐시로 보아
  **오프라인 + 24 GB GPU 제약**으로 추정된다. RadImageNet 경로는 런타임에 `torch.hub`로
  GitHub에 접속해야 한다.)
- **두 프로파일의 값은 절대 같은 표에 섞지 말 것.** 같은 예측물 실측 예: docker **57.87** vs
  research **1.61**. `metrics.json`에 `fid_profile`과 `fid_num_images`가, `fid.json`에
  추가로 `model_name`/`scored_stems_sha1`이 기록된다(옛 `fid.json`엔 `profile`
  키가 없으며 그건 research를 뜻한다). `scripts/aggregate_wan_epoch_sweep.py`는 한 스윕에
  두 프로파일이 섞이면 **집계 전에 중단**한다.
- **⚠ 두 프로파일은 모델 순위가 다르다 (2026-07-29, 동일 예측물 12런 실측).**
  Spearman ρ = 0.748 / Kendall τ = 0.545 — 상관은 있으나 대체 가능하지 않다. 가장 큰 이동은
  `report2ct_wan_mask`(research 6위 → **docker 1위**)와 `report2ct` cfg7/cfg5(1·2위 → 5·6위).
  최하위 2개(text2ct 97.6, generatect 328.8)는 두 프로파일이 일치하며 docker가 훨씬 크게 벌린다.
- **docker n=100은 표본 노이즈가 커서 작은 FID 격차는 큰 의미가 없다.** paired volume bootstrap
  300회([tests/fid_profile_bootstrap/run.py](../tests/fid_profile_bootstrap/run.py), 같은 재표집
  볼륨으로 두 모델을 동시에 채점해 공통 분산을 상쇄):

  | 비교 | 관측 차 | bootstrap std | P(A가 더 좋음) | 판정 |
  |---|---|---|---|---|
  | wan_mask_v2 vs wan_mask | +0.26 | 1.71 | 0.43 | **구분 불가 (동전던지기)** |
  | wan/cfg3 vs report2ct/cfg5 | −0.78 | 2.86 | 0.68 | **유의하지 않음** |
  | wan_mask_v2 vs wan/cfg3 | −8.50 | 3.25 | **0.993** | **유의함** |

  모델별 std는 2.5~3.1(값의 ~5 %)이다. mask 계열이 나머지를 앞선다는 결론은 견고한 반면,
  3~6위 안의 순서나 mask 두 모델 사이의 우열처럼 작은 격차는 docker FID만으로 판단하기 어렵다 —
  그 구간은 CLIP·FVD를 함께 보는 편이 낫다.
  **이 주의사항은 2.5D-FID에만 해당한다** — CLIP과 FVD_CTCLIP은 프로파일과 무관하게 생성물
  전량(300)으로 측정되므로 이 노이즈의 영향을 받지 않는다. (bootstrap 평균이 point보다 1~1.3 높은 것은 복원추출로 중복 볼륨이
  생겨 다양성이 줄어드는 FID의 알려진 상향 편향이다.)
- **지표별 표본 수가 다르다**: docker 프로파일에서 FID는 앞 100개만, CLIP/FVD는 전량을 쓴다.

### docker가 채점하는 100개는 어느 것인가

상류는 filelist를 **자체 정렬 후 앞 100개로 절단**한다([compute_fid_2-5d_ct.py:555-556, :569-570]
(../third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/compute_fid_2-5d_ct.py)). 우리도 같은
방식으로 **이 런 예측의 정렬 앞 100개**를 쓰고, GT는 동일 stem으로 짝지운다.

같은 `task.n_samples`로 만든 런끼리는 동일 집합이라(`load_eval_cases`의 부분집합이 nested)
상호 비교가 성립한다 — 실제로 재채점한 12런 모두 `scored_stems_sha1 = 5a07f98fee91`로 같았다.
**n이 다른 런끼리는 채점 집합이 달라 비교할 수 없고**, 그걸 드러내라고 `fid.json`에 그 해시를 남긴다.

> **"리더보드가 채점할 100개"를 로컬에서 재현하는 옵션은 없다 — 원리상 불가능하다.**
> 서버는 우리가 갖고 있지 않은 **별도의 히든 테스트셋**을 채점하므로, valid_v2에서 어떤 100개를
> 고르든 그것과 일치시킬 수 없다. (한때 `fid_scored_set=canonical` 옵션이 있었으나 이 이유로
> 2026-07-29 제거했다. 로컬 수치는 모델 간 상대 비교용이고, 리더보드 절대값은 제출로만 확인된다.)

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
    모델마다 하드링크 재사용(공정 비교 + GPU 절약). 캐시 키에 **feature net 이름이 포함**되어
    프로파일별로 자동 분리된다. `_shared_gt_feat_dir(gt_dir, model)`의 `model`은 **기본값 없는
    필수 인자** — 상류가 feature 파일명에 모델명을 넣지 않기 때문에(그 `suffix` 변수는 dead
    code), 기본값을 두면 호출부가 조용히 다른 feature 공간을 같은 디렉터리에 섞을 수 있다.
  - **기존 예측 재채점**: 재생성 없이 `scripts/rescore_predictions.py --metrics fid_2p5d
    --fid-profile docker --pred-dir <d>/predictions --out <d>/fid_docker`. `--out`을 새 하위
    디렉터리로 주면 원본 `metrics.json`은 그대로 보존된다.

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
- **상대 경로 금지**: 세 지표 모두 상류 스크립트를 `cwd=third_party/.../ctgen_evaluation`으로
  실행한다. 상대 경로를 넘기면 자식이 **그 디렉터리 기준**으로 해석해 filelist를 못 찾고
  `FileNotFoundError`(또는 빈 데이터셋 → `there is no enough data to be split into 1 partitions`)로
  죽는다. 2026-07-29에 `CTGenEvaluator`가 `gt_dir`/`pred_dir`/`out_dir`/`prompt_xlsx`를 내부에서
  `.resolve()` 하도록 고쳤고 `test_subprocess_paths_are_absolute`가 이를 고정한다.
  (증상: 30초 만에 종료 + `incomplete cached pred features (N/N missing)`.)
- **스테일 심링크**: `/opt/app/FVD/ctnet` 이 구경로를 가리켜 dangling이면 `link.exists()`가 False라
  `symlink_to`가 `FileExistsError`. → `_setup_fvd_paths()`가 재지정. CLIP `/opt/app/models/*` 도 동일류.
- **editable-install 깨짐**: `ctnet`/`transformer_maskgit`/`ct_clip` 은 `pip install -e`로 구 서브모듈
  경로에 묶여 있어 reorg 후 `ModuleNotFoundError`. → 우리는 `PYTHONPATH` 주입으로 우회(재설치 불필요).
- **HU range/축순서**: `evaluate_fvd.py`는 HU `[-1000,1000]`→`[-1,1]`, `(D,H,W)` 가정. 생성물 spacing은
  모델별로 truthful하게 stamp([[ctgen_eval_spacing_convention]]).
- **출력 스키마**: a945900부터 평가 컨테이너 `metrics.json`은 `{"metrics": {...8키...}}`로 한 번 감싼다.
  우리 로컬 경로는 스크립트를 직접 호출해 자체 dict를 만들므로 영향 없음(도커 컨테이너를 통째로 돌리지 않음).
