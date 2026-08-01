# eval 코드 이해 가이드 — 읽기 순서 + 실행 예시 (절대 경로)

## Context

`run_eval` / `eval` 관련 코드가 최근 두 갈래로 크게 바뀌었다: (1) `fid_profile` 시스템(docker/research/docker_n300 — 이번 세션 내내 논의한 부분), (2) per-sample/subgroup/Dice/QC 확장 레이어(`src/eval/analysis/`, 세션 초반에 구현). 사용자가 "새로 변경된 코드를 내가 이해 못 하고 있다"며 직접 코드를 읽고 검증하고 싶다고 했음 — 이 문서는 추가 구현 계획이 아니라, **지금 이 턴에 실제 파일을 다시 읽어서 검증한** 읽기 순서 가이드 + 실행 예시다. 3개의 Explore 서브에이전트가 `src/eval/analysis/` 전체, `fid_profile` 배관, entry-point 스크립트를 각각 훑었고, 이어서 `third_party/vlm3d_dockers`(공식 docker 원본 코드 — squeezenet 경로/n=100의 실제 출처)까지 1개 더 확인했다. 아래 줄번호는 전부 이 턴에 실제로 읽어서 확인된 것만 인용.

**총 분량 감** — analysis 패키지 13개 파일(0~411줄), task 계열 핵심 3개 파일, entry-point 스크립트 5개, config 2개, upstream 파일 2~3개, test 17개. 전부 정독할 필요는 없고, 아래는 "이해에 필요한 최소 순서"다.

> ⚠ **2026-07-31 재검증 업데이트**: 이 가이드 작성(07-30 14:34) 이후 `ctgen.py`/`run_eval.py`/`orchestrate.py`/`summary.py`/`ctgen.yaml`가 실제로 바뀜(mtime 확인) + `score_pred_dir.py`가 `rescore_predictions.py`로 rename됨 — 전부 재확인 완료. 핵심: `CTGenEvaluator`의 출력(`metrics.json`/`fid.json`/filelist 등)이 라이브 `run_eval.py` run에서도 이제 top-level `out_dir`이 아니라 **`<out_dir>/fid_<profile>/`**로 감(기존엔 post-hoc rescore만 그랬음) — profile 충돌을 가드로 막던 것에서 애초에 경로가 안 겹치게 만드는 구조적 수정으로 바뀜. `metrics.json`은 이제 매 스코어링 패스마다 덮어쓰지 않고 `_history` 기록과 함께 **병합**됨. 아래 Part 3/Part 5/Part 7/Part 8에 정정 반영. Part 4(analysis 패키지) + Part 3의 8-11번(`_fid_refstats.py`/`subgroup_refstats.py`/`precompute_subgroup_fid_refstats.py`/`subgroup_setlevel.py`)은 mtime상 이 변경 이전부터 그대로라 안 건드림.

---

## Part 0 — 코드 읽기 전에 (5분)

1. `/workspace/docs/ctgen_local_eval.md` — 이미 존재하는 문서. 목차:
   ```
   1:   # ctgen 로컬 평가 방법 (VLM3D Task 4)
   7:   ## TL;DR
   10:  # 전체 파이프라인: 생성 → proxy GT 준비 → 채점
   31:  ## 3개 지표
   39:  ### 2.5D-FID 프로파일 (2026-07-29 도입, 기본값 = docker)
   113: ### docker가 채점하는 100개는 어느 것인가
   132: ## 동작 구조 (각 지표가 어떻게 도는가)
   155: ## ⚠️ FVD 주의 — 현재 로컬에서 계산 불가
   170: ## 검증 기록 (2026-06-09, a945900 마이그레이션 직후)
   180: ## 자주 나는 함정
   ```
   39번과 113번 섹션부터 먼저 읽는 게 제일 빠름 — `fid_profile` 프로즈 설명이 이미 있음.

2. `/workspace/docs/submodule_pins.md` 10번 줄 + 30-33번 줄 — **`fid_profile`이 왜 생겼는지의 실제 원인**:
   - 10번 줄: `vlm3d_dockers` 핀 SHA = `a94590095847824d664e3a86f357924207f777fb` (2026-06-09 재핀, 이전 `c73fe07`)
   - 30-33번 줄: 이 재핀이 컨테이너 자신의 FID 호출에 `--model_name squeezenet1_1`을 조용히 추가했는데, path reorg 때문에 재핀한 거라 아무도 눈치 못 채고 **7주간** 우리 FID가 (별도 지정 없이) 스크립트 자체 기본값인 `radimagenet_resnet50`을 계속 써왔음. 2026-07-29에 발견. `fid_profile`은 그 재발 방지용 — "어떤 프로파일인지 명시적으로 고르고 기록"하게 만든 것.

---

## Part 1 — upstream vlm3d-docker 원본 (우리 wrapper가 재현하려는 "스펙")

우리 wrapper를 읽기 전에 원본을 먼저 보면 이해가 빠름.

3. **`/workspace/third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/evaluation.py`** — 컨테이너 진입점.
   - 216-231번 줄: 실제 `torchrun` 호출. **229-230번 줄이 `--num_images 100 --model_name squeezenet1_1`을 리터럴로 넘김** — 우리가 추측한 게 아니라 공식 컨테이너가 실제로 이렇게 부름.
   - 196-199번 줄: `evaluate_clip.py` 호출. 185-187번 줄: `evaluate_fvd.py` 호출.

4. **`/workspace/third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/compute_fid_2-5d_ct.py`** — 실제 FID 스크립트 (`torchrun`으로 실행되는 그 파일).
   - 515-524번 줄: feature network 선택 — `if model_name == "radimagenet_resnet50": ... else: squeezenet1_1`. **squeezenet1_1은 스크립트 자체 기본값이 아님** (스크립트 자체 기본은 397번 줄 `model_name="radimagenet_resnet50"`) — `evaluation.py`가 명시적으로 `--model_name squeezenet1_1`을 넘겨야만 도달하는 분기.
   - 553-556번 줄(real) / 567-570번 줄(synth): `num_images` 자르기 로직 — 파일 리스트를 **정렬 후 앞에서 N개 slice**. "채점되는 100개"는 정렬 순서 기준 앞 100개, 그 이상의 로직 없음.
   - 626번 줄(real) / 666번 줄(synth): 피처 캐시 체크(`os.path.isfile`) — **캐시 키가 볼륨 파일명에만 의존**(622/662번 줄 `out_fp` 조립에 모델명 안 들어감; 519/524에서 만든 `suffix` 변수는 만들어놓고 캐시 키엔 안 씀). 우리 `ctgen.py` 57-64번 줄이 바로 이 사실을 인용하며 "그래서 `feat_subdir`는 profile마다 달라야 한다"고 설명하는 부분.
   - 646-648번 줄 + 734-740번 줄: `torch.vstack`로 GT 전체를 GPU에 한 번에 쌓는 부분 — 이게 3001-GT 스케일에서 OOM 나는 원인(우리 `ctgen.py` 197-199, 910-917번 줄이 이걸 명시).
   - ⚠ **함정**: 같은 폴더에 이름이 비슷한 `evaluate_fid_2p5d_ct.py`(하이픈 없음, `evaluate_` 접두사)가 따로 있는데 이건 `evaluation.py`가 절대 호출 안 하는 죽은 코드. `compute_fid_2-5d_ct.py`(하이픈 있음, `compute_` 접두사)만 진짜.

---

## Part 2 — 우리 config (지금 만질 수 있는 노브)

5. `/workspace/configs/eval/task/ctgen.yaml` (93줄, 전체 한 번 읽기) — 특히:
   - `fid_profile: docker` (25-46번 줄 주변에 docker/docker_n300/research 설명 주석 블록)
   - `subgroup_fid_profile: research` (87번 줄 근처) — **`fid_profile`과 완전히 별개 키**임을 여기서 확인 (헷갈리기 쉬운 지점)
   - `metrics:` 블록(54-66번 줄) — 기존 4개(`fvd/fvd_ctclip/clip_score/fid_2p5d`) + 신규 5개(`per_sample/dice/hd95/subgroup/subgroup_setlevel`, 전부 기본 `false`)
   - `qc_figures: true`, `qc_n: 5` — 유일하게 **기본 on**인 신규 플래그
6. `/workspace/configs/eval/subgroup/default.yaml` (44줄) — `label_burden_bands`(9-13), `organ_clusters`(17-39, 4개 클러스터), `subgroup_fid_small_n: 100`(43번 줄). 5-7번 줄 주석: mask-explainability(축 D)는 의도적으로 이 파일에 없음(보류).

---

## Part 3 — `fid_profile` 메커니즘 (우리 wrapper) — 이번 세션 대화의 핵심

7. **`/workspace/src/eval/tasks/ctgen.py`** (07-31 재확인: 1056줄, +94줄 — 아래 줄번호 전부 갱신) — 이 순서로:
   - **L30-86: `_merge_metrics(out_dir, results, metric_names)` — 2026-07-31 신규**. `metrics.json`을 덮어쓰지 않고 이전 패스의 키를 보존한 채 병합, `_history`에 날짜별 감사 기록(추가/교체된 키) 남김. docstring: "FVD만 재채점하면 기존 CLIP+FID가 담긴 파일이 1-key로 줄던" 문제를 고친 것.
   - L134-151(구 50-93번): `_FID_PROFILES` 딕셔너리 본체(134-150) + `feat_subdir` 유일성 설명 블록 주석(111-133, Part 1의 4번과 연결됨, 더 길어짐). `_DEFAULT_FID_PROFILE = "docker"`(151번 줄).
   - L175-247(구 114-186번): `_shared_gt_feat_dir(gt_dir, model)`(175-190) / `_link_shared_gt_features`(193-228) / `_populate_shared_gt_features`(231-247) — **run 간 공유되는 GT 피처 캐시**, 로직 동일·줄번호만 이동.
   - L250-403(구 189-342번): `_fid_from_cached_features` — subprocess 자체가 뱉는 값을 절대 안 믿고, 캐시된 볼륨별 피처에서 CPU로 재계산하는 부분(OOM 회피), 로직 동일.
   - L406-496(구 345-427번): `CTGenEvaluator.__init__`(424-449, `fid_profile` 파라미터 검증) + **`evaluate()`(455-496, 내용 바뀜)** — 이제 `results`를 직접 `json.dump` 안 하고 `_merge_metrics(out_dir, results, ran)`를 호출해서 반환.
   - **L773-1031(구 704-962번): `_run_fid`** — 골격 동일, 제일 중요. 특히: 824-859(volume-count 분기), 868(`features_dir = out_dir / prof["feat_subdir"]`), 909-916(ref-stats npz 경로), 961-962(subprocess `--num_images`).
   - ⚠ **`_run_fid`/`evaluate()`가 받는 `out_dir` 인자 자체의 의미가 바뀜** — `ctgen.py` 내부 로직은 안 바뀌었지만, 이제 호출부(`run_eval.py`)가 top-level `out_dir`이 아니라 `<out_dir>/fid_<profile>/`을 넘겨줌. 자세한 건 Part 5.

8. **`/workspace/src/eval/tasks/_fid_refstats.py`** (307줄, 순수 수학 레이어) — 파일 순서대로: `compute_ref_stats`/`save_ref_stats`/`load_ref_stats`/`fid_from_ref_stats`(39-155, 기존 CPU 경로) → `_frechet_distance_fast`(185-228, 이번 세션에 추가한 GPU Cholesky+eigvalsh, 실패 시 CPU로 자동 폴백) → `fid_from_ref_stats_files_fast`/`fid_from_ref_stats_fast`/`_mu_sigma_from_files`/`fid_from_file_lists_fast`(231-307).

9. **`/workspace/src/eval/analysis/subgroup_refstats.py`** (129줄) — **8번과 이름이 비슷하지만 다른 파일**이니 혼동 주의. "29축" 서브그룹 전용 캐시: `subgroup_refstats_dir(gt_dir, model)`(46-58) / `load_cached_axis_refstats`(61) / `precompute_axis_refstats`(85).

10. `/workspace/scripts/precompute_subgroup_fid_refstats.py` — 9번의 캐시를 실제로 만드는 CLI. `--fid-profile`(기본 `research`, choices는 `_FID_PROFILES` 전체) 인자로 어느 네트워크의 캐시를 만들지 고름.

11. **`/workspace/src/eval/analysis/subgroup_setlevel.py`** (411줄) — `run()`(262-408, 파라미터에 `fid_profile: str = "research"` 있음 — config의 `subgroup_fid_profile`이 여기로 흘러들어옴) → 내부 `_fid_subset()`(44-153, 축별 캐시 히트 시 fast path, 아니면 pred만 재적재, 그것도 없으면 GT+pred 둘 다 full restack).

---

## Part 4 — per-sample / subgroup / Dice / QC 확장 레이어 (세션 초반 구현, additive)

**`/workspace/src/eval/analysis/orchestrate.py`(07-31 재확인: 132줄, +3줄)를 제일 먼저** — 이게 전체 호출 순서의 지도임. `run_metrics_analysis()`가 **신규 파라미터 `metrics_path`를 받게 됨**(2026-07-31 — `metrics.json`이 top-level에서 `fid_<profile>/`로 옮겨간 것을 `summary.write_summary`에 알려주기 위함, Part 3/Part 5 참고). 디스패치 본문(89-129번 줄, 실측 갱신):
```python
89      if not any([per_sample, dice, hd95, subgroup_flag, subgroup_setlevel_flag]):
90          return
...
98      df = persample.build_per_sample(...)      # 플래그 하나라도 켜지면 무조건 재생성
111     df.to_csv(paths["per_sample_csv"], index=False)
114     if subgroup_flag: subgroup.run(...)
117     if subgroup_setlevel_flag: subgroup_setlevel.run(...)
129     summary.write_summary(out_dir, metrics_path=metrics_path)   # ← 인자 하나 늘어남 (구: summary.write_summary(out_dir))
```
`run_qc_figures()`(39-57번 줄)는 이 디스패치와 완전히 독립 — 어떤 metric 플래그와도 무관하게 항상 호출됨(`run_eval.py`가 기본 `true`로 부름).

이 지도를 따라 leaf → root 순서로 (파일, 줄수, 한 줄 요약):

| 순서 | 파일 | 줄수 | 요약 |
|---|---|---|---|
| 12 | `src/eval/analysis/labels.py` | 130 | 18-label 상수/burden/class/cluster 파생 — 나머지 전부의 기반 |
| 13 | `src/eval/analysis/subgroup_config.py` | 62 | `subgroup/default.yaml` 로드+검증 |
| 14 | `src/eval/analysis/segment.py` | 89 | TotalSegmentator 래퍼. `_segment_vista3d`(49-62)는 의도적 스텁(`NotImplementedError`) — 버그 아님 |
| 15 | `src/eval/analysis/seg_metrics.py` | 382(08-01, +141) | Dice + 표면거리(HD95/HD/ASSD), `compute_seg_metrics`(274) — input mask 기준 + GT mask 기준 둘 다. **2026-08-01**: `_surface_metrics_per_organ`(139)이 MONAI 리덕션을 그대로 복제해 거리 패스 1회로 3지표를 뽑음(실측 2.8배 빠름, `# MODIFIED` 주석으로 원본 대조 가능); 한쪽만 빈 organ은 upstream이 HD/ASSD에 inf를 주므로 우리가 NaN으로 고정(`OUR POLICY`). ⚠ 표면거리 `_mean`은 NaN 아닌 organ만 평균 → 장기를 통째로 못 만든 모델이 유리해짐(Dice는 0점 처리). 그래서 분모를 `n_organs_scored_to_<ref>` + `organs_missing_to_<ref>`(`"esophagus:pred"` 형식, 빈 쪽을 명시)로 같이 기록하고 subgroup에서도 평균 집계 |
| 16 | `src/eval/analysis/clip_persample.py` | 202 | per-sample CLIP-T2I/I2I (`run`, 40번 줄) — upstream이 `np.mean`으로 버리는 값을 보존 |
| 17 | `src/eval/analysis/qc_figures.py` | 229 | GT-vs-생성물 3-plane 그림 (`run`, 175번 줄) |
| 18 | `src/eval/analysis/persample.py` | 267 | `analysis/per_sample.csv` 생성 (`build_per_sample`, 92번 줄) — 모든 downstream이 groupby하는 원자 테이블 |
| 19 | `src/eval/analysis/subgroup.py` | 107 | per_sample.csv → per_abnormality/label_burden/organ_cluster (`run`, 80번 줄). **축 D(mask-explainability)는 스텁조차 없음** — 5-8번 줄 docstring: 사용자가 세그맵 육안 확인 후 정의하기로 보류 |
| 20 | `src/eval/analysis/summary.py` | 181(07-31 재확인, +9) | `analysis/SUMMARY.md` 롤업 (`write_summary(out_dir, metrics_path=None)`, 139번 줄). **2026-07-31**: `metrics_path` 신규 파라미터 — `metrics.json`이 top-level에서 `<out_dir>/fid_<profile>/`로 옮겨간 뒤에도 헤드라인 섹션이 그걸 찾게 함; `None`이면 예전처럼 top-level을 봐서 그 이전 run도 그대로 렌더링됨 |

---

## Part 5 — 진입점 스크립트 (실제로 어떻게 돌리는가)

21. `/workspace/scripts/run_eval.py` (07-31 재확인: 312줄, +8줄) — 모듈 docstring(1-76번 줄)에 예시 명령 + `task.fid_profile`/`task.is_mask_model` 등 신규 플래그 레퍼런스가 있음. `_run_ctgen`(128-276):
    - **⚠ 정정**: 예전에 "143번 줄 프로파일 충돌 가드(`_refuse_cross_profile_overwrite`)"라고 썼는데, 그 함수는 지금 코드에 없음 — 애초에 가이드 작성 시점에 이미 낡은 인용이었음. 지금 있는 건 **완전히 다른** `_refuse_split_run_dir`(101-125번 줄, `main()`에서 L290에 호출) — `out_dir`과 Hydra의 run dir이 다른 폴더를 가리키면 거부(프로파일과 무관, "`hydra.run.dir=`로 override하지 말고 `out_dir=`을 쓰라"는 가드).
    - **L149-151: `score_dir = out_dir / f"fid_{fid_profile}"` — 2026-07-31 신규.** 이게 이번 변경의 핵심: `CTGenEvaluator.evaluate(pred_dir, score_dir)`(L234)가 이제 top-level `out_dir`이 아니라 이 profile-이름 폴더에 씀 — 라이브 run과 사후 재채점(`rescore_predictions.py`)이 같은 위치 컨벤션을 공유하게 됨(프로파일 충돌이 가드가 아니라 애초에 구조적으로 불가능해짐).
    - L189-203(`qc_figures` 게이트, try/except) → L209-225(`per_sample`+`clip_score` 동시 켜지면 CLIP 중복 실행 코드로 차단) → L227-234(`CTGenEvaluator` 생성 + `evaluate(pred_dir, score_dir)`) → L244-265(`orchestrate.run_metrics_analysis` 호출 — **`metrics_path=score_dir / "metrics.json"` 인자 추가**, Part 4 참고).
22. `/workspace/scripts/rescore_predictions.py`**(구 `score_pred_dir.py` — 2026-07-31에 rename됨, 내용은 거의 동일 115줄)** — 기존 VLM3D 4개 지표만 사후 재채점. `<out>/fid_<profile>/` 서브폴더 컨벤션을 **소유**하는 스크립트(L82-86). 2026-07-29 클로버 사고의 직접적 수정. docstring이 이제 `run_eval.py`도 같은 위치 컨벤션을 쓰게 됐다고 명시.
23. `/workspace/scripts/score_subgroups.py` — per-sample/dice/subgroup/subgroup_setlevel/QC를 기존 predictions에 사후 실행. `--subgroup-fid-profile` 포함 전체 argparse가 36-74번 줄.
24. `/workspace/scripts/compare_runs.py` — 두 run의 `analysis/per_sample.csv`를 `target_id`로 inner join해 페어 비교(36-68번 줄), 출력은 `<out>/per_sample_delta.csv` + `<out>/subgroup_delta_<축>.csv`(152-171번 줄). CI 없음(의도적).

---

## Part 6 — 테스트 (검증 수단 그 자체)

`/workspace/tests/eval_analysis/` 17개 파일 — 위 모듈과 거의 1:1 대응:
```
_helpers.py                        test_manifest_consume.py           test_subgroup_config.py
test_backward_compat.py            test_output_layout.py              test_subgroup_fid_smalln.py
test_clip_persample_equiv.py       test_persample_schema.py           test_subgroup_refstats.py
test_compare_runs.py               test_qc_figures.py                 test_subgroup_stats.py
test_fid_gpu_frechet.py            test_seg_metrics_dice.py
test_fid_profiles.py               test_labels.py
test_surface_metrics.py            (2026-08-01: was test_hd95_mm.py — HD95 + HD + ASSD)
```
실행: `CUDA_VISIBLE_DEVICES=3 python -m pytest tests/eval_analysis -q` (직전 실행 결과 77 passed / 1 skipped — skip은 `test_clip_persample_equiv.py`, 환경 게이트로 추정, 이번 세션 변경과 무관). `test_fid_profiles.py`가 Part 3의 profile-분리 불변식을 직접 검증하는 파일이라, 이해가 맞는지 셀프체크하기 제일 좋음.

---

## Part 7 — 실행 예시 + 실제로 저장되는 폴더 구조 (전부 실측)

### 예시 A — 지금 당장 디스크에 있는 것 (명령 불필요)
`qc_figures`는 기본 `true`라서 이미 25개의 실제 run에 결과가 있음:
```
/workspace/outputs/report2ct_wan_mask_v2/eval_ep299_n300_sp0.75_1.3_cfgt5_cfgm0/analysis/
  figures/
    cases.json
    valid_1238_a_1.png, valid_276_a_1.png, valid_1018_a_1.png, valid_1061_a_2.png, valid_1020_a_1.png
  SUMMARY.md          ← 이 run만 존재 (아래 Part 8 참고)
```
25개 중 대부분(예: `text2ct/eval_released_n300_sp0.75_3.0/analysis/`)은 `figures/`만 있고 `SUMMARY.md`도 없음 — QC 그림만 켜져 있었다는 뜻.

### 예시 B — 기존 VLM3D 지표만 특정 profile로 재채점
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/rescore_predictions.py --metrics fid_2p5d \
  --fid-profile docker --pred-dir <run_dir>/predictions --out <run_dir>
```
(⚠ 스크립트명 정정 — 2026-07-31에 `score_pred_dir.py`→`rescore_predictions.py`로 rename됨.)
실제 예시(`report2ct_wan_repa_strel/eval_ep079_n300_sp0.75_1.3_cfg1/fid_docker/`, 실측 — 이 예시 자체는 rename 이전에 만들어진 산출물이지만 폴더 모양은 그대로 유효):
```
fid_docker/
  pred_filelist.txt
  gt_filelist.txt
  metrics.json      # {"FID_2p5D_Avg": 86.64, "fid_num_images": 100, "fid_profile": "docker", ...}
  fid.json           # {"model_name": "squeezenet1_1", "num_images": 100, "scored_stems_sha1": "5a07f98f...", ...}
  fid_features_squeezenet1_1/pred/*.mha 대응 .pt 피처들
```
**⚠ 2026-07-31부터는 `run_eval.py`(예시 C·D 아님, 일반 라이브 run)도 이 `fid_<profile>/` 구조에 직접 씀** — 더 이상 사후 재채점 전용이 아님. `<run_dir>/metrics.json`이 아니라 `<run_dir>/fid_docker/metrics.json`(또는 `fid_research/`)을 봐야 함. `analysis/`, `predictions/`, `prompts.xlsx`, `.hydra/`는 여전히 `<run_dir>` top-level 그대로.

### 예시 C — 기존 predictions에 분석 확장 전체를 사후 실행
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/score_subgroups.py \
  --pred-dir outputs/report2ct_wan_mask/eval_.../predictions \
  --out outputs/report2ct_wan_mask/eval_.../ \
  --n 300 --is-mask-model --dice --hd95 --per-sample --subgroup --subgroup-setlevel
```
이 조합은 **아직 실제 `outputs/*/analysis/`에 착지한 적 없음** — 스크래치패드 테스트로만 실행됨. 실측 산출물 모양(스크래치패드 `docker_test_per_sample.csv`, 300행 실데이터 기준):
```
analysis/
  per_sample.csv        # 65 컬럼: sample_id,target_id,scan_id,patient_id,gen_path,gt_path,condition,
                         #   cond_mask_source_id,seed,cfg_scale_text,cfg_scale_mask,run_id,label_overlap,
                         #   label_burden,label_class,labels_missing,label_<18개>,cluster_<4개>,burden_band,
                         #   source_label_<18개>,clip_t2i,clip_i2i,status,failure_reason
  subgroup/
    per_abnormality.csv, label_burden.csv, organ_cluster.csv
  setlevel/
    setlevel_fid_fvd.csv  # 실측 헤더: real_n,gen_n,real_patients,FID_2p5D_XY,FID_2p5D_YZ,FID_2p5D_XZ,
                           #   FID_2p5D_Avg,axis,below_threshold — 29행(overall/normal/disease/라벨18/밴드4/클러스터4)
  figures/
  SUMMARY.md
```

### 예시 D — 두 run 페어 비교
```bash
python scripts/compare_runs.py \
    --baseline outputs/report2ct_wan/eval_.../analysis/per_sample.csv \
    --treatment outputs/report2ct_wan_mask/eval_.../analysis/per_sample.csv \
    --out outputs/compare_wan_vs_wan_mask
```
→ `outputs/compare_wan_vs_wan_mask/per_sample_delta.csv` + `subgroup_delta_{per_abnormality,label_burden,organ_cluster}.csv`.

---

## Part 8 — 검토 중 직접 확인해볼 만한 실측 이상 징후

1. **`SUMMARY.md`가 있는데 `per_sample.csv`가 없는 run이 2개** — `report2ct_wan_mask_v2/eval_ep299_n300_sp0.75_1.3_cfgt5_cfgm0/` 와 `...cfgm1.0/`. `orchestrate.py`의 로직상(Part 4 인용) `write_summary`가 돌려면 그 전에 `per_sample.csv`가 이미 생성+저장됐어야 함 — 그런데 지금 두 run 다 `per_sample.csv`가 디스크에 없음. **⚠ 07-31 재검증: 이번 refactor로는 설명 안 됨** — `metrics_path` 변경은 헤드라인 섹션이 `metrics.json`을 찾는 위치만 바꿨을 뿐, per_sample.csv/summary 순서 자체는 그대로라 여전히 미해결 — 사후 삭제 또는 다른 실행 경로 추정. 두 `SUMMARY.md` 내용을 직접 열어보면 단서가 나올 것.
2. **upstream에 이름이 비슷한 죽은 파일 존재** (Part 1의 4번 함정) — `compute_fid_2-5d_ct.py`(진짜) vs `evaluate_fid_2p5d_ct.py`(죽은 코드, Dockerfile엔 복사되지만 `evaluation.py`가 절대 안 부름). grep으로 코드 찾을 때 헷갈리기 쉬움.
3. **`fid_profile`(메인 4대 지표용) vs `subgroup_fid_profile`(서브그룹 전용)은 완전히 독립된 두 config 키** — 하나 바꿔도 다른 하나는 안 바뀜. `subgroup_setlevel.run()`이 요구하는 `features_dir`가 메인 run이 실제로 채점한 `fid_profile`과 다르면 `FileNotFoundError`로 즉시 죽음(두 키 이름을 에러 메시지에 명시) — 조용히 틀린 값 내는 실패 모드는 아님.
4. **(07-31 신규) `CLAUDE.md`의 "eval 폴더명 규칙" 절이 낡음** — `hydra.run.dir=`로 override하라는 옛 안내가 있는데, 지금 `run_eval.py`의 `_refuse_split_run_dir`(L101-125)이 `out_dir`과 Hydra run dir이 다르면 명시적으로 거부함. `out_dir=`만 override하는 게 맞음. 코드 읽다가 이 문서 안내를 그대로 따르면 바로 막힘.
5. **(07-31 신규) `run_eval.py`로 라이브 run을 새로 돌리면 `<out_dir>/metrics.json`이 아니라 `<out_dir>/fid_<profile>/metrics.json`을 봐야 함** — 예시 A(2026-07-30 이전 산출물)처럼 옛 run들은 여전히 top-level에 `metrics.json`이 있을 수 있어서, 두 세대가 디스크에 섞여 있음. 어느 쪽인지는 `<out_dir>/fid_docker/` 또는 `fid_research/` 서브폴더 존재 여부로 구분 가능.
