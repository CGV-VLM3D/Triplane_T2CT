#!/usr/bin/env bash
# report2ct_wan epoch sweep — one lane's worth of checkpoints, generate -> decode -> score.
# Modeled on scripts/run_wan_stepcal.sh's 3-stage structure. Fixed regime: n=300, n_steps=100,
# cfg=1.0, spacing 0.75/0.75/1.3. Scoring is run_eval.py, so each epoch also gets CLIP + FID +
# FVD_CTCLIP + analysis/ + QC figures + .hydra/ provenance in one pass.
#
# Usage:
#   GPU=<2|3> EPOCHS="009 049 089 ..." LANE=<tag> FID_PROFILE=research \
#     bash scripts/sweep_wan_epochs.sh
#
# Resumable: skips any epoch whose OUT dir already holds this profile's metrics.json at
# <OUT>/fid_<profile>/ (or one of the three pre-2026-07-31 layouts — see the skip check below).
# Each stage is timeout-guarded (with -k so a SIGTERM-blocking
# hang still gets SIGKILLed) and failures are logged and skipped rather than aborting the
# whole lane (a scoring run has previously hung for ~4.5h stuck on "incomplete cached pred
# features" — timeouts bound that failure mode).
#
# Timeout calibration note (fixed after code review caught a miscalibration): the original
# values were derived from a SOLO run and left almost no margin once 3 lanes actually share a
# GPU — measured live at 3-per-GPU, generate ran at 16.0s/latent (vs ~6.2s/sample assumed),
# i.e. 80min for n=300 against a 90min timeout. Recalibrated below against contended,
# real-world numbers, with TIMEOUT_SCORE deliberately kept well under the ~4.5h pathological
# hang duration so that failure mode is still caught rather than allowed to run past it.
set -u
set -f  # disable globbing — EPOCHS/$SP word-splitting is intentional, expansion is not
cd /workspace || exit 1

: "${GPU:?set GPU=0 or 1}"
: "${EPOCHS:?set EPOCHS=\"009 049 089 ...\"}"
# FID 프로파일 — 공백으로 구분한 목록. 기본은 **둘 다** 채점 (run_eval.py 의 기본과 동일).
# `docker`(squeezenet, 앞 100볼륨)와 `research`(radimagenet_resnet50, 전체 세트)는 같은
# FID_2p5D_* 키에 **비교 불가능한 스케일**로 쓴다 (한 예측 세트에서 docker 57.87 vs research 1.61).
# 예전엔 하나만 골라야 해서 이 변수를 필수로 뒀는데(고르면 나머지는 없으니), 이제 둘 다 재므로
# 그 함정이 없다 — `docker` 는 공식 컨테이너와, `research` 는 2026-07-29 이전 기록·논문 envelope
# 과 비교 가능한 유일한 값이라 어느 쪽도 다른 쪽을 대체하지 못한다.
FID_PROFILE="${FID_PROFILE:-docker research}"
LANE="${LANE:-unnamed}"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "[lane $LANE] GPU must be a plain integer, got: $GPU" >&2
  exit 2
fi

# 다른 학습 arm(REPA 등)에도 같은 regime을 쓰려고 env로 뺐다. 기본값은 baseline 그대로라
# 기존 호출은 바뀌지 않는다. REPA 체크포인트도 그대로 먹는다 — `_load_wan_checkpoint`가
# `unet.` 접두사만 골라 실으므로 `repa.*` 6개 키는 무시된다 (확인 완료).
CKPT_DIR="${CKPT_DIR:-/workspace/outputs/report2ct_wan/2026-07-16_3/checkpoints}"
# baseline sweep의 결과는 2026-07-29에 `report2ct_wan/ep_sweep/` 하위로 정리했다. 기본값이
# 옛 최상위를 가리키면 resume-skip이 전부 빗나가 30 epoch을 통째로 재생성하고, 집계기
# (aggregate_wan_epoch_sweep.py, SWEEP_DIR=.../ep_sweep)도 새 결과를 못 본다.
OUT_BASE="${OUT_BASE:-/workspace/outputs/report2ct_wan/ep_sweep}"
WANPY=/opt/conda/envs/wan/bin/python
N=300
STEPS=100
# cfg 는 env 로 뺀다. epoch sweep 은 cfg=1 로 도는데(cfg>1 은 cond/uncond 를 한 배치로 묶어
# 생성이 정확히 2배 비싸고, sweep 의 질문은 "어느 epoch 이 좋은가"이지 최종 운용점이 아니다),
# 이긴 epoch 은 제출 지점인 cfg=5 에서도 재봐야 한다 — baseline ep299 에서 cfg 1->5 는 FID
# +0.137 / T2I +16.6 을 움직이고, 그건 우리가 재려는 arm 간 격차(FID ~0.07)보다 크다.
CFG="${CFG:-1.0}"
# 디렉터리 이름 태그: "1.0" -> "cfg1", "5" -> "cfg5" (기존 dir 이름과 호환)
CFG_TAG="${CFG%.0}"
SP="0.75 0.75 1.3"

# Expected complete file sizes (fixed geometry — same for every epoch/scan), used to detect
# a timeout-triggered SIGTERM that landed mid-write (non-atomic np.save/sitk.WriteImage).
NPY_MIN_BYTES=16000000     # true size ~16,777,344
MHA_MIN_BYTES=100000000    # true size ~132,645,170

# Generous margins over the *measured, contended* per-stage time (~80min generate at 3-per-GPU,
# ~57-116min decode observed under partial contention). TIMEOUT_SCORE is kept below the known
# ~270min pathological hang so that failure mode is still caught, not out-waited.
TIMEOUT_GENERATE=9000    # 150 min
TIMEOUT_DECODE=18000     # 300 min
TIMEOUT_SCORE=12000      # 200 min
KILL_AFTER=120           # SIGKILL 120s after SIGTERM if the process ignores it

log() { echo "[lane $LANE] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

FAILED_COUNT=0

for EP in $EPOCHS; do
  CKPT="$CKPT_DIR/epoch_${EP}.ckpt"
  OUT="$OUT_BASE/eval_ep${EP}_n300_sp0.75_1.3_cfg${CFG_TAG}"

  # 프로파일별로 따로 판정한다 — 이미 있는 건 두고 **빠진 것만** 채점한다. 한 프로파일만 빠졌을 때
  # 둘 다 다시 도는 낭비(research FID 하나가 ~32분)를 피하려는 것이다.
  # 정본 경로는 `<OUT>/fid_<profile>/`. 나머지 셋은 옛 레이아웃이라 재개용으로만 인정한다:
  # `<OUT>/metrics/fid_<profile>/` (2026-07-31 이전 sweep), `<OUT>/metrics/` (2026-07-29 이전
  # rescore_predictions.py), `<OUT>/metrics.json` (2026-07-31 이전 run_eval.py) — 뒤 둘은
  # fid_profile 키가 생기기 전이라 `research` 로만 인정한다.
  MISSING=""
  for P in $FID_PROFILE; do
    if [[ -f "$OUT/fid_${P}/metrics.json" || -f "$OUT/metrics/fid_${P}/metrics.json" ]]; then
      continue
    fi
    if [[ "$P" == "research" && ( -f "$OUT/metrics/metrics.json" || -f "$OUT/metrics.json" ) ]]; then
      continue
    fi
    MISSING="${MISSING:+$MISSING }$P"
  done
  if [[ -z "$MISSING" ]]; then
    log "ep${EP}: already scored ($FID_PROFILE), skipping"
    continue
  fi
  [[ "$MISSING" != "$FID_PROFILE" ]] && log "ep${EP}: scoring only the missing profile(s): $MISSING"
  # Hydra 리스트 문법 — "docker research" -> "[docker,research]"
  PROFILE_ARG="[${MISSING// /,}]"
  if [[ ! -f "$CKPT" ]]; then
    log "FAILED ep${EP} stage checkpoint-missing: $CKPT not found"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  log "ep${EP}: generate start"
  timeout -k "$KILL_AFTER" "$TIMEOUT_GENERATE" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/generate_wan_latents.py \
    --ckpt "$CKPT" --out "$OUT" --n "$N" --n-steps "$STEPS" --cfg-scale "$CFG" --spacing $SP
  if [[ $? -ne 0 ]]; then
    log "FAILED ep${EP} stage generate"
    find "$OUT/latents" -name "*.npy" -size -${NPY_MIN_BYTES}c -delete 2>/dev/null
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  log "ep${EP}: decode start"
  timeout -k "$KILL_AFTER" "$TIMEOUT_DECODE" env CUDA_VISIBLE_DEVICES="$GPU" "$WANPY" scripts/decode_wan_latents.py \
    --latent-dir "$OUT/latents" --out "$OUT/predictions" --spacing $SP
  if [[ $? -ne 0 ]]; then
    log "FAILED ep${EP} stage decode"
    find "$OUT/predictions" -name "*.mha" -size -${MHA_MIN_BYTES}c -delete 2>/dev/null
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  log "ep${EP}: score start"
  # 채점은 `run_eval.py` 로 한다 (2026-07-31 이전엔 rescore_predictions.py 였다). model=report2ct_wan
  # 은 **pass-through 샘플러**라 UNet/VAE 를 만들지 않고 위에서 디코딩해 둔 .mha 를 그대로 채점한다
  # — 하나라도 없으면 조용히 재생성하지 않고 FileNotFoundError 로 죽는다.
  # rescore_predictions.py 와 수치는 동일함을 확인했다 (ep099 재채점, 7개 지표 전부 diff 0.0e+00).
  # 이쪽으로 옮기면 analysis/ + summary.json + QC 그림 + .hydra/ 출처가 한 패스에 따라오고,
  # `fvd_ctclip` 도 기본 on 이라 따로 얹던 패스가 없어진다.
  # ⚠ `task.metrics.fvd=false`: 리더보드 1차 지표 FVD_CTNet 은 셔틀된 체크포인트가 손상 stub 이라
  #    항상 NaN 이다 — 켜두면 시간만 쓰고 아래 NaN 검사에 걸려 멀쩡한 채점을 실패로 만든다.
  timeout -k "$KILL_AFTER" "$TIMEOUT_SCORE" env CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_eval.py \
    task=ctgen model=report2ct_wan out_dir="$OUT" \
    task.n_samples="$N" task.fid_profile="$PROFILE_ARG" task.metrics.fvd=false
  if [[ $? -ne 0 ]]; then
    log "FAILED ep${EP} stage score"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  # A failed FID/CLIP computation writes a NaN-filled metrics.json and exits 0 (CTGenEvaluator
  # returns NaN dicts rather than raising) — treat that as a failure so it gets retried later,
  # instead of the resume-skip permanently accepting broken data as "done". Checked per profile:
  # a NaN in one must not leave the OTHER looking unscored, nor be accepted as done itself.
  NAN_FOUND=""
  for P in $MISSING; do
    if grep -qi "nan" "$OUT/fid_${P}/metrics.json" 2>/dev/null; then
      rm -f "$OUT/fid_${P}/metrics.json"
      NAN_FOUND="${NAN_FOUND:+$NAN_FOUND }$P"
    fi
  done
  if [[ -n "$NAN_FOUND" ]]; then
    log "FAILED ep${EP} stage score-nan (NaN in profile(s): $NAN_FOUND)"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  log "ep${EP}: DONE ($MISSING)"
  for P in $MISSING; do
    echo "--- fid_$P ---"
    cat "$OUT/fid_${P}/metrics.json" 2>/dev/null
  done
done

log "lane $LANE ALL DONE (failed: $FAILED_COUNT)"
exit $(( FAILED_COUNT > 0 ? 1 : 0 ))
