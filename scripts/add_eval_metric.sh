#!/usr/bin/env bash
# 이미 채점이 끝난 ctgen eval 디렉터리에 **지표를 추가로** 매긴다 — 재생성 없음.
#
# `predictions/` 를 그대로 재사용하고, 결과는 `rescore_predictions.py` 가 강제하는
# `fid_<profile>/` 하위에 쓴다. 그 런의 본 채점(`metrics.json` / `metrics/`)은 건드리지 않는다.
#
# 산출 경로는 SUBDIR 유무로 갈린다:
#   SUBDIR 없음 -> <OUT>/fid_<profile>/
#   SUBDIR 있음 -> <OUT>/<SUBDIR>/fid_<profile>/
# 프로파일 이름만으로는 **metric set 이 구분되지 않으므로**, 같은 프로파일로 서로 다른 지표를
# 두 번 매길 때는 반드시 SUBDIR 로 갈라야 한다 (안 그러면 뒤엣것이 앞엣것을 덮는다).
# 예: research FID 는 `fid_research/`, research 프로파일의 FVD 는 `fvd/fid_research/`.
#
# 사용:
#   # docker 프로파일 FID 추가 (research 수치는 그대로 둔 채) -> <OUT>/fid_docker/
#   GPU=2 METRICS=fid_2p5d PROFILE=docker \
#     bash scripts/add_eval_metric.sh outputs/<arm>/eval_ep*_cfg1
#
#   # FVD_CTCLIP 추가 -> <OUT>/fvd/fid_research/
#   GPU=2 METRICS=fvd_ctclip PROFILE=research SUBDIR=fvd \
#     bash scripts/add_eval_metric.sh outputs/<arm>/eval_ep*_cfg1
#
# ⚠ FVD 주의: 리더보드 1차 지표인 **`FVD_CTNet` 은 로컬에서 계산할 수 없다** — 셔틀된 CT-Net
#    체크포인트가 손상 stub 이라 항상 NaN 이다. `fvd_ctclip` 은 CT-CLIP zero-shot feature 로
#    만든 **우리 로컬 proxy** 이고, arm 간 상대 비교에만 쓸 수 있다 (리더보드 값과 비교 불가).
#
# 재개 가능: 산출 `metrics.json` 이 이미 있으면 건너뛴다.
set -u
cd /workspace || exit 1
: "${GPU:?set GPU=<n>}"
: "${METRICS:?set METRICS=\"fvd_ctclip\" | \"fid_2p5d\" | \"clip_score fid_2p5d\" ...}"
SUBDIR="${SUBDIR:-}"
PROFILE="${PROFILE:-research}"
N="${N:-300}"
TIMEOUT="${TIMEOUT:-12000}"
KILL_AFTER="${KILL_AFTER:-120}"

if [ "$SUBDIR" = "metrics" ]; then
  echo "SUBDIR=metrics 는 sweep 의 본 채점을 덮어쓴다 — 다른 이름을 쓸 것" >&2; exit 2
fi

ok=0; skip=0; miss=0; fail=0
for OUT in "$@"; do
  [ -d "$OUT" ] || continue
  DEST="$OUT${SUBDIR:+/$SUBDIR}"
  if [ -f "$DEST/fid_$PROFILE/metrics.json" ]; then
    echo "[skip] $OUT ($DEST/fid_$PROFILE exists)"; skip=$((skip + 1)); continue
  fi
  n_pred=$(find "$OUT/predictions" -name '*.mha' 2>/dev/null | wc -l)
  if [ "$n_pred" -lt "$N" ]; then
    echo "[MISS] $OUT — predictions $n_pred < $N (decode 먼저)"; miss=$((miss + 1)); continue
  fi
  echo "[run ] $OUT -> $DEST/fid_$PROFILE ($METRICS)"
  # shellcheck disable=SC2086  # METRICS 의 단어 분리는 의도적
  timeout -k "$KILL_AFTER" "$TIMEOUT" env CUDA_VISIBLE_DEVICES="$GPU" \
    python scripts/rescore_predictions.py --pred-dir "$OUT/predictions" --out "$DEST" \
      --n "$N" --metrics $METRICS --fid-profile "$PROFILE"
  if [ $? -ne 0 ]; then echo "[FAIL] $OUT"; fail=$((fail + 1)); else ok=$((ok + 1)); fi
done
echo "[done] scored=$ok skipped=$skip incomplete=$miss failed=$fail"
