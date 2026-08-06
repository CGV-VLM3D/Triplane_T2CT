# REPA runbook — report2ct_wan + frozen SPECTRE teacher

Representation alignment (U-REPA 구조) 을 `report2ct_wan` 학습에 붙이는 경로. 전부 **opt-in**이며
`repa: null`이면 loss path가 baseline과 bit-identical하다 (`tests/test_report2ct_wan_repa.py`가 고정).

- 도입 근거와 측정: [tests/repa_probe/README.md](../tests/repa_probe/README.md)
- 논문: `paper_pdf/{REPA,U-REPA,iREPA,HASTE,VideoREPA}.pdf` · 참고 구현: `third_party/repa_refs/`
- teacher: `third_party/spectre` (SHA in [submodule_pins.md](submodule_pins.md))

## 손실

```
L = L_velocity + repa_weight · ( hard_weight · L_cos + rel_weight · L_rel )      # U-REPA Eq.5
```
`repa_weight`(λ)는 `Report2CTModule`이, 괄호 안은 `RepaAligner`가 계산한다.
기본값 λ=0.5 / hard=1 / rel=3 은 U-REPA의 최적값이다.

## 0. 사전 준비 (user-owned, 1회)

```bash
# SPECTRE 코드 + 가중치 (weights are CC-BY-NC-SA)
git submodule update --init third_party/spectre
huggingface-cli download cclaess/SPECTRE \
    spectre_backbone_vit_large_patch16_128_no_vla.pt \
    spectre_backbone_vit_large_patch16_128.pt \
    spectre_combiner_feature_vit_large.pt \
    --local-dir /workspace/data/checkpoints/spectre/
pip install loralib        # spectre-fm의 선언된 의존성
```
`huggingface_hub`는 **올리지 말 것** — transformers 4.46 / diffusers 0.31이 물려 있다.
`spectre/utils/modeling.py`가 import하는 `load_state_dict_from_file`은
`src/baselines/spectre_adapter.py:_install_hf_shim()`이 채워 넣는다 (로컬 경로 로드에서는 호출되지 않음).

## 1. teacher feature precompute (GPU 1장, ~5 h)

CT를 **Wan latent와 똑같은 그리드**로 올려 SPECTRE dense token을 뽑는다. 두 teacher(SSL/VLA)와
두 해상도(32³/16³)를 한 패스에서 저장한다 — 16³는 32³의 avg-pool이라 추가 연산이 0이다.

```bash
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=2 nohup python scripts/precompute_spectre_features.py \
    --ids-file /workspace/data/ctrate_toy_v2/train/ids.json \
    --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \
    --out-root /workspace/data/report2ct_wan --num-shards 4 --shard $S \
    > logs/spectre_precompute_s${S}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
# valid_v2 ids-file + valid_fixed 로 한 번 더 (같은 --out-root)
```

| 산출물 | shape | 크기 |
|---|---|---|
| `data/report2ct_wan/spectre_{ssl,vla}_32/<id>.npy` | `(32768, 1080)` fp16 | 70.8 MB/scan |
| `data/report2ct_wan/spectre_{ssl,vla}_16/<id>.npy` | `(4096, 1080)` fp16 | 8.85 MB/scan |
| `spectre_vla_32/<id>_global.npy` | `(1080,)` | scan-level (VLA만) |

teacher 2종 × 6,304 scan ≈ **1.0 TB**. `df -h /workspace`로 먼저 확인할 것 (md0 기준, `df /` 아님).

⚠ **z end-pad는 이 스크립트가 한다.** `spectre/windowing.py`는 pad가 아니라 **center-crop**이라,
253 slice를 그대로 넣으면 `253 // 64 = 3` → 192로 잘려 양 끝 61 slice(24 %)가 조용히 사라진다.
`SpectreBackbone.window()`는 crop 배수가 아닌 입력을 받으면 자르지 않고 **예외를 던진다**.

⚠ TF32가 기본이다 (fp32 대비 3.7× 빠르고 token cosine 최저 0.99998 — `--no-tf32`로 끌 수 있음).

## 2. 학습

```bash
# Track A — from-scratch 100-epoch 파일럿 (U-REPA regime: hard + soft), ~12 h
CUDA_VISIBLE_DEVICES=3 nohup python src/train.py experiment=report2ct_wan_repa \
    trainer.max_epochs=100 callbacks.model_checkpoint.every_n_epochs=10 \
  > logs/report2ct_wan_repa_trackA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Track B — 학습된 baseline을 관계형 손실만으로 finetune (VideoREPA regime), ~1.5 h
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=report2ct_wan_repa \
    task_name=report2ct_wan_repa_ftB \
    model.init_from_ckpt=outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt \
    model.lr=2e-6 model.warmup_steps=0 \
    model.repa.hard_weight=0 model.repa.rel_loss=videorepa_l1_st \
    trainer.max_epochs=10 callbacks.model_checkpoint.every_n_epochs=2

# Track B 대조군 — 같은 step 수를 REPA 없이 (이게 없으면 A/B가 아니다)
#   위와 동일 + task_name=report2ct_wan_ft_control model.repa_weight=0
```
`scripts/launch_repa_runs.sh`가 precompute 완료를 기다렸다가 위 셋을 자동으로 띄운다.

⚠ **`model.init_from_ckpt`는 Lightning의 `ckpt_path=`가 아니다.** 후자는 optimizer와 LR 스케줄까지
복원하는데, 300 epoch을 다 돈 런의 `PolynomialLR`은 이미 ~0이라 사실상 학습이 되지 않는다.
`init_from_ckpt`는 `unet.*` + `scale_factor`만 가져오고 optimizer는 새로 만든다.

## 3. 학습 중 봐야 할 것

| metric | 기대 | 어긋나면 |
|---|---|---|
| **`train/repa_cos_gap`** | 상승 | **이게 진짜 지표다.** `repa_cos` 자체는 해부학적 위치 prior로 0.45까지 공짜로 오른다 (`tests/repa_probe/u5_overfit/`) |
| `train/repa_cos` | 상승 | 안 오르면 정렬이 안 되는 것 → probe로 복귀 |
| `train/loss_repa` | 하강 | |
| `train/loss_diff` | baseline 동일 epoch과 비슷 | 유의미하게 나쁘면 **λ 과대** — REPA의 주장은 둘 다 좋아진다는 것이지 denoising을 파는 게 아니다 |
| `train/loss` | (총합, 기존 곡선 유지) | `src/callbacks/loss_curve.py`가 그리는 값 |

## 4. 평가

기존 파이프라인 그대로. 비교 대상은 baseline `outputs/report2ct_wan/2026-07-16_3`의 **같은 epoch**.

```bash
GPU=0 EPOCHS="079 089 099" LANE=repaA bash scripts/sweep_wan_epochs.sh
```
⚠ baseline sweep의 ep099(FID 2.388)는 이웃(ep089 1.506 / ep109 1.584) 대비 명백한 outlier다.
**단일 epoch이 아니라 ep079/089/099 추세**로 비교할 것.

## 5. 노브

`configs/model/report2ct_wan_repa.yaml`. 기본값이 논문과 다른 곳은 전부 `tests/repa_probe/`의 측정 결과다.

| 노브 | 기본 | 대안 |
|---|---|---|
| `repa.projector` | `mlp` | `conv` (iREPA) — U4에서 열세였다 |
| `repa.teacher_grid` | `[16,16,16]` | `[32,32,32]` + `data.spectre_dir=…_32` — 정렬은 더 어렵고 step은 +55 % |
| `repa.hard_weight` | `1.0` | `0` = VideoREPA식 soft-only |
| `repa.rel_loss` | `urepa_l2` | `videorepa_l1_st` (in-plane/out-of-plane 분리) · `cos_ce` (ATTA 대체) |
| `repa.rel_scope` | `global` | `within_crop` — SPECTRE crop 경계를 Gram에서 제거 (공짜, U3의 22.9 % 낙차 대응) |
| `repa.spatial_norm_teacher` | `true` | `false` — cosine은 오르지만 공간 정보를 전달하지 않는다 |
| `repa_tap` | `middle_block` | `up_blocks.0` — U4에서 cosine이 가장 높았다(해상도 교란 있음) |
| `repa_stop_step` | `null` | 예 `40000` — HASTE one-shot termination |
| `data.spectre_dir` | `spectre_ssl_16` | `spectre_vla_16` — text-aligned teacher ablation (재추출 불필요) |

## 6. ATTA (HASTE)를 쓰지 않는 이유

HASTE의 attention alignment는 teacher와 student가 **같은 token 집합** 위에서 attention을 계산할 때만
성립한다. SPECTRE의 self-attention은 crop(128×128×64) 내부에서만 일어나고, 한 crop은 student
middle-stage token 2×2×2개 분량이라 대응시킬 집합이 없다. 게다가 SPECTRE는 fused SDPA를 써서
attention을 노출하지 않는다(wrapper가 `output_attentions`를 명시적으로 막는다).
관계형 손실이 이미 relational prior를 전달하므로, 분포 매칭 성격까지 원하면
`rel_loss=cos_ce`(Gram row-softmax + cross-entropy)로 충분하다.
