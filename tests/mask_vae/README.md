# 마스크 VAE (mask VAE) — 자기완결 실험

TS `ts_seg` 마스크를 압축하는 3D VAE를 학습한다. 학습된 인코더로 마스크를 `(4,120,120,64)` seg latent으로
굽고(Stage 2), 이후 Report2CT에서 이미지 latent과 **채널 concat**해 **MASK+TEXT→CT** 오라클 ablation에 쓴다
(Stage 3, 별도).

> 격리 원칙: `src/`·`configs/`를 건드리지 않는 self-contained 실험. 검증된 **가중치만** 본 프로젝트로 가져간다.
> 계획: `/root/.claude/plans/velvety-conjuring-badger.md`.

## 파일
| 파일 | 역할 |
|---|---|
| `preprocess.py` | ts_seg → RAS + Resize(480,480,256, **nearest**) + 64³ 랜덤크롭; **uint8 캐시**; embed/onehot/raw 입력 변환 |
| `model.py` | `build_mask_vae` — 번들 `AutoencoderKlMaisi`에서 in/out/latent/num_splits/norm_float16 override |
| `dataset.py` | train_fixed/valid_fixed 마스크 DataLoader (**no_chest+error 제외** → train 46,393 / valid 3,001) |
| `train.py` | DiceCE+KL 학습 루프 + val + results(metrics·loss curve·recon vis) + resume |
| `viz.py` | GT vs 재구성 라벨맵 슬라이스 비교 이미지 |

## 파이프라인
```
ts_seg .nii.gz (라벨 0~117, 스캔당 1개)
  └ preprocess: RAS → Resize(480,480,256, nearest) → [uint8 캐시] → 64³ 랜덤크롭
  └ 입력: 라벨 → Embedding(118, 16) → 16채널   (embed 채택)
  └ VAE(AutoencoderKlMaisi in=16 / out=118 / latent=4, 4× 다운샘플)
        recon(B,118,64³), z_mu/z_sigma(B,4,16³)
  └ loss: DiceCELoss(softmax, to_onehot_y, include_background=False) + 1e-7·KL
  └ [Stage 2 precompute] 전체 480³ 인코딩 → (4,120,120,64) seg latent  ← 이미지 latent과 정렬 (TODO)
```

## 확정된 셋팅 (실험으로 결정)
- **입력 = embed** (learnable `Embedding(118,16)`). raw는 ordinality로 실패(fg_dice~0.01), onehot(~0.4)보다 embed(~0.82)이 최고.
- **lr 1e-4 / kl_weight 1e-7** (MAISI 기본). warmup 있으면 안정 + 빠름.
- **bf16 autocast**(GradScaler 없음), **warmup 150 step**, **grad clip norm 10**, **num_splits=1**, `norm_float16=False`.
- **DiceCE `include_background=False`** (배경 지배적 → Dice는 장기 집중, 배경은 CE).
- **패치 64³ 먼저 → 128³ 이어학습**(MAISI progressive, `--resume`).

## 실행
```bash
# 학습 (embed·MAISI 기본이 default). 결과 → results/<exp>/
CUDA_VISIBLE_DEVICES=1 python -m tests.mask_vae.train --exp-name <exp> \
  --limit 300 --steps 2000 --batch-size 8 --cache-rate 1.0 --num-workers 32 \
  --val-limit 16 --vis-samples 6

# 128³ 이어학습 (progressive)
CUDA_VISIBLE_DEVICES=1 python -m tests.mask_vae.train --exp-name <exp>_128 \
  --patch 128 --batch-size 2 --resume results/<exp>/best.pt ...
```
결과: `results/<exp>/` → `best.pt`(val 기준) · `last.pt` · `metrics.json` · `loss_curve.png`(epoch축) · `vis/`(GT vs recon).

## 검증된 것
- overfit(8마스크) 수렴 + generalization(300마스크 → **held-out val_fg 0.82**) 확인 → 파이프라인·안정화 OK.
- 병목은 **GPU 계산(~1.5s/step @64³ batch8)**, I/O 아님(캐시 시). num_splits는 ~17%만 영향.

## 다음 단계
- **Step 4b — 47k 실학습**: batch 키우기(GPU 여유), 47k는 RAM 캐시 불가 → **PersistentDataset(디스크) 또는 resize된 마스크 precompute**로 I/O 해결. 64³ → 128³ progressive.
- **Step 5 — `precompute.py`**: 학습 VAE로 전체 480³ 인코딩 → `{id}_maskemb.nii.gz` `(120,120,64,4)`, 이미지 latent과 격자·affine 정렬 확인.
- **Stage 3**: Report2CT에서 seg latent concat (별도 계획, `src/`·`configs/` 수정).
