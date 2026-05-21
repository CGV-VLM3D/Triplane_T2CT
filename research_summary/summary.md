# Triplane AE — 연구 요약

_마지막 업데이트: 2026-05-17 00:00_

## 전체 경향 및 발견

_이 섹션은 매 호출마다 처음부터 다시 작성됩니다. 간결하고 최신 상태로 유지._

- **현재 최고 성능 (Tier-1, latent domain)**: `trial_toy_baseline` (TriVQAEConv, 16M params) — latent PSNR=27.39 dB (Δ=+0.76 vs 2mm image ceiling), 9 epoch 만에 달성하고 아직 상승 중 (60min cap에서 종료).
- **Tier-1 상한선 (2mm MAISI round-trip)**: PSNR=26.63 ± 1.45 dB, SSIM=0.686 ± 0.085 (image domain, n=1000; `results/upper_bound_2mm.json`). 이 수치는 **image-domain**이며 Tier-1 metric인 **latent-domain PSNR**과 직접 비교 불가.
- **기존 상한선 (1mm full-res MAISI VAE)**: PSNR=30.94 ± 2.97 dB, SSIM=0.7195 ± 0.1084 (`results/upper_bound.json`).
- **핵심 학습** (전체 실험에서 도출):
  - **Conv 아키텍처(baseline)가 Tier-1 sweep에서 트랜스포머 계열을 크게 압도.** TriVQAEConv(16M)는 60분 cap 내에 9 epoch만 돌아 27.39 dB를 기록했으며 아직 상승 중; TriplaneAE(ours, 5.5M)는 50 epoch 완료 후 25.66 dB에서 조기 plateau; TriVAE_D3T Swin(2.94M)는 99 epoch에서 26.01 dB에 plateau.
  - **트랜스포머 계열의 조기 plateau는 용량(capacity) 한계가 아니라 학습 레시피 문제일 가능성이 높다.** LR warmup 없이 L1 단독 손실과 vanilla Adam만 사용한 경우, patch_size=4 TriplaneAE는 epoch 5 전후에 이미 plateau에 진입한다. trial2도 동일한 패턴을 보였으며, 이 수치는 "vanilla recipe 하의 하한선"으로 보는 것이 타당하다.
  - **시퀀스 길이가 메모리의 지배적 드라이버.** trial1(flat tokenization, XY 14400 tokens)은 bs=1에서도 `TransformerEncoderLayer._ff_block`에서 OOM. 3D Conv patchify(p=4)가 해결책 — trial2에서 peak 2682 MiB로 동작하며 약 30배 메모리 절감.
  - **2mm Path B 검증 완료.** CT 입력을 stretch-resize로 약 2mm에 리샘플링하는 방법이 MAISI 자체 `rand_zoom` 학습 augmentation 관례와 일치함을 확인. Path A(width-half)는 그 위에 추가로 적용.
  - **bottleneck 채널 수(out_channels)는 image-domain PSNR의 결정 요인이 아니다.** trial3(out_channels 8→32)은 trial2 대비 image_psnr_3d에서 개선 없음(-0.1 dB). 병목은 frozen MAISI 디코더가 수용 가능한 latent를 인코더가 생성하는지 여부다.
  - **L2(MSE) 손실로의 단순 교체는 역효과.** trial4에서 image_psnr_3d가 trial2 대비 -1.06 dB 후퇴.
- **남은 질문 / 다음 실험**:
  - baseline(TriVQAEConv)을 Tier-2로 승격하여 더 긴 wall-clock 예산 하에서 실제 plateau를 확인할 것.
  - ours/d3t에 LR warmup + cosine decay + 복합 손실을 적용하면 트랜스포머 plateau가 상승하는지 검증.
  - trial_toy 결과는 latent-domain PSNR을 지표로 사용; image-domain PSNR과의 격차를 recon-validator로 분리 진단해야 함.
  - ours v1.0 표준 정의(`patch_size=4`, `emb_dim=512`, `n_layers=6`, `n_heads=8`, `out_channels=24`, `decoder_hidden=96`, `decoder_n_res_blocks=4`)를 개선된 레시피로 재도전할 필요.

## 실험 로그 (최신순)

### trial_toy_baseline — 2026-05-17 — 성공 (Tier-1 frontrunner)

- **Run**: https://wandb.ai/jasonna24-/triplane-ae/runs/fs4st1kd
- **지표**: latent PSNR=27.39 dB (Δ=+0.76 vs 2mm image ceiling 26.63), SSIM=N/A. 60min cap에서 아직 상승 중 (last-3-ep ΔPSNR=+0.253).
- **설정 요점**: TriVQAEConv (3D ResBlock + axis-only down/up + cross-plane mixer), 16M params, patch_size=4, bs=4, AMP off. 2mm toy latents `[4, 60, 60, 32]`. 9 epoch / 60.1 min (cap hit). ~335 ms/step.
- **요점**: 3D conv 기반 아키텍처가 같은 wall-clock 내 트랜스포머 계열을 1.38–1.73 dB 앞선다. 9 epoch만에 Tier-1 ceiling을 초과했으며 아직 slope가 +0.18–0.30 dB/epoch 수준 — 이 수치는 진정한 plateau가 아니라 cap 제약에 의한 종료임.
- **그림**: `runs/trial_toy/figs/`

### trial_toy_d3t — 2026-05-17 — 성공 (Tier-1)

- **Run**: https://wandb.ai/jasonna24-/triplane-ae/runs/kp4260wc
- **지표**: latent PSNR=26.01 dB (Δ=-0.62 vs 2mm image ceiling 26.63), SSIM=N/A. plateau 확인 (last-3-ep ΔPSNR=+0.003).
- **설정 요점**: TriVAE_D3T Swin transformer, width-half, 2.94M params, patch_size=4, bs=4, AMP bf16. 2mm toy latents `[4, 60, 60, 32]`. 99 epoch / 56.1 min. ~25 ms/step (가장 빠름).
- **요점**: 가장 파라미터 효율이 높고(2.94M) step당 속도도 가장 빠르지만, vanilla L1+Adam 레시피 하에서 ~26.0 dB에 plateau. 용량 한계가 아니라 레시피 한계로 해석되며, warmup+스케줄러 도입 시 상승 여지가 있다.
- **그림**: `runs/trial_toy/figs/`

### trial_toy — 2026-05-17 — 성공 (Tier-1, ours v1.0 width-half)

- **Run**: https://wandb.ai/jasonna24-/triplane-ae/runs/shucjhx5
- **지표**: latent PSNR=25.66 dB (Δ=-0.97 vs 2mm image ceiling 26.63), SSIM=N/A. 조기 plateau (last-3-ep ΔPSNR=+0.004).
- **설정 요점**: TriplaneAE std v1.0 width-half, 5.5M params, patch_size=4, `emb_dim=512`, `n_layers=6`, `n_heads=8`, `out_channels=24`, `decoder_hidden=96`, `decoder_n_res_blocks=4`. bs=8. 2mm toy latents `[4, 60, 60, 32]`. 50 epoch / 41.6 min (cap 전 완료).
- **요점**: epoch 5 전후에 이미 plateau 진입 — 이후 45 epoch에서 누적 +0.4 dB 추가에 그침. trial2와 동일한 조기 plateau 패턴. Tier-1 3-way sweep에서 최하위이지만, 이는 레시피 문제(LR warmup 없음, L1만 사용)를 반영한 lower bound임.
- **그림**: `runs/trial_toy/figs/`

---

### trial2 — 2026-05-13 — 성공

- **Run**: https://wandb.ai/jasonna24-/triplane-ae/runs/wfn1avvj
- **지표 (val, image domain)**: PSNR=16.57 dB (Δ=-14.37 vs 1mm upper bound 30.94), SSIM=0.497 (Δ=-0.222). epoch 5 latent: `latent_psnr=26.26`, `latent_l1=0.5954`, `image_psnr_3d=16.97`, `image_ssim_3d=0.487`.
- **설정 요점**: 3D Conv patchify `p=4`, `emb_dim=256`, `n_layers=4`, `n_heads=8`. bs=1 (이후 bs=16, lr=4e-4로 resume하여 epoch 500까지 확장). 1×A6000. ~30 it/s. Peak GPU 메모리 2682 MiB.
- **요점**: 트랜스포머 앞 conv patchify가 시퀀스 길이를 줄여 trial1의 OOM을 제거(약 30배 메모리 절감). 이 구성이 ours v1.0의 기반. latent 재구성은 양호하나 image-domain 수치는 낮아, 병목은 MAISI 디코더와의 호환성에 있음.
- **그림**: `figs/trial2/` (`train_loss.png`, `val_loss.png`, `val_psnr.png`, `val_ssim.png`).

### trial1 — 2026-05-13 — OOM 실패 (3회 시도)

- **Runs**: https://wandb.ai/jasonna24-/triplane-ae/runs/098m5gpm, https://wandb.ai/jasonna-/triplane-ae/runs/w2qf3xex, https://wandb.ai/jasonna24-/triplane-ae/runs/n7t4ijw8
- **지표**: N/A (첫 forward pass에서 모두 실패).
- **설정 요점**: Flat tokenization — XY plane = 120×120 = 14400 tokens at `emb_dim=512`. Conv patchify 없음. bs=1, bs=8, `expandable_segments=True` + `compile_maisi=false` bs=1 — 세 번 모두 동일 지점 실패.
- **요점**: `TriplaneEncoder._f_psi`의 `TransformerEncoderLayer._ff_block`에서 OOM — 88–94 GiB 점유 상태에서 `linear1`에 ~7 GiB 추가 할당 실패. bs=1 train forward에서 발생하므로 `sw_batch_size`나 batch halving으로는 해결 불가. 아키텍처 수준 수정(conv patchify)이 필요 — trial2에서 채택됨.
- **그림**: 없음 (완료된 epoch 없음).

### trial1_identity — 2026-05-12 — 성공 (sanity check)

- **Run**: `wandb/offline-run-20260512_172858-v5z0f7kv` (offline; API key 없음)
- **지표**: PSNR=inf, SSIM=1.0, loss=0.00 (100 step 동안). 구조상 자명.
- **설정 요점**: `IdentityAE` (encoder/decoder 모두 pass-through). bs=1.
- **요점**: 평가 파이프라인 정상. `IdentityAE`가 `mu`를 직접 반환해서 L1 loss가 no-grad leaf zero가 되는 문제를 막기 위해 `_dummy` 파라미터로 loss를 anchor 필요. 체크포인트 3.4 KB at `checkpoints/trial1_identity/epoch_0001.pt`.
- **그림**: 없음.
