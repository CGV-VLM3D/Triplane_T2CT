# Triplane AE — 연구 요약

_마지막 업데이트: 2026-05-14 02:20_

## 전체 경향 및 발견

_이 섹션은 매 호출마다 처음부터 다시 작성됩니다. 간결하고 최신 상태로 유지._

- **현재 최고 성능**: `trial2` — PSNR=16.57 dB (Δ=−14.37 vs upper bound), SSIM=0.497 (Δ=−0.222). 전체 validation 지표를 산출한 첫 end-to-end 성공 실행.
- **상한선 (MAISI VAE)**: PSNR=30.94 ± 2.97 dB, SSIM=0.7195 ± 0.1084 (CT-RATE val 1000 vol round-trip; `results/upper_bound.json`).
- **핵심 학습**:
  - trial1을 죽인 원인은 MAISI 디코더 warmup이 아니라 **encoder attention의 OOM**. 세 번의 trial1 시도(bs=1, bs=8, `expandable_segments`+`compile_maisi=false`로 bs=1) 모두 `TriplaneEncoder._f_psi`의 `TransformerEncoderLayer._ff_block`에서 동일하게 터짐. 핵심 메모리 driver는 batch size가 아니라 **시퀀스 길이** — bs=1에서 train forward 자체가 OOM이면 `sw_batch_size` 같은 knob은 무의미.
  - **트랜스포머 앞에 3D Conv patchify(p=4)를 두는 것이 해결책.** trial2는 `emb_dim=256, n_layers=4, n_heads=8` 구성으로 단일 A6000에서 30 it/s, peak GPU 메모리 단 2682 MiB로 동작 — trial1 대비 대략 30× 메모리 절감.
  - MAISI VAE와의 격차는 여전히 큼 (~14 dB PSNR, ~0.22 SSIM). epoch 5 기준 latent 공간 PSNR(26.26)은 image 공간 PSNR(16.97)보다 훨씬 높아, 병목은 모델의 표현력 자체가 아니라 **frozen MAISI 디코더가 받아들일 수 있는 latent를 인코더가 만들어낼 수 있는가**에 있음.
  - Identity sanity check(`trial1_identity`)는 평가 파이프라인의 정합성을 확인 — PSNR=∞, SSIM=1.0. `IdentityAE`가 `mu`를 직접 반환해서 L1 loss가 no-grad leaf zero가 되는 문제를 막기 위해 `_dummy` 파라미터로 loss를 anchor해야 했음.
- **남은 질문 / 다음 실험**:
  - 이제 메모리 여유가 ~91 GiB 있으니 더 큰 patchify 임베딩(`emb_dim=512`, `n_layers=6`)으로 끌어올릴 수 있을까?
  - SDPA나 grad checkpointing을 도입하면 conv patchify 없이도 trial1 아키텍처를 더 큰 capacity로 다시 시도할 수 있을까?
  - image vs latent PSNR의 14 dB 격차는 (a) frozen MAISI 디코더가 off-manifold latent에 민감해서인지, (b) triplane 디코더의 재구성 오차 때문인지 — recon-validator로 분리해서 진단 필요.

## 실험 로그 (최신순)

### trial2 — 2026-05-13 — 성공

- **Run**: https://wandb.ai/jasonna24-/triplane-ae/runs/wfn1avvj
- **지표 (val, 현재 최고)**: PSNR=16.57 dB (Δ=−14.37), SSIM=0.497 (Δ=−0.222). epoch 5 latent: `latent_psnr=26.26`, `latent_l1=0.5954`, `image_psnr_3d=16.97`, `image_ssim_3d=0.487`.
- **설정 요점**: 3D Conv patchify `p=4`, `emb_dim=256`, `n_layers=4`, `n_heads=8`. Train `bs=1` on 1× A6000. ~30 it/s. Peak GPU 메모리 2682 MiB (극도로 낮음). 50 epoch ETA ~3h.
- **요점**: 트랜스포머 앞 conv patchify가 시퀀스 길이를 줄여 trial1의 OOM을 제거(≈30× 메모리 절감). epoch 6+ 까지 학습 안정. 남은 격차의 대부분은 image 공간 — latent 자체는 잘 복원됨.
- **그림**: `figs/trial2/` (`train_loss.png`, `val_loss.png`, `val_psnr.png`, `val_ssim.png`).

### trial1 — 2026-05-13 — OOM 실패 (3회 시도)

- **Runs**: https://wandb.ai/jasonna24-/triplane-ae/runs/098m5gpm, https://wandb.ai/jasonna-/triplane-ae/runs/w2qf3xex, https://wandb.ai/jasonna24-/triplane-ae/runs/n7t4ijw8
- **지표**: N/A (첫 forward pass에서 모두 실패).
- **설정 요점**: Flat tokenization — XY plane = 120×120 = 14400 tokens at `emb_dim=512`. Conv patchify 없음. bs=1, bs=8, 그리고 `expandable_segments=True` + `compile_maisi=false`로 bs=1 — 세 번 모두 같은 지점에서 실패.
- **요점**: `TriplaneEncoder._f_psi`의 `TransformerEncoderLayer._ff_block`에서 OOM — 88–94 GiB가 이미 사용된 상태에서 `linear1`에 ~7 GiB 추가 할당 실패. bs=1의 train forward 단계에서 발생하므로 `sw_batch_size`나 OOM 재시도(batch halving)로는 해결 불가. 아키텍처 수준의 수정(conv patchify, SDPA, grad checkpointing)이 필요 — trial2에서 채택됨.
- **그림**: 없음 (완료된 epoch 없음).

### trial1_identity — 2026-05-12 — 성공 (sanity check)

- **Run**: `wandb/offline-run-20260512_172858-v5z0f7kv` (offline; API key 없음)
- **지표**: PSNR=∞, SSIM=1.0, loss=0.00 (100 step 동안). 구조상 자명.
- **설정 요점**: `IdentityAE` (encoder/decoder 모두 pass-through). bs=1.
- **요점**: 평가 파이프라인 정상. `IdentityAE`가 `mu`를 직접 반환해서 L1 loss가 no-grad leaf zero가 되는 문제를 막기 위해 `_dummy` 파라미터로 loss를 anchor 필요. 체크포인트 3.4 KB at `checkpoints/trial1_identity/epoch_0001.pt`.
- **그림**: 없음.
