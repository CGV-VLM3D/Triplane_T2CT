# U9 — HASTE stage-wise termination 진단

실행: `CUDA_VISIBLE_DEVICES=1 python -m tests.repa_probe.u9_haste.run`
(2026-08-01/02, 학습이 이미 끝난 `report2ct_wan_repa_300`의 30개 체크포인트에 재학습 없이 적용)
원자료: [results/haste_rho_curve.json](results/haste_rho_curve.json) (v1, 8 volume/1 draw) ·
[results/haste_rho_curve_v2.json](results/haste_rho_curve_v2.json) (v2, 16 volume/repeat=3/common-random-numbers)

## 판정: **기각.** 300 epoch 전 구간에서 grad(L_diff)·grad(L_repa) 코사인의 후반부 반전이 안 보인다.

`report2ct_wan_repa_300`은 `repa_weight=0.5`, `hard_weight=1.0`, `repa_stop_step=null`로 **300
epoch(187.5k step) 내내 정렬 손실을 끄지 않고** 학습됐다. HASTE("REPA Works Until It Doesn't",
arXiv:2505.16792)는 이런 always-on 설정이 후반부에 오히려 FID를 깎아먹는다고 주장하고, 그 근거로
`rho_t = cos(grad_theta L_diff, grad_theta L_repa)`가 학습 진행에 따라 양수(도움) → 0(무해) →
음수(방해)로 넘어간다는 걸 gradient-angle autopsy(Sec 2.2, Fig 3/4)로 보였다. 이 probe는 우리
학습에서도 같은 전환이 있는지 재학습 없이 확인한다.

## 수치 — v1 → v2 (노이즈를 줄여 재확인)

| | v1 (8 volume, 1 draw) | v2 (16 volume, repeat=3, common seed) |
|---|---|---|
| 포인트당 표본 수 | 2 | 12 |
| ep009-149(전반) 평균 `rho_overall` | 0.069 | 0.065 |
| ep159-299(후반) 평균 `rho_overall` | 0.077 | 0.073 |
| 유일한 음수 지점 | ep159 (−0.013) | ep159 (−0.009) |
| 결론 | 하락 추세 없음, 그러나 포인트당 노이즈(±0.02~0.05)가 신호와 비슷한 크기 | 같은 결론이 표준편차로 뒷받침됨 |

**후반이 전반보다 낮아지지 않고, 오히려 근소하게 높다** — 두 버전 모두 동일한 방향. v2는 체크포인트
간 **common random numbers**(같은 (batch, timestep, repeat) 조합에 전 체크포인트가 동일한 노이즈
시드를 쓰도록 강제)로 "체크포인트 고유 신호"와 "그날의 노이즈 draw"를 분리했고, 표본을 6배로 늘려
포인트당 표준편차를 함께 기록했다 — v1에서 "노이즈에 묻혀서 하락 추세를 못 봤을 수도 있다"는 우려를
직접 검증하기 위한 재확인이었다.

## 타임스텝별로 분해하면 (v2, epoch 전체 범위)

| timestep(frac) | 의미 | 관찰 |
|---|---|---|
| t=0.9 (고노이즈) | coarse/semantic 복원 | 전 구간 확실히 양수(0.10~0.48). REPA가 의미적 정렬에서 계속 도움 |
| t=0.1 (저노이즈) | fine-detail 복원 | ep009부터 이미 −0.02~+0.04 근방. 표준편차(대부분 0.01~0.03)를 보면 **거의 모든 지점이 0과 통계적으로 구별 안 됨** |

HASTE Fig.3/4가 보고한 패턴("fine-detail 구간에서 conflict가 먼저, 뚜렷하게 음수로 온다")과 다르다.
우리 데이터는 **"fine-detail 구간이 처음부터 끝까지 약하게 무관"**에 가깝고, "초반엔 돕다가 후반에
방해로 반전"하는 시간적 구조가 없다.

## 함의 — Phase 1(bifurcation 재학습)을 보류한 이유

원래 계획은 이 진단으로 τ(전환점)를 찾아 `ckpt_path=epoch_<τ>.ckpt model.repa_weight=0`으로 이어
학습해 "REPA 계속 vs 끔"을 대조하는 것이었다. 그런데 **끊어낼 만한 "이전엔 괜찮았던" 지점 자체가
안 보이므로**, 이 개입은 근거가 약하다. `repa300`에서 관찰된 baseline 대비 근소한 FID 손해(연구
프로파일 기준)는 "후반부 gradient 반전"이 아니라 **fine-detail 타임스텝에서 시종일관 약하게 무관한
정렬 신호** 쪽으로 설명하는 게 이 데이터와 더 맞는다. 그래서:

- `repa_stop_step`은 기본값 `null`을 유지한다(README의 "이 스터디가 바꾼 기본값" 표에 기록).
- HASTE 라인의 추가 실험(Phase 1)은 착수하지 않는다.
- 이 FID 손해를 더 줄이고 싶다면, "후반에 끄기"가 아니라 **fine-detail 구간(t 낮은 곳)의 REPA
  weight를 처음부터 낮게 주는 설계**가 이 관찰과 더 부합하는 다음 후보다(미착수).

## 방법론

- **측정 파라미터**: `unet.conv_in` + `unet.down_blocks` + `unet.middle_block` — REPA gradient가
  middle_block hook을 통해 실제로 흘러가는 조상 전체. `up_blocks`/`out`은 diffusion loss만 받으므로
  애초에 비교 대상이 아니다.
- **체크포인트 재구성**: `hydra.utils.instantiate(cfg.model)`로 unet+noise_scheduler+repa를
  통째로 재생성한 뒤 `load_state_dict(strict=True)`. 기존 eval 샘플러(`_load_wan_checkpoint`)는
  `unet.*`만 읽고 `repa.*`를 버리므로 이 진단에는 쓸 수 없었다.
- **timestep 주입**: `tests/repa_probe/u4_align`과 동일한 관례로 `frac × num_train_timesteps`를
  `use_timestep_transform` 없이 직접 주입(t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}).
- **v2의 common random numbers**: `torch.manual_seed(seed)`를 `_shared_forward` 호출 직전에 걸어
  `torch.randn_like(images)` 노이즈 draw를 체크포인트 간에 완전히 동일하게 만든다 — 페어드 비교로
  노이즈-draw 분산을 제거하고 체크포인트(=epoch) 신호만 남긴다.
- **probe 데이터**: train split 앞 N개 volume 고정(v1=8, v2=16), 전 체크포인트가 동일한 probe set을
  본다.

## 한계

- probe는 여전히 표본이 작다(16 volume × repeat 3 = 체크포인트당 48회 forward). 아주 느린(수십
  epoch에 걸친) 미세한 하락 추세라면 v2로도 완전히 배제하진 못한다 — 다만 그런 추세가 있다 해도
  Phase 1이 기대한 "뚜렷한 반전 지점"은 존재하지 않는다는 결론은 유지된다.
- teacher가 SPECTRE-SSL 16³ pooled 하나뿐이다. 다른 teacher/해상도에서 같은 패턴이 재현되는지는
  확인하지 않았다.
