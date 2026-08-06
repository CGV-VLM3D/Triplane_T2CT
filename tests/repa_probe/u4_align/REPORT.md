# U4 — student↔teacher 정렬 가능성 (REPA 도입 go/no-go)

실행: `CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u4_align.run --n-volumes 48 --steps 1500`
(2026-07-28, valid_v2 48 스캔 = train 32 / held-out 16, timestep 5지점, UNet **완전 frozen**)
원자료: [results/alignability.json](results/alignability.json) (steps=1500) ·
[results/alignability_steps300.json](results/alignability_steps300.json) (수렴 확인용)

## 판정: **GO.** frozen UNet + projector만으로 held-out cosine **+0.56**에 도달한다.

참고로 U-REPA는 SiT↓가 **학습을 20만 스텝 돌린 뒤** token-wise cosine 0.60에서 정체한다고 보고했다
(Fig 3). 우리는 UNet을 전혀 건드리지 않고 projector만 1500스텝 맞춰 0.56에 닿는다 — 정렬할 표현이
이미 거기 있다는 뜻이고, 학습이 그걸 더 끌어올릴 여지가 있다는 뜻이다.

## 수치 (held-out 16 볼륨 × 5 timestep, steps=1500)

| ckpt | tap | projector | teacher grid | fit | teacher norm | **cos↑** | rel↓ | proj params |
|---|---|---|---|---|---|---|---|---|
| ep299 | middle | **mlp** | 16³ | cos | on | **+0.5625** | 0.0202 | 2.9M |
| ep299 | middle | conv | 16³ | cos | on | +0.5303 | 0.0373 | 14.9M |
| ep299 | up0 | conv | 16³ | cos | on | +0.5782 | 0.0265 | 14.9M |
| ep299 | down3 | conv | 16³ | cos | on | +0.5295 | 0.0417 | 14.9M |
| ep299 | middle | mlp | 32³ | cos | on | +0.4714 | 0.0283 | 2.9M |
| ep299 | middle | conv | 32³ | cos | on | +0.4460 | 0.0463 | 14.9M |
| ep009 | middle | conv | 16³ | cos | on | +0.5287 | 0.0340 | 14.9M |
| ep099 | middle | conv | 16³ | cos | on | +0.5366 | 0.0359 | 14.9M |
| ep299 | middle | conv | 16³ | **rel** | on | **−0.0004** | 0.0189 | 14.9M |
| ep009 | middle | conv | 16³ | **rel** | on | +0.0020 | 0.0171 | 14.9M |
| ep299 | middle | conv | 16³ | cos | **off** | +0.6969 | 0.0476 | 14.9M |

## 결정 4개

### 1. projector = **MLP** (iREPA의 conv 주장은 여기서 재현되지 않았다)
MLP가 cosine(0.5625 vs 0.5303)과 관계형 손실(0.0202 vs 0.0373) 양쪽에서 이기고, **파라미터는 1/5**
(2.9M vs 14.9M). 300→1500 스텝에서 순위가 유지되므로 미학습 아티팩트가 아니다.
⚠ 단서: iREPA의 주장은 **end-to-end 학습 수렴 속도**에 관한 것이고 이 probe가 재는 건 **도달 가능한
정렬량**이다. 다른 것을 재고 있으므로 "iREPA가 틀렸다"가 아니라 "우리 설정에서 conv를 기본값으로 삼을
근거가 없다"가 맞는 결론. `projector: conv`는 ablation 노브로 남긴다.

### 2. teacher grid = **16³** (32³보다 정렬이 **쉽다**)
32³에서 cosine이 오히려 떨어진다(0.4714 vs 0.5625). 당연한 결과다 — student tap은 8³이라 8배 많은
서로 다른 타깃을 설명할 방법이 없다. **"pooling하면 디테일이 날아가지 않나"라는 우려에 대한 답이
여기 있다: 그 디테일은 애초에 8³ tap이 표현할 수 없다.** I/O가 8배 싼 것은 덤.
32³ arm은 그대로 남아 있으므로(같은 precompute) 학습 단계에서 재확인 가능.

### 3. tap = **middle_block** (up0가 더 높지만 해상도 교란이 있다)
`up_blocks.0`가 cosine 0.5782로 가장 높다. 그러나 up0는 native 해상도가 이미 16³라 **업샘플이 전혀
필요 없다** — 다른 arm과 공평한 비교가 아니다. 게다가 U-REPA가 `middle`을 고른 근거는 도달 cosine이
아니라 **FID**다. 기본값은 `middle_block`으로 두고 `up_blocks.0`은 유망한 ablation으로 기록한다.

### 4. teacher spatial norm = **ON** (cosine이 낮아지는 게 정상이다)
norm을 끄면 cosine이 0.5303 → **0.6969**로 크게 오르지만 관계형 손실은 0.0373 → 0.0476으로 **나빠진다**.
U3에서 본 대로 raw teacher는 강한 global 성분을 공유하므로, projector가 평균 방향만 맞춰도 높은 cosine이
공짜로 나온다 — 공간 정보를 전혀 전달하지 않는 "쉬운 cosine"이다.
→ **arm 간 model selection에 raw cosine을 쓰면 안 된다.** norm을 켠 상태의 cosine이 힘들게 얻은 값이다.

## hard vs soft — 이 probe가 말할 수 있는 것과 없는 것

`fit=rel`(관계형만으로 적합)은 관계형 손실 자체는 잘 내린다(0.0189 < cos-fit의 0.0373). 그런데
**token-wise cosine은 정확히 0에 머문다**(−0.0004 / +0.0020). 두 목적함수가 실제로 다른 것을 재고
있다는 직접 증거이고, `tests/test_repa_aligner.py`의 회전 불변성 테스트가 설명하는 그대로다 —
관계형 손실은 표현을 회전 자유도까지만 결정한다.

→ **from-scratch(Track A)에서 hard 항을 빼면 안 된다.** U-REPA Table 15(λ=0 → FID 5.72→10.91)와 같은 방향.

⚠ **이 probe가 말할 수 없는 것**: ep009 / ep099 / ep299의 도달 cosine이 0.529 / 0.537 / 0.530으로
사실상 같다. 즉 "사전학습이 진행될수록 hard 정렬이 어려워진다"는 가설은 **지지되지 않는다**.
VideoREPA의 주장은 "hard 정렬이 **finetuning 중에 기존 표현을 파괴한다**"는 *학습 동역학*에 관한 것인데,
UNet을 frozen한 이 probe는 그걸 잴 수 없다. Track B(soft-only finetune)의 근거는 여전히 **가설**이며
학습 arm으로만 가릴 수 있다.

## 방법론

- projector는 **train 32 볼륨에서만** 적합하고 held-out 16 볼륨에서 보고한다. UNet은 전 구간 frozen이므로
  이 수치는 "정렬이 가능한 표현인가"이지 "학습이 되는가"가 아니다.
- timestep은 RFlow 시간축의 0.1/0.3/0.5/0.7/0.9 다섯 지점, 볼륨마다 노이즈 seed 고정(arm 간 비교 가능).
- steps=300과 1500을 모두 남겼다. **관계형 적합은 300스텝에서 미학습이었다**(rel 0.0525 → 1500에서 0.0189)
  — 300스텝 결과만 봤다면 "관계형 손실은 최적화도 안 된다"고 잘못 결론냈을 것이다. cosine 쪽 순위는 두
  설정에서 동일하다.
