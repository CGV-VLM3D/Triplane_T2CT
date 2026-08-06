# U8 — 표현의 의미적 품질: REPA 논문의 linear-probe / k-NN / CKA 분석 (CT-native)

REPA §3의 분석을 그대로 옮기되 ImageNet 라벨 대신 CT-RATE의 **18개 abnormality 라벨**을 쓴다
(다중 라벨이라 정확도 대신 **평균 AUROC**).

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.repa_probe.u8_semantic.run --n-volumes 300 --with-context \
  --ckpts base_ep009=outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_009.ckpt … \
          repa_ep099=outputs/report2ct_wan_repa/2026-07-29/checkpoints/epoch_099.ckpt
```
원자료: [results/semantic.json](results/semantic.json) · 그림: [figs/U8_semantic_by_stage.png](figs/U8_semantic_by_stage.png)

## 측정 설정

| 항목 | 값 |
|---|---|
| 볼륨 | `datalist_wan_2560.json`의 **validation 앞 300개** (valid_v2) |
| 분할 | **train 200 / test 100** — datalist 순서 그대로 앞 2/3 (셔플 없음, 아래 ⚠) |
| 체크포인트 | baseline 7점(ep009·029·079·119·179·239·299) + REPA 4점(ep009·029·059·099) = **11** |
| 조건 | context **OFF**(헤드라인) + **ON**(`--with-context`) 두 모드 = 22 pass |
| timestep | RFlow 시간축 **0.5** 고정 (`TIMESTEP_FRAC`) |
| 노이즈 | 볼륨마다 `torch.manual_seed(2000 + batch_idx)` — **체크포인트 간 동일 노이즈** |
| tap | 9곳 (`conv_in` … `up_blocks.2`), 각 tap 출력을 공간축 global mean-pool → `(B, C)` |
| teacher | precompute된 16³ dense 격자를 mean-pool (`spectre_ssl` / `spectre_vla`) + `spectre_vla` scan-level CLS |
| probe | linear = 라벨별 로지스틱 회귀(`C=1.0`, StandardScaler), k-NN = cosine top-20 이웃의 라벨 평균 |
| CKA | 선형 CKA, **300볼륨 전체** 사용 (분할과 무관) |
| GPU / 소요 | GPU 1개, batch 8, **약 8분** (11 ckpt × 2 모드) |

⚠ **분할 고정은 2026-07-29에 고친 것이다.** 그 전에는 `dm.train_dataloader()`(`shuffle=True`)의 순서로
잘라서 **분할이 런마다 바뀌었고**, 그래서 probe AUROC를 런 간에 비교할 수 없었다(같은 teacher가
0.670 vs 0.711로 갈렸다). CKA는 원래 영향이 없었다(`base_ep029` 0.4375 / 0.4377로 두 런 일치).
셔플 분할로 뽑은 옛 결과는 `results/semantic_shuffledsplit.json`,
그 이전 ep029 단일 런은 `results/semantic_ep029_context.json`에 남겼다 — **모두 이 문서에 의해 대체됨.**

## 헤드라인: **REPA는 ep009에 semantic gap을 메우고, baseline은 300 epoch 동안 못 메운다**

`middle_block` tap. 왼쪽이 context OFF(헤드라인), 오른쪽이 ON:

| ep | lin(OFF) | k-NN(OFF) | **CKA(OFF)** | lin(ON) | k-NN(ON) | CKA(ON) | | FID_Avg | CLIP_T2I |
|---|---|---|---|---|---|---|---|---|---|
| base_ep009 | 0.645 | 0.571 | 0.504 | 0.845 | 0.772 | 0.337 | | 2.21 | 17.4 |
| base_ep029 | 0.645 | 0.532 | 0.438 | 0.849 | 0.767 | 0.356 | | 1.75 | 21.4 |
| base_ep079 | 0.632 | 0.529 | **0.358** ← 최저 | 0.851 | 0.793 | 0.326 | | **1.435** ← 바닥 | 37.1 |
| base_ep119 | 0.657 | 0.537 | 0.397 | 0.837 | 0.768 | 0.388 | | 1.61 | 44.2 |
| base_ep179 | 0.651 | 0.576 | 0.481 | 0.830 | 0.748 | 0.480 | | 1.67 | **48.2** ← 최고 |
| base_ep239 | 0.628 | 0.567 | 0.470 | 0.821 | 0.736 | 0.465 | | 1.47 | 45.4 |
| base_ep299 | 0.631 | 0.575 | 0.464 | 0.825 | 0.727 | 0.465 | | 1.461 | 46.9 |
| **repa_ep009** | **0.685** | **0.670** | **0.828** | 0.774 | 0.699 | 0.839 | | — | — |
| repa_ep029 | 0.685 | 0.672 | 0.876 | 0.783 | 0.698 | 0.874 | | — | — |
| repa_ep059 | 0.676 | **0.675** | 0.885 | 0.752 | 0.697 | 0.882 | | — | — |
| repa_ep099 | 0.673 | 0.669 | **0.889** | 0.737 | 0.694 | 0.884 | | — | — |
| *teacher SPECTRE-SSL* | *0.714* | *0.675* | *—* | | | | | | |

(FID/CLIP은 같은 baseline 체크포인트의 `outputs/report2ct_wan/ep_sweep/eval_ep*_n300_sp0.75_1.3_cfg1`
결과를 참고로 나란히 둔 것이다. REPA arm의 생성 지표는 아직 없다.)

### 1. baseline은 semantic gap을 **스스로 줄이지 못한다**

linear probe가 300 epoch 내내 **0.628~0.657, 폭 0.029**로 평평하고 단조 추세가 없다. k-NN도
0.529~0.576 사이를 오갈 뿐이다. teacher(0.714 / 0.675)와의 격차는 ep009에서나 ep299에서나 같다.

→ REPA가 말하는 "**느리게** 개선된다"(Fig 2c)가 우리 파이프라인에서는 "**개선되지 않는다**"에 가깝다.
양날이다: 스스로 못 메우는 gap이라면 외부 주입의 명분이 커지지만, 동시에 **그 gap이 생성 품질과
무관할 가능성**도 열어 둔다 — FID는 gap이 그대로인 채로 ep079에 이미 바닥을 쳤다.

### 2. REPA는 그 gap을 **9 epoch 만에** 메우고 그 뒤로는 평평하다

`repa_ep009`에서 이미 k-NN **0.670**으로 teacher(0.675)와의 격차가 **0.005**다. baseline이 300 epoch
동안 낸 최고치가 0.576(ep179)이니 **9 epoch 만에 그걸 0.094 넘어섰다.** CKA도 0.828로 baseline
최고치(0.504)의 1.6배. 이후 ep029/059/099는 0.876 / 0.885 / 0.889로 **포화**이고, `repa_ep059`의
k-NN 0.6748은 teacher 0.6747과 사실상 동률이다.

→ "REPA가 표현 학습을 가속한다"는 주장은 **표현 축에서 확정**이다. 남은 질문은 이게 생성 지표로
옮겨가는가뿐이고, 그건 FID/CLIP으로만 답할 수 있다.

### 3. baseline CKA는 U자를 그린다 — 최저점이 FID 바닥과 겹친다

0.504 → 0.438 → **0.358(ep079)** → 0.397 → 0.481 → 0.470 → 0.464.
**최저점이 FID가 바닥을 찍는 ep079와 정확히 겹치고, 회복 구간이 CLIP-T2I가 오르는 구간과 겹친다.**
읽자면 — 이미지 품질을 끌어올리는 동안에는 표현이 teacher에서 **멀어지고**, 그 뒤 텍스트 정합을
배우면서 다시 가까워진다. REPA가 **레이어 축**에서 관찰한 것("고주파 디테일을 만들려면 semantic-rich
표현에서 벗어나야 한다")과 방향이 같은데 우리는 그게 **시간 축**에서 나타난다.

⚠ probe가 평평한 채로 CKA만 움직이므로, **CKA 변화는 "얼마나 semantic한가"가 아니라 "SPECTRE와 얼마나
같은 방식으로 조직돼 있는가"의 변화**로 읽어야 한다.

### 4. context가 CKA를 끌어내리는 폭이 학습에 따라 사라진다 — REPA는 처음부터 없다

`ΔCKA = CKA(context ON) − CKA(OFF)`:

| ep | 009 | 029 | 079 | 119 | 179 | 239 | 299 |
|---|---|---|---|---|---|---|---|
| **baseline** | **−0.167** | −0.082 | −0.033 | −0.008 | −0.002 | −0.005 | **+0.001** |
| **REPA** | **+0.011** | −0.003 | −0.003(ep059) | −0.004(ep099) | | | |

baseline은 초기에 텍스트를 넣으면 표현이 이미지 teacher에서 **크게 멀어지는데**(−0.167), 학습이
진행되면서 그 이격이 **단조로 사라지고 끝에는 부호까지 뒤집힌다**. 텍스트를 별도 신호로 얹어 두던
것에서 **이미지와 같은 좌표계로 융합하는** 쪽으로 옮겨간다고 읽힌다.
**REPA는 ep009부터 이미 그 상태다**(|ΔCKA| ≤ 0.011). baseline이 300 epoch에 걸쳐 도달하는 지점을
9 epoch에 갖고 시작한다.

`Δlinear`도 갈린다 — baseline은 전 구간 **+0.18~+0.22**로 유지되는데 REPA는 **+0.064~+0.098**이고
줄어드는 추세다. REPA 쪽이 **텍스트 복사에 덜 기댄다**(OFF에서 이미 높고 ON에서 덜 오른다).

### 5. 정렬은 tap 지점에 국소적이다

`base_ep299` vs `repa_ep099`, context OFF:

| tap | CKA(base) | CKA(REPA) | k-NN(base) | k-NN(REPA) |
|---|---|---|---|---|
| conv_in | 0.201 | 0.189 | 0.513 | 0.522 |
| down_blocks.0 | 0.182 | 0.167 | 0.521 | 0.511 |
| down_blocks.1 | 0.301 | **0.204** ↓ | 0.533 | 0.504 |
| down_blocks.2 | 0.448 | **0.341** ↓ | 0.555 | 0.584 |
| down_blocks.3 | 0.467 | **0.598** | 0.566 | **0.671** |
| **middle_block (tap)** | 0.464 | **0.889** | 0.575 | **0.669** |
| up_blocks.0 | 0.498 | **0.620** | 0.604 | **0.674** |
| up_blocks.1 | 0.489 | **0.423** ↓ | 0.620 | 0.620 |
| up_blocks.2 | 0.427 | 0.429 | 0.580 | 0.564 |

tap에서 최대(0.464 → 0.889)이고 바로 이웃(down3, up0)까지 번지되 **그 바깥에서는 오히려 내려간다.**
표현 자원이 tap 쪽으로 재배치된 것으로 읽힌다. REPA가 의도한 대로 국소적으로 작동한다는 직접 증거다.

### 6. "pre-bottleneck은 어떤가"에 대한 답

`down_blocks.3`은 `middle_block`과 같은 방향으로 움직이지만 CKA가 낮다(0.598 vs 0.889).
k-NN은 0.671로 오히려 근소 우위지만 차이가 작다(0.669). baseline에서도 semantic 품질이 거의 같다.
→ **`middle_block` 유지가 맞다.** U-REPA의 U-Net 결론과도 일치한다.
baseline의 layer별 semantic 품질은 전체적으로 **평평하다**(k-NN 0.51~0.62). DiT처럼 뚜렷한 peak
layer가 있는 구조가 아니라, U-Net에서는 down2~up1 구간이 고만고만하다.

## ⚠ 반드시 알아야 할 교란: 라벨이 **리포트에서 추출된 것**이다

18개 abnormality 라벨은 radiology report에서 뽑은 것이고, UNet은 **바로 그 리포트를 cross-attention
조건으로 받는다**. 조건을 켠 채로 probe하면 "이미지 표현이 얼마나 semantic한가"가 아니라
"텍스트 조건을 얼마나 잘 베껴 왔는가"를 재게 된다. 실제로 baseline은 **전 구간 +0.18~+0.22**가 뛴다
(§4 표). 그래서 **기본값은 context를 0으로 두는 것**(CFG unconditional 경로)이고, 헤드라인은 전부 그 조건이다.

§4의 ΔCKA가 이 해석의 **독립 증거**다: context ON의 이득이 "이미지 표현이 더 semantic해져서"였다면
이미지 teacher와의 CKA도 같이 올라야 하는데, 초기 baseline에서는 **반대로 내려간다**.

**내부 정합성 확인**: `conv_in` / `down_blocks.0` / `down_blocks.1`은 context ON/OFF에서 **완전히 동일한
수치**가 나온다. `attention_levels: [false, false, true, true]`라 그 세 곳엔 cross-attention이 없기 때문이다
— context drop이 실제로 작동하고 있다는 증거다.

이 교란은 REPA 서사 자체에도 유보를 건다: REPA는 **class-conditional ImageNet**에서 측정됐다. 라벨
하나로는 semantic 표현을 대신할 수 없으니 denoiser가 스스로 배워야 한다. 우리는 리포트 전문을 받으므로
**REPA가 메우려는 gap의 상당 부분이 이미 조건화로 공급된다** — 이득의 여지가 구조적으로 작을 수 있다.

## teacher의 semantic 상한

| representation | linear AUROC | k-NN AUROC |
|---|---|---|
| **spectre_vla scan-level** (combiner CLS) | **0.788** | **0.800** |
| spectre_vla dense mean-pool | 0.747 | 0.684 |
| spectre_ssl dense mean-pool | 0.714 | 0.675 |

scan-level 임베딩이 dense 격자를 평균낸 것보다 확실히 낫다(당연 — 그러라고 학습된 표현이다).
VLA가 SSL보다 semantic한 것도 예상대로다(리포트로 정렬 학습). 다만 **U3/U6에서 본 공간 구조는
SSL이 더 좋았다** — semantic과 spatial이 갈리는 지점이고, iREPA의 주장(생성에는 spatial이 중요)대로면
REPA teacher로는 SSL이 맞다.

## 한계

- baseline과 REPA는 **다른 학습 런**이다. epoch 축을 나란히 놓는 것은 두 런의 epoch 정의가 같다는
  가정에 기댄다(같은 datalist·같은 batch size).
- REPA arm은 **ep099까지만** 존재한다(학습 진행 중). baseline의 U자 회복 구간(ep179~)에 해당하는
  지점이 아직 없다.
- test 100볼륨이라 AUROC의 표준오차가 작지 않다. 순위는 견고해 보이지만 소수점 셋째 자리는
  신뢰하지 말 것. CKA는 300볼륨 전체를 쓰므로 더 안정적이다(세 번의 런에서 넷째 자리까지 재현).
- CT-RATE 라벨 자체가 리포트에서 **자동 추출**된 것(`*_predicted_labels.csv`)이라 그 자체로 노이즈가 있다.
- probe는 **timestep 0.5 한 점**에서만 쟀다. REPA 논문도 단일 timestep을 쓰지만, 우리 U4는 정렬이
  timestep에 따라 달라짐을 봤으므로 이 축의 민감도는 미측정이다.

## 다음

**REPA arm의 FID/CLIP.** 표현 축의 가속은 확정됐고, 남은 것은 그게 생성으로 옮겨가는지다.
baseline은 `outputs/report2ct_wan/ep_sweep/eval_ep*_n300_sp0.75_1.3_cfg1`로 30 epoch 전 구간이
이미 측정돼 있으므로, REPA ep009/029/059/099를 **같은 설정**(n300, sp 0.75/0.75/1.3, cfg1)으로
돌리면 곧바로 겹쳐 볼 수 있다. 특히 볼 지점은 **초기 구간에서 CLIP-T2I 곡선이 baseline 위로 뜨는가**다
— baseline은 CLIP이 ep179까지 느리게 올랐고, REPA의 이득이 있다면 거기서 나와야 한다.
