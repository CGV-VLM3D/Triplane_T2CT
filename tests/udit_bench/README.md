# `tests/udit_bench/` — U-DiT 백본 비용 실측 (로드맵 Phase 2a)

`docs/vlm3d_research_roadmap.md` Phase 2 착수 전, 지금까지의 모든 비용 추정(활성화
메모리 ~3.2GB, MFU 25–30%)이 **가정 위에** 있었던 걸 실측으로 교체하는 프로브.
계획: `/root/.claude/plans/u-repa-joyful-narwhal.md`

**격리 원칙**(`tests/repa_probe/README.md` 관례): 여기 코드는 `src/`·`configs/`를 건드리지
않는다. 더미 랜덤 텐서만 쓰고, 학습에는 아무것도 연결돼 있지 않다.

## 한 줄 결론

**활성화 메모리 추정은 맞았다 (3.9GB 실측 vs 3.2GB 추정, U-Net 5.8GB 대비 −33%).
처리량 추정은 너무 비관적이었다** — GPU 자원을 경합하는 상태에서도 U-Net(toy 5k 실측
10.4 samp/s/GPU)보다 U-DiT가 2배 이상 빠르게(21.49 samp/s/GPU) 나왔다. RTX PRO 6000
Blackwell ×3 기준 full 46k/300ep 환산: U-Net ~5.2–5.7일 vs **U-DiT ~2.5일**
(GPU가 자유로우면 더 짧아질 여지가 있음 — 아래 "측정 조건" 참조).
**이 규모에서는 gradient checkpointing이 필요 없다** (batch16도 69.5/97GB로 여유).

## 구조

```
tests/udit_bench/
├── README.md      ← 이 파일
├── model.py        ← HierarchicalUDiT3D: L0(32³/d384/window8) → L1(16³/d768) → L2(8³/d1152 bottleneck) → L1 → L0
├── bench.py        ← 실측 러너: peak/steady-state memory, step time, TFLOPS
└── results.json     ← 마지막 실행 결과 (batch×checkpoint 6-way grid)
```

## 아키텍처 — 로드맵 스펙의 구체화

로드맵 문서의 "L0 4블록 / L1 8블록 / L2 12블록"을 **인코더-보틀넥-디코더 대칭 U자형**으로
구체화했다: L0 = 2 enc + 2 dec(윈도우 8³, shifted-window 교대), L1 = 4 enc + 4 dec(global),
L2 = 12 블록 보틀넥(한 번만 실행). patch=2 (Wan latent `(16,64,64,64)`을 네이티브 해상도로
소비). 윈도우 partition/reverse는 `monai.networks.nets.swin_unetr`의 검증된 함수를 그대로
가져다 썼다(재구현 없음). **텍스트 조건 없음** — cross-attn/MM-DiT 비용은 이미 분석적으로
계산돼 있어(로드맵 Phase 2c, +5% FLOPs) 실측이 필요한 쪽은 3D 계층 자체의 진짜 비용이었다.
그래서 params(404M)·FLOPs(1.65 TFLOP/샘플)는 로드맵 2c 표의 "cross-attn DiT 465M/2.39T"와
직접 비교 대상이 아니다 — 후자는 cross-attention을 포함한 수치다.

## 실측 결과

3× RTX PRO 6000 Blackwell(≈ CLAUDE.md의 "A6000 Blackwell") 중 1장, batch 4/8/16 × checkpoint on/off:

| batch | ckpt | peak mem | act/sample | 처리량 | achieved TFLOPS |
|---|---|---|---|---|---|
| 4 | on | 9.22 GB | 773 MB | 16.04 samp/s | 79.6 |
| 8 | on | 12.53 GB | 796 MB | 17.01 samp/s | 84.4 |
| 16 | on | 20.47 GB | 890 MB | 17.75 samp/s | 88.0 |
| 4 | off | 21.39 GB | 3888 MB | 20.19 samp/s | 100.2 |
| 8 | off | 37.43 GB | 3980 MB | 21.49 samp/s | 106.6 |
| **16** | **off** | **69.51 GB** | **4027 MB** | **22.58 samp/s** | **112.0** |

steady-state(파라미터+옵티마이저, fp32 AdamW) 6.2–6.6GB — 404M×16B/param 이론값(6.46GB)과
일치, 하네스가 올바르게 재고 있다는 교차검증.

## 로드맵 추정과의 대조

| | 로드맵 추정 | **실측** | 평가 |
|---|---|---|---|
| 활성화/샘플 (checkpoint 없음) | ~3.2 GB | **3.9 GB** | 22% 높음 — "구현 오버헤드로 30–50% 증가 가능" 캐비어트 안에 있음 |
| U-Net 대비 절감 | −45% | **−33%** (5.8GB → 3.9GB) | 방향은 맞음, 폭은 과장돼 있었음 |
| MFU / 처리량 | 25–30% (근거 없는 가정) | **U-Net의 2배+** (21.49 vs 10.4 samp/s) | **추정이 너무 비관적이었음** — 아래 참조 |
| checkpoint 필요성 | "grad ckpt로 batch 8까지" 가정 | **불필요** — batch16도 69.5/97GB | 단순화: 이 스케일에선 checkpoint 안 켜도 됨 |

## 학습 시간 재계산 (실측 기반, 이전 5–8일 추정 대체)

배치 8 · checkpoint off · A6000 Pro(RTX PRO 6000 Blackwell) ×3 선형 스케일링 가정:

| | 처리량(3-GPU) | 46,393 볼륨/epoch | 300 epoch |
|---|---|---|---|
| Wan U-Net (toy 실측 10.4 samp/s/GPU 외삽) | 31.2 samp/s | ~24.8분 | **~5.2–5.7일** |
| **Wan U-DiT (이번 실측, GPU 경합 중)** | **64.47 samp/s** | **~12.0분** | **~2.5일** |

## 측정 조건 — 반드시 감안할 것

- **GPU 경합 상태에서 잰 값.** 사용자의 `generate_wan_latents.py`가 4-GPU 전체(GPU 2 포함)를
  쓰는 도중 GPU 3 한 장을 나눠 써서 측정했다(사용자가 "지금 GPU 3에서 바로 진행"으로 명시
  승인). **메모리 수치는 이 경합에 영향받지 않는다**(프로세스별 별도 할당자 arena). **처리량/
  TFLOPS는 이 경합 때문에 과소평가돼 있다** — 즉 위 "U-DiT가 2배 빠르다"는 결론은 이미
  DiT에 불리한 조건에서 나온 하한값이다. GPU가 온전히 비었을 때 재측정하면 격차가 더 벌어질
  가능성이 높다.
- **더미 랜덤 데이터, 텍스트 조건 없음, 실제 로그/체크포인트 저장 없음.** 실제 학습 wall-clock은
  두 모델 모두 이보다 늘어나지만, U-Net과 DiT에 상대적으로 비슷하게 적용될 오버헤드라
  "DiT가 더 빠르다"는 상대적 결론 자체는 유지될 가능성이 높다.
- **DDP 3-GPU 선형 스케일링은 가정이지 측정이 아니다.**
- **achieved_tflops는 checkpoint on/off에 동일한 3×fwd-FLOPs 배수를 적용한 값**이라
  checkpoint on 쪽은 실제 재계산 비용(이론상 ~4/3배)을 과소산정한다 — checkpoint on/off
  간 TFLOPS를 직접 비교하지 말 것. batch/off 간 비교(위 학습 시간 표에서 쓴 것)는 영향받지
  않는다.

## 재현

```bash
CUDA_VISIBLE_DEVICES=3 python tests/udit_bench/bench.py \
  --dims 384,768,1152 --depths 2,4,12 --num-heads 6,12,18 --window 8 \
  --batch-sizes 4,8,16 --checkpoint both --n-warmup 5 --n-timed 10
```

## 알아두면 좋은 것 — adaLN-Zero의 step-0 zero-grad

CPU shape 스모크(코드에 남아있지 않음, 개발 중 1회성 검증)에서 174개 파라미터 텐서 중
128개가 첫 backward에서 정확히 grad=0으로 나왔다 — 버그가 아니라 DiT/U-REPA의
adaLN-Zero 초기화(`adaLN_modulation`의 마지막 Linear를 weight=bias=0으로 초기화)의
직접적 결과다. `[[maisi-unet-zero-init-output-conv]]` 메모의 MAISI UNet zero-conv와
같은 메커니즘 — 첫 옵티마이저 스텝 후 자동 해소된다. 벤치마크의 메모리/시간 측정 자체에는
영향 없음(dense 계산 그래프라 gate 값이 0이어도 동일한 FLOPs/메모리가 소요됨).
