# U2b — teacher forward 정밀도 · 학습 스텝 오버헤드

원자료: [results/precision.json](results/precision.json) · [results/steptime.json](results/steptime.json)

## 1. teacher 추출 정밀도 → **TF32**

precompute가 GPU-bound로 나왔다(4 shard에서 24 s/it, GPU util 100 %). ViT-L의 matmul이 기본 fp32라
tensor core를 전혀 못 쓰고 있었다. 다만 teacher feature는 REPA의 정렬 타깃이라 마음대로 낮출 수 없어,
**상대 구조가 보존되는지**를 먼저 재고 골랐다 (기준을 미리 정함: cos_mean ≥ 0.9999 **and** cos_min ≥ 0.999).

`python -m tests.repa_probe.u2b_io.precision` (valid_1000_a_1, 32³ 전체 토큰, fp32 기준선 대비)

| mode | forward | cos_mean | cos_min | max&#124;Δ&#124; | SRSS |
|---|---|---|---|---|---|
| fp32 | 1.314 s | 1.0 | 1.0 | 0 | 0.128078 |
| **tf32 ← 채택** | **0.353 s (3.7×)** | 0.99999988 | **0.99998** | 0.078 | 0.128084 |
| bf16 | 0.203 s | 0.99995595 | 0.99556 | 0.768 | 0.128054 |
| fp16 | 0.205 s | 0.99999946 | 0.99991 | 0.143 | 0.128059 |
| (fp32 → fp16 저장 왕복) | — | 1.0 | 0.9999996 | 0.004 | 0.128085 |

TF32는 기준을 **자릿수 단위로** 통과하면서 3.7× 빠르다. bf16은 6.5×로 더 빠르지만 최악 토큰 cosine이
0.9956까지 떨어져 꼬리가 나쁘고, fp16 autocast는 학습된 적 없는 정밀도에서 attention logit 오버플로 위험이
있다. 전체 파이프라인에서 load가 여전히 크므로 bf16/fp16의 추가 이득은 작다.
→ `scripts/precompute_spectre_features.py`는 TF32가 기본, `--no-tf32`로 끌 수 있다.

## 2. 학습 스텝 오버헤드 → **teacher grid 16³ 확정**

`CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u2b_io.steptime --steps 40 --num-workers 12`
(batch 8, bf16-mixed, 실제 datamodule + UNet forward/backward. precompute가 아직 돌고 있어 완성된
valid 701 스캔으로 임시 datalist를 만들어 쟀다 — 파일 크기와 파일시스템이 같아 train에 그대로 전이된다.)

| arm | step | dataloader 대기 | teacher I/O | 오버헤드 | peak GPU |
|---|---|---|---|---|---|
| baseline (REPA off) | 0.8917 s | 0.0004 s | 0 | — | 80.1 GB |
| **REPA 16³ ← 채택** | 0.9141 s | 0.0006 s | 70.8 MB/step | **+2.5 %** | 80.1 GB |
| REPA 32³ | 1.3794 s | **0.1814 s** | 566.2 MB/step | +54.7 % | 87.8 GB |

- 16³는 **계획의 S6 기준(< 25 %)을 크게 통과**한다. 44 GB짜리 train teacher 세트가 page cache에
  상주하므로 dataloader 대기가 사실상 0이다.
- 32³는 dataloader 대기가 스텝당 0.18 s로 튀어 **I/O 바운드**가 된다. 100 epoch 기준 11 h → 17 h.
  peak GPU도 87.8 GB로 96 GB에 가까워진다.
- ⚠ 절대 step 시간(0.89 s)은 실제 학습(0.67 s)보다 크다 — 이 측정은 GPU 2에서 precompute 4 shard가
  CPU/디스크를 함께 쓰는 중에 GPU 3에서 돌렸다. 판단에 쓰는 건 **비율**이다.

**U4와 결론이 일치한다**: 16³는 I/O가 8배 싼 데다 정렬도 **더 잘 된다**(cos 0.5625 vs 0.4714) —
student tap이 8³라 32³의 추가 디테일을 애초에 표현할 수 없기 때문이다. 32³ 파일은 그대로 보존돼 있으므로
논문 충실성 ablation은 언제든 가능하다.

## 부수적으로 잡은 버그

`Report2CTWanRepaDataModule`이 teacher 로더를 **bound method**로 `Lambdad`에 넘기고 있었다. MONAI는
transform을 모든 dataloader worker에 pickle하므로, bound method는 DataModule 전체(CacheDataset 포함)를
끌고 가서 worker가 `ConnectionResetError`로 죽었다. `functools.partial` + module-level 함수로 교체.
`num_workers: 32`인 실제 학습에서 곧바로 터졌을 버그다.
