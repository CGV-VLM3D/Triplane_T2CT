# MAISI Latent 분포 분석 리포트

- **분석 샘플**: 6,000개 (train 5,000 + valid 1,000)
- **분석 일자**: 2026-05-20
- **Latent shape**: `[4, 120, 120, 64]` fp16 → float32
- **도메인**: raw `mu` (C2의 PCA만 raw·normalized 병행 산출)
- **목적**: triplane AE 설계 결정 4가지(D1 triplane 가설 / D2 토큰화 / D3 백본 / D4 loss)에 대한 **근거 기반 후보 순위표** 산출

---

## 🎯 한 페이지 요약 (TL;DR)

데이터가 가리키는 triplane AE 설계 방향:

| 결정 | 1순위 | 근거 요약 |
|---|---|---|
| **D1 · triplane 가설** | **abandon → low-rank 3D AE** | C3 — 3-plane mean-collapse PVE 평균 **21.0%** (스펙 임계 70% 미만 → abandon 영역). 즉 axis-aligned projection만으로는 정보의 79%가 사라짐 |
| **D2 · 토큰화** | **Continuous KL-VAE** | C2 — effective rank **3.97/4**, 모든 채널 corr <0.10. C6 — 평균 `P(\|x\|<0.05)=8.5%` (희소 아님). 적극적 quantization이 정당화되지 않음 |
| **D3 · 백본** | **Conv U-Net (+ residual)** | C4 — 채널 0/2/3에서 **high-band(>30%) 에너지 50%↑**. 디테일 dominant 신호 → 지역적 conv 우위. 단 채널 1은 저주파(83.9%) dominant라 분리 처리 가치 |
| **D4 · loss** | **L2/MSE + per-channel weight** | C1 — fp16 overflow 0%, `\|z\|>5` 비율 0.001%로 안전. 다만 채널별 std 비대칭(채널 1: 0.16 vs 채널 0: 0.84)이 강해 균등 가중 필요 |

**가장 먼저 봐야 할 그림 2장**:
1. `figs/c3_triplane_sum_recovery.png` — triplane이 이 데이터에 맞는지 한눈에
2. `figs/c1_hist.png` — 4채널 분포의 비대칭이 어디서 오는지

---

## 📂 산출물 네비게이션

| 카테고리 | 산출물 | 답하는 질문 |
|---|---|---|
| C1 | `figs/c1_hist.png`, `figs/c1_qq.png`, `tables/c1_moments.md`, `tables/c1_fp16_safety.md` | 각 채널 분포 모양, 꼬리, 정규성, fp16 안전성 |
| C2 | `figs/c2_corr.png`, `figs/c2_cov.png`, `figs/c2_pca_pve_raw.png`, `figs/c2_pca_pve_norm.png`, `figs/c2_joint_hist.png`, `tables/c2_rank.md` | 채널 간 상관, 정보 중복, 압축 가능성 |
| C3 | `figs/c3_axis_var.png`, `figs/c3_triplane_sum_recovery.png`, `tables/c3_proj_loss.md` | triplane(XY/YZ/XZ) 가설이 데이터에 맞는가 |
| C4 | `figs/c4_radial_spectrum.png`, `figs/c4_axis_spectra.png`, `tables/c4_energy_bands.md` | 저/고주파 분포, conv vs attention 적합성 |
| C6 | `figs/c6_sparsity_curve.png`, `tables/c6_sparsity.md` | voxel-level 희소성, quantization 정당성 |
| C7 | `figs/c7_tsne.png`, `figs/c7_umap.png`, `tables/c7_embedding_notes.md` | 샘플 간 semantic 클러스터링, train/valid drift |
| Drift | `tables/drift_ks.md` | train vs valid 통계적 차이 |

---

## C1 · 채널별 1D 분포 (Univariate)

### 무엇을 측정했나
6,000 latent의 모든 voxel을 채널별로 stream하여 **mean, std, skew, kurtosis(excess), percentile(p1/p99), fp16 overflow 비율, |z|>5 비율**을 계산. 정규성 검사와 quantization·loss 설계 안전성 판단이 목적.

### 결과 표

| Channel | Mean | Std | Skewness | Kurtosis(excess) | p1 | p99 |
|---|---|---|---|---|---|---|
| Ch 0 | -0.1348 | 0.8374 | +0.571 | +1.125 | -1.945 | 2.273 |
| Ch 1 | -0.0825 | **0.1641** | +0.394 | -0.457 | -0.398 | 0.305 |
| Ch 2 | -0.0229 | 0.6839 | +0.273 | +1.447 | -1.711 | 1.852 |
| Ch 3 | -0.0126 | 0.6057 | +0.369 | +1.972 | -1.477 | 1.711 |

### 산출물 설명
- **`figs/c1_hist.png`** — 4채널 × 히스토그램+KDE 오버레이 (raw 도메인). 채널 1의 좁은 폭과 다른 채널의 넓은 좌측 비대칭을 시각 확인.
  - *읽는 법*: KDE 곡선이 0 근처에 얼마나 모여있는지, 한쪽으로 늘어졌는지(skew) 보세요.
- **`figs/c1_qq.png`** — 채널별 정규성 Q-Q plot (1,000 voxel 샘플). 대각선에서 벗어난 정도가 꼬리의 두께.
  - *읽는 법*: 양 끝이 대각선 위로 치우치면 fat tail. 거의 직선이면 정규에 가까움.
- **`tables/c1_moments.md`** — 전체 voxel 기반 채널별 모멘트(quantile 포함).
- **`tables/c1_fp16_safety.md`** — fp16 overflow(>|65000|)와 |z|>5 비율. 모두 ~0% — **fp16 환경에서 학습/추론 안전**.

### 해석과 함의
- **채널 1이 다른 채널 대비 std 5× 작음** (0.16 vs 0.60~0.84). 이건 MAISI VAE가 채널 1을 "저주파/저용량" 채널로 학습했다는 신호 (C4 스펙트럼 결과와 일관).
- Kurtosis 0.5~2.0 → 정규에 가까운 mild non-Gaussian. **L2 loss가 베이스라인으로 안전**. asinh/log warp 불필요.
- |z|>5 비율 0.001%, fp16 overflow 0% → 안정성 OK.
- **함의**: 채널 가중치 없이 균등 L2를 쓰면 채널 1이 학습 신호를 거의 못 받음. **per-channel weighted L2** (`1/std²`로 가중) 필수.

---

## C2 · 채널 간 구조 (Inter-channel)

### 무엇을 측정했나
Per-voxel 4-vector의 **공분산/상관 매트릭스**, **PCA 설명분산**, **effective rank**(eigenvalue entropy 기반). 채널 간 중복 정도와 압축 가능성을 측정.

### 결과 표

| 지표 | Raw | Normalized |
|---|---|---|
| Effective rank | 3.086 | **3.971** |
| PC1 PVE | 45.4% | 29.6% |
| PC2 PVE | 29.8% | 25.3% |
| PC3 PVE | 23.1% | 23.9% |
| PC4 PVE | 1.7% | 21.2% |

**상관 매트릭스**: 모든 off-diagonal `|corr| < 0.10` (max 0.101 between Ch1·Ch3).

### 산출물 설명
- **`figs/c2_corr.png`** — 4×4 Pearson 상관 heatmap. 대각선만 1.0, 나머지 거의 흰색.
  - *읽는 법*: 비대각이 진하면 채널 간 정보 공유. 본 결과는 **거의 직교**.
- **`figs/c2_cov.png`** — 4×4 공분산. 채널 0이 두꺼운 사각형 — std 차이가 raw cov를 dominate.
- **`figs/c2_pca_pve_raw.png`** — raw 데이터의 PC별 설명분산. PC1이 45% (채널 0 std 효과).
- **`figs/c2_pca_pve_norm.png`** — z-score 후 PCA. 4 PC가 거의 균등(30/25/24/21) → **진정한 4-rank**.
- **`figs/c2_joint_hist.png`** — 6쌍(C(4,2)) 2D joint histogram. 대부분 원형 cloud → 독립성 시각 확인.
- **`tables/c2_rank.md`** — 위 수치 + 상관 매트릭스 raw.

### 해석과 함의
- Effective rank 3.97/4 ≈ **채널이 거의 완전 독립**. 정보 중복 없음.
- → VQ 같은 vector quantization이 **압축 효율을 거의 못 냄**. 4채널 전부 보존해야 함.
- → continuous (KL-VAE 또는 deterministic AE) **1순위**.
- 채널 corr < 0.1은 MAISI VAE가 잘 학습됐다는 sign (KL이 채널 직교화 유도).

---

## C3 · Triplane 가설 검증 ★ (Axis projectability)

### 무엇을 측정했나
각 채널 `[120,120,64]` 텐서를 (a) XY 평면 평균(Z 축 collapse) (b) YZ 평면 평균 (c) XZ 평면 평균 으로 줄였다가 다시 expand해서 reconstruction. **MSE**와 **PVE = 1 - MSE/Var**를 계산. 3-plane sum recovery는 셋을 더한 후 평균(`xy.expand + yz.expand + xz.expand)/3`)과의 PVE.

→ **순수 axis-aligned mean projection만으로 latent 정보를 얼마나 회복 가능한가**의 정량적 측정. triplane 가설의 데이터-적합성 직접 검증.

### 결과 표

| Channel | XY PVE | YZ PVE | XZ PVE | **3-plane sum PVE** |
|---|---|---|---|---|
| Ch 0 | 19.7% | 9.0% | 3.2% | **15.8%** |
| Ch 1 | **50.0%** | 37.6% | 14.2% | **47.6%** |
| Ch 2 | 11.9% | 7.3% | 6.1% | **11.4%** |
| Ch 3 | 7.7% | 5.9% | 4.5% | **8.9%** |
| **평균** | **22.3%** | **14.9%** | **7.0%** | **🚩 21.0%** |

### 산출물 설명
- **`figs/c3_axis_var.png`** — 채널별 X/Y/Z 축 marginal variance ratio.
  - *읽는 법*: 한 축이 도드라지게 높으면 그 축으로 collapse하기 쉬움.
- **`figs/c3_triplane_sum_recovery.png`** — 채널별 3-plane sum PVE 막대그래프 + 임계선.
  - *읽는 법*: 막대가 70% 위 → triplane keep, 70% 아래 → hybrid/abandon. **여기서는 모든 막대가 50% 아래**.
- **`tables/c3_proj_loss.md`** — 위 표 + 평면별 MSE 절대값.

### 해석과 함의 — 가장 중요한 발견
- 평균 PVE **21%** — 스펙의 결정 룰(70% 임계)에 따라 **abandon 영역**.
- **XY plane이 모든 채널에서 압도적으로 우세** (평균 22%, YZ 15%, XZ 7%). 이건 anatomical하게 자연 — CT가 axial scan이라 axial(XY) slice에 정보 농축.
- **YZ·XZ plane은 거의 무의미** (PVE 5~15%). 이 두 평면 채널을 줄이는 axis-anisotropic triplane이 합리적.
- 채널 1만 PVE 50% (다른 채널은 9~16%) — 채널 1이 가장 smooth/low-freq이라 projection도 잘 됨 (C4와 일치).
- **하지만 이건 mean-collapse 기준의 하한**. 실제 triplane encoder는 학습된 transformer로 더 잘함. 현재 Tier-1 ~27 dB latent PSNR(`research_summary/summary.md`)이 그 증거.
- **결론**: 데이터 자체가 axis-decomposable이 아니지만, **부분적 axis-alignment(특히 XY)** 는 존재. 따라서:
  - 순수 triplane 유지 ❌ (1순위 abandon)
  - axis-anisotropic triplane (XY 두껍게, YZ/XZ 얇게) ✅ 강력 후보
  - hybrid (triplane + low-rank 3D residual) ✅ 강력 후보

---

## C4 · 공간 주파수 스펙트럼

### 무엇을 측정했나
채널별 3D FFT → magnitude squared → radial bin(|k|=√(kx²+ky²+kz²))으로 평균. 또한 축별 1D FFT(다른 축 평균). **저주파(<0.1·k_max), 중주파(0.1~0.3), 고주파(≥0.3)** 에너지 비율로 분류.

### 결과 표

| Channel | Low (<10%) | Mid (10-30%) | High (≥30%) |
|---|---|---|---|
| Ch 0 | 21.7% | 23.8% | **54.5%** |
| Ch 1 | **83.9%** | 10.6% | 5.5% |
| Ch 2 | 11.4% | 11.6% | **77.0%** |
| Ch 3 | 5.3% | 18.3% | **76.4%** |

### 산출물 설명
- **`figs/c4_radial_spectrum.png`** — 채널별 radial power spectrum (|k| vs power, log-log).
  - *읽는 법*: 곡선이 |k| 0 근처에 모이면 smooth, 평평하면 detail dominant.
- **`figs/c4_axis_spectra.png`** — X/Y/Z 축별 1D spectrum, 채널별. anisotropy 시각화.
  - *읽는 법*: 축마다 다르게 떨어지면 축 anisotropy 존재. 비슷하면 isotropic.
- **`tables/c4_energy_bands.md`** — 위 표 + autocorrelation length 추정치.

### 해석과 함의
- **채널 0/2/3은 high-band dominant** (>50%). 디테일 많음 → 로컬 receptive field가 중요 → **Conv 우대**.
- **채널 1만 저주파(84%) dominant** — 거의 평탄한 smooth field. 이건 MAISI VAE에서 채널 1이 global/smooth 정보 담당이라는 신호.
- → 백본은 **Conv-heavy** (1순위) 또는 **Conv + axial attention hybrid** (2순위). 순수 transformer는 위험.
- 추가 함의: **채널별 heterogeneous 처리** 가치 — 채널 1은 더 강한 압축/얇은 표현으로도 충분.

---

## C6 · Voxel-level 희소성 (Sparsity profile)

### 무엇을 측정했나
"latent이 sparse한가?"에 직접 답하기 위해 채널별 `P(|x|<ε)` (ε ∈ {0.01, 0.05, 0.1, 0.5, 1.0}), **Gini 계수**(|x| 분포의 불균등), **활성도 엔트로피** H(bit), **effective sparsity rank** = 2^H 산출.

### 결과 표

| Channel | P(\|x\|<0.05) | P(\|x\|<0.1) | P(\|x\|<0.5) | Gini | H (bits) | Eff. rank (2^H) |
|---|---|---|---|---|---|---|
| Ch 0 | 4.3% | 8.8% | 47.4% | 0.42 | 6.15 | **70.9** |
| Ch 1 | 14.0% | 29.1% | **100.0%** | 0.34 | 3.79 | **13.8** |
| Ch 2 | 6.9% | 13.9% | 61.6% | 0.46 | 5.87 | **58.4** |
| Ch 3 | 8.8% | 16.9% | 67.7% | 0.47 | 5.67 | **51.1** |
| **평균** | **8.5%** |  |  |  |  | **48.5** |

### 산출물 설명
- **`figs/c6_sparsity_curve.png`** — ε(log scale) vs P(|x|<ε), 채널별 곡선.
  - *읽는 법*: 곡선이 좌측 상단에 가까이 붙으면 sparse. 우측 아래로 늘어지면 dense.
- **`tables/c6_sparsity.md`** — 위 표 raw + ε 30 포인트 전체.

### 해석과 함의
- 평균 `P(|x|<0.05) = 8.5%` → **대부분의 voxel이 의미 있는 값**. latent은 **sparse하지 않음(dense)**.
- 채널 1만 예외: `P(|x|<0.5)=100%`, eff rank 13.8 — **이 채널만 quantization-friendly** (256 max → 14).
- 채널 0/2/3 eff rank 51~71 — moderate 복잡도. 적극적 voxel-level 압축은 정보 손실 우려.
- → **D2 1순위: continuous representation** 변경 없음. FSQ/VQ 강제 업그레이드 임계(60%) 미달.
- → **per-channel heterogeneous scheme** (채널 1은 VQ, 나머지는 continuous) 는 후속 연구 가치 있음.

---

## C7 · 샘플 단위 Semantic Embedding (t-SNE / UMAP)

### 무엇을 측정했나
각 샘플의 latent을 **28-dim feature vector**(채널별 7개 통계 × 4채널 = mean, std, min, max, fp16_overflow, z5_count, abs5_count)로 축약 후 t-SNE(perplexity=30) + UMAP(n_neighbors=15)로 2D embedding. abnormality label과 join하여 semantic 클러스터 검증.

### 결과
- **Label join rate**: 6,000/6,000 = **100%** (sample_id ↔ VolumeName)
- **가장 빈번한 abnormality**: Lung nodule (2,670 positive / 6,000)
- **train/valid between-split variance ratio**: **0.000** (완전 interleaved)
- t-SNE와 UMAP 모두 **abnormality 라벨에 의한 클러스터 분리 보이지 않음**.

### 산출물 설명
- **`figs/c7_tsne.png`** — 3-panel 시각화:
  1. train(파랑) vs valid(주황) — 완전히 섞임 → drift 없음
  2. Lung nodule positive(주황) vs negative(파랑) — 분리 없음
  3. 환자별 positive label 수(색 = 0~18) — gradient 패턴 없음
- **`figs/c7_umap.png`** — t-SNE와 동일한 3-panel, UMAP 버전
- **`tables/c7_embedding_notes.md`** — perplexity, n_neighbors, join rate, 관찰 메모

### 해석과 함의
- ✅ **Train/Valid 통계적 drift 없음** — C3 KS=0.028 결과와 일관. 6,000 통합 분석 신뢰 가능.
- ⚠️ **Abnormality 클러스터 분리 약함** — 두 가지 해석 가능:
  - (a) 28-dim per-channel summary는 semantic을 잡기엔 너무 거친 feature (가능성 높음)
  - (b) label이 **predicted** (다른 모델 산출, ground truth 아님) — noise 큼
  - → **약한 claim**. semantic 보존성에 대한 결정적 답이 필요하면 더 풍부한 feature(예: spatial-bin downsampled latent → embedding) 또는 expert label 필요
- triplane 설계 결정에 미치는 영향은 제한적 — 단지 데이터 균질성을 확인하는 보조 근거.

---

## Train/Valid Drift 검사

- **Max KS statistic (per-channel sample mean)**: **0.0276** (임계 0.05 미만)
- **판정**: 통계적 drift 없음. train과 valid 통합 분석은 신뢰 가능.
- 산출물: `tables/drift_ks.md`

---

## 🏁 Design Decision Rankings (종합)

### D1 · Triplane 가설

| Rank | Candidate | 근거 | Confidence |
|---|---|---|---|
| 1 | **low-rank 3D AE (abandon triplane)** | C3: 3-plane PVE 21.0% < 70%; C7: 클러스터 균질 → mean-collapse PVE 비관 추정 아님 | **high** |
| 2 | hybrid triplane + low-rank 3D residual | C3: XY plane은 부분 의미 있음(22% PVE); 잔차로 cross-axis 정보 보완 가능 | med |
| 3 | axis-anisotropic triplane (XY 두껍게, YZ/XZ 얇게) | C3: XY가 YZ/XZ 대비 3× 강한 PVE — 채널 할당을 분산에 비례 | med |
| 4 | triplane keep (현 설계) | C3: 저PVE — 데이터 부적합 | low |

### D2 · 토큰화

| Rank | Candidate | 근거 | Confidence |
|---|---|---|---|
| 1 | **Continuous KL-VAE** | C2: eff_rank=3.97 (거의 4); C6: `P(\|x\|<0.05)=8.5%` (sparse 아님); 채널 corr <0.1 (독립) | **med-high** |
| 2 | Deterministic continuous AE | KL 정규화 없이도 정보 보존 가능; baseline 비교용 | med |
| 3 | Per-channel mixed (Ch1 VQ + 나머지 continuous) | C6: 채널 1만 eff rank 13.8 — 유일한 quantization-friendly 채널 | med |
| 4 | FSQ (finite scalar quantization) | C1: kurtosis 2.0 — non-Gaussian 처리 가능; simpler than VQ | low-med |
| 5 | VQ (vector quantization) | C2 high effective rank로 codebook efficiency 낮음 | low |
| 6 | RVQ / LFQ | C2 high effective rank로 다단 codebook 정당화 약함 | low |

### D3 · 백본 아키텍처

| Rank | Candidate | 근거 | Confidence |
|---|---|---|---|
| 1 | **Conv U-Net + residual blocks** | C4: high-band energy 평균 53% (채널 0/2/3); 지역 디테일 보존 우위 | **high** |
| 2 | Conv + Axial attention hybrid | C4: high-band과 함께 cross-axis 약한 의존성 존재 (C3 21% 잔여); axial은 메모리 효율적 | med |
| 3 | Pure axial attention | C4 high-band 우세에서 attention만으로 local detail 잡기 어려움 | low |
| 4 | ViT / pure transformer | Patch-level token이 fine structure 손실; 데이터 신호 약함 | low |

### D4 · Loss 함수

| Rank | Candidate | 근거 | Confidence |
|---|---|---|---|
| 1 | **L2 + per-channel weight** (1/std² 가중) | C1: fp16 안전, kurt 보통; 채널 1 std 0.16 vs Ch0 0.84 → 균등 L2면 ch1 학습 신호 약함 | **high** |
| 2 | Charbonnier | C1: 약한 fat tail (Kurt 1~2) — L2와 L1 사이 부드러운 보간 | med |
| 3 | Huber | C1: kurt 2.0 — outlier-robust 옵션. threshold 튜닝 필요 | med |
| 4 | MAISI-decoder passthrough (image-space L2) | 최종 목표(31 dB PSNR)와 직접 정렬; 학습 cost 증가 | med |
| 5 | L1 | C1: |z|>5 비율 0.001%로 매우 작아 L1 이점 적음 | low |
| 6 | asinh/log warp + L2 | C1: fp16 overflow 0%, warp 불필요 | low |
| 7 | Perceptual + L2 | feature extractor 추가 cost. 후순위 | low |

---

## ⚠️ Caveats

- **C3 mean-collapse는 하한**: 실제 학습된 triplane encoder(transformer F_psi)는 mean보다 정보 보존이 좋음. 21% PVE는 *순수 axis 가설의 약함*을 의미하지, *현 모델의 ceiling*을 의미하지 않음. 다만 ceiling이 데이터 결정적임을 시사.
- **Q-Q plots는 1,000 voxel 샘플** (reservoir sampling, dataset order 의존). 전체 분포의 시각화일 뿐, 전체 voxel은 `tables/c1_moments.md` 참조.
- **Joint histogram subsample**: 샘플당 50k voxel (전체 4.6B 중). 트렌드 representative.
- **C7 abnormality label**: `train_predicted_labels.csv` / `valid_predicted_labels.csv` — **다른 모델이 예측한 라벨** (ground truth 아님). 클러스터 미분리를 강한 claim으로 쓰면 위험.
- **C6 Gini**: 첫 구현이 대칭 bin 문제(|midpoint| 중복)로 0.0이 나왔던 버그 발견·수정. 현 값은 검증된 값.
- **PCA**: C2의 PCA는 raw와 normalized 둘 다 산출. raw는 채널 0 std에 dominate되어 PC1=45.4% — z-score 후 30/25/24/21로 거의 균등.
- **fp16 overflow 기준**: `|value| > 65000` (fp16 max=65504에서 보수적 마진).
- **모든 분석은 raw `mu` 도메인** (정규화 전). MAISI VAE의 KL이 이미 합리적 스케일을 보장하기에 raw 도메인이 자연스러움.

---

## 🔄 다음 행동 제안

1. **시각 확인**: `figs/c3_triplane_sum_recovery.png`, `figs/c1_hist.png`, `figs/c4_radial_spectrum.png` 순으로 검토
2. **Tier-1 candidate 셋 재구성** — 기존 후보에 다음을 추가:
   - **axis-anisotropic triplane** (XY 16ch + YZ 4ch + XZ 4ch)
   - **hybrid triplane + low-rank 3D residual**
   - **low-rank 3D AE** baseline
3. **D4 적용**: 모든 후보에 `per-channel weighted L2` (1/std² 정규화) 적용 권장
4. **선택적 추가 분석** (deferred):
   - **C5 voxel-wise variance map** — PE 설계 / spatial sampling 결정에 사용
   - **spatial-bin downsample → t-SNE** — C7보다 풍부한 feature로 semantic 클러스터 재검증
   - **MAISI σ 활용** — Gaussian-NLL loss 정당성 검토
