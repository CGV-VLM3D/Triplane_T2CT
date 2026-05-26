# Deep Interview Spec: MAISI Latent Distribution Analysis (for Triplane AE Design)

## Metadata
- Interview ID: maisi-latent-analysis-2026-05-20
- Rounds: 4 (Round 0 + 3 scoring rounds + 1 contrarian/expansion round)
- Final Ambiguity Score: 17.2%
- Type: brownfield (triplane AE 코드/Tier 실험 진행 중)
- Generated: 2026-05-20
- Threshold: 20%
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.86 | 0.35 | 0.301 |
| Constraint Clarity | 0.85 | 0.25 | 0.213 |
| Success Criteria | 0.78 | 0.25 | 0.195 |
| Context Clarity | 0.80 | 0.15 | 0.120 |
| **Total Clarity** | | | **0.828** |
| **Ambiguity** | | | **0.172** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| C1 · Per-channel univariate | active | 채널 4개의 1D 분포·moments·fp16 안전성 | D2(quantization), D4(loss) 후보 ranking에 직접 입력 |
| C2 · Inter-channel structure | active | 채널 corr·PCA·effective rank | D2(quantization) 후보 ranking에 직접 입력 |
| C3 · Spatial-axis anisotropy & projectability | active | 축별 marginal·triplane projection MSE | D1(triplane 가설) 후보 ranking — 가장 결정적 |
| C4 · Spatial frequency spectrum | active | 3D/축별 radial spectrum·energy bands | D3(backbone) 후보 ranking — 장거리 구조 측정 |
| C5 · Inter-sample variability & spatial location maps | **deferred** | 샘플 간 voxel-wise variance map | 다음 라운드로. PE 결정 시 재오픈 |

## Goal
6,000개의 1mm MAISI VAE latent (`mu.pt`, shape `[4, 120, 120, 64]` fp16)의 분포·구조를 **4개 영역(univariate / inter-channel / axis-projectability / spatial spectrum)** 으로 분석해, 각 영역에서 산출한 정량 지표를 근거로 4개 triplane AE 설계 결정(D1: triplane 가설, D2: 토큰화, D3: 백본, D4: loss)의 후보들을 **순위로 정렬**한 evidence-based report를 만든다. 사용자가 최종 설계 선택.

## Constraints
- GPU: `CUDA_VISIBLE_DEVICES=0` (1장)만 사용. **실제 실행 직전 별도 허락 받고 시작** (이 스펙 작성 시점에는 GPU 미사용).
- Walltime 제한 없음.
- 데이터: 1mm 원본만, 학습 5,000 + 검증 1,000 (전체 6,000개). 2mm 토이는 이번 라운드 미포함.
- 도메인: **raw `mu.pt`를 주축**으로 분석. PCA만 normalized·raw 둘 다 산출 (스케일 영향 큰 분석).
- 분석은 read-only — `/workspace/datasets/datasets/latents/`는 절대 쓰지 않음.
- 산출물 위치: `/workspace/analysis/maisi_latent_distribution/`
- 결과 형식: `REPORT.md` 1개 + `figs/*.png` + `tables/*.md` + 캐시 (`cache/per_sample_stats.parquet` 등).

## Non-Goals
- 모델 학습 / 추론 코드 수정 (스펙 단계). 분석 결과를 보고 따로 결정.
- 2mm 토이 latent 분석 (다른 라운드).
- C5 (sample variability 공간 맵) — deferred.
- 새로운 데이터 전처리, 텍스트 컨디셔닝, diffusion.
- "설계를 단정적으로 추천"하지 않음. 후보 순위만.

## Acceptance Criteria
- [ ] `analysis/maisi_latent_distribution/REPORT.md` 가 존재하고, 4개 결정(D1~D4) 각각에 대한 후보 순위표(rank + 정량 근거)를 포함한다.
- [ ] **C1 산출물**: `figs/c1_hist.png` (4채널 히스토그램+KDE), `figs/c1_qq.png` (Q-Q vs normal), `tables/c1_moments.md` (mean/std/skew/kurt/min/max/p1/p5/p50/p95/p99), `tables/c1_fp16_safety.md` (|z|>5 비율, ±65504 근접 voxel 수).
- [ ] **C2 산출물**: `figs/c2_corr.png`, `figs/c2_cov.png`, `figs/c2_pca_pve_raw.png`, `figs/c2_pca_pve_norm.png`, `figs/c2_joint_hist.png` (6쌍), `tables/c2_rank.md` (effective rank, explained variance).
- [ ] **C3 산출물**: `figs/c3_axis_var.png` (X/Y/Z 축별 marginal 분산 비율, per channel), `tables/c3_proj_loss.md` ("XY/YZ/XZ projection만 남기면 latent MSE +X dB"), `figs/c3_triplane_sum_recovery.png` (3 plane mean → 합산 복원 PVE).
- [ ] **C4 산출물**: `figs/c4_radial_spectrum.png` (3D radial |k|, per channel), `figs/c4_axis_spectra.png` (X·Y·Z 1D), `tables/c4_energy_bands.md` (저/중/고주파 에너지 비율, autocorr length).
- [ ] **`REPORT.md` 결정 섹션**: D1·D2·D3·D4 각각 아래 형식
  ```
  ### D{n}: <decision name>
  | Rank | Candidate | Evidence (수치) | Confidence |
  | 1 | ... | C{x}: ... | high/med/low |
  ```
- [ ] 학습 / 검증 split의 통계 차이 확인 (KS-statistic per channel) — 큰 차이 있으면 분리, 없으면 합산본만 노출.
- [ ] 실행 직전 사용자에게 GPU 사용 허락 명시적으로 받음.
- [ ] 6,000개 전수 처리. 샘플링 결과 아님.
- [ ] 재실행 가능 (캐시된 per-sample stats 사용 시 < 5분에 리포트 재생성).

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "정규화된 latent와 raw latent 결과가 다를 것" | Contrarian: per-channel affine은 corr·spectrum 불변, hist 모양도 불변 | C1·C3·C4는 raw 한 도메인. C2 PCA만 둘 다 산출 |
| "분석은 자연스럽게 triplane keep을 정당화할 것" | Contrarian: triplane 가설이 깨질 가능성 | D1 후보에 abandon → low-rank 3D AE 포함 |
| "토큰화는 continuous KL VAE 또는 표준 VQ 중 선택" | Contrarian: FSQ/LFQ/RVQ 등 modern alternative | D2 후보를 8개로 확장 |
| "Loss는 latent space MSE 변형으로 충분" | Contrarian: 최종 목표는 image space PSNR | D4 후보에 MAISI-decoder passthrough loss 포함 |
| "C5 (variability map)도 꼭 필요" | Simplifier: 첫 라운드 결정과 직접 연결 부족 | Deferred. PE 결정 시 재오픈 |

## Technical Context
**Brownfield 코드 (이미 존재)**:
- 데이터 로더: `src/data/maisi_latent_dataset.py` — `mu.pt` `[4,120,120,64]` fp16 → float32, `normalize=True` 시 channel-wise affine.
- 채널 통계: `/workspace/datasets/datasets/latents/stats.json` (mean/std는 이미 계산됨; **skew/kurt/quantile/outlier는 미계산** — 이 분석에서 새로 산출).
- Triplane AE: `src/models/triplane_encoder.py`, `triplane_decoder.py`, `triplane_ae.py`. 인코더 출력 = `{z_xy[B,8,30,30], z_yz[B,8,30,16], z_xz[B,8,30,16]}`, patch_size=4.
- Upper bound: 30.94 ± 2.97 dB PSNR (`results/upper_bound.json`).
- 분석 스크립트는 `reference/scripts/compute_latent_stats.py` 1개만 존재 (mean/std만 산출).

**상수**:
- 채널 std (raw): `[0.8371, 0.1642, 0.6836, 0.6054]` → 채널 1은 다른 채널 대비 ~5× 작음. 분석에서 이 비대칭의 원인을 explicit하게 다뤄야 함.

**환경**: Python 3.11, PyTorch 2.x, MONAI, wandb. Hydra. A6000 Blackwell.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| MAISILatent | core domain | shape `[4,120,120,64]`, fp16, mu, sigma | 4 Channels 보유; 6,000개 인스턴스 |
| Channel | core domain | idx ∈ {0,1,2,3}, mean, std, skew, kurt | belongs to MAISILatent |
| Triplane | core domain | planes ∈ {XY,YZ,XZ}, per-plane shape | derived from MAISILatent via axis collapse |
| Decision | core domain | id ∈ {D1,D2,D3,D4}, scope | informed by Components C1~C4 |
| DesignCandidate | supporting | name, decision_id, rationale | belongs to Decision; ranked by RankingRule |
| RankingRule | supporting | metric, threshold, candidate_preference | uses outputs of C1~C4 |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 5 | 5 | - | - | N/A |
| 2 | 5 | 0 | 0 | 5 | 100% |
| 3 | 6 | 1 (DesignCandidate, RankingRule merged into D~) | 0 | 5 | 83% |
| 4 | 6 | 0 | 0 | 6 | 100% (수렴) |

## Analysis Plan (구체 실행 단위)

### Stage 0: Pipeline scaffold
- 디렉터리 생성: `/workspace/analysis/maisi_latent_distribution/{figs,tables,cache}`.
- 단일 진입점 스크립트: `scripts/analyze_maisi_latent_distribution.py` (Hydra 없이 간단한 CLI 또는 `python -m`).
- 데이터 로딩: `MAISILatentDataset(split="train", normalize=False)` + `split="valid"` 두 번. fp16 → float32 즉시 변환.

### Stage 1: Streaming pass A — per-sample stats → parquet
- 한 샘플당 per-channel `[sum, sum_sq, sum_cube, sum_quartic, min, max, count]`, fp16 overflow count, |z|>5 count.
- 6,000 행 × 4 채널 → `cache/per_sample_stats.parquet`.
- Welford 누적으로 전역 mean/std/skew/kurt 계산 → `tables/c1_moments.md`.

### Stage 2: Histogram + Q-Q (C1)
- 256-bin per-channel histogram, raw 도메인. 6,000개 streaming 합산.
- Q-Q는 채널당 1,000개 랜덤 voxel 샘플 (전체는 4.6B voxel, Q-Q는 시각화면 충분).

### Stage 3: Channel covariance + PCA (C2)
- per-voxel 4-vector → online covariance accumulation. GPU 텐서 ops.
- raw·normalized PCA 각각 산출. effective rank = exp(entropy of eigvals).
- 6쌍 2D joint hist (`H×W` bins=128).

### Stage 4: Axis-projection MSE (C3) ★
- 각 채널마다 raw `[120,120,64]` 텐서에 대해:
  - `xy_proj = x.mean(dim=2) → expand → reconstruction` → MSE in dB
  - 같은 식으로 YZ, XZ projection.
  - **3-plane mean sum recovery**: `x ≈ (xy_proj.expand + yz_proj.expand + xz_proj.expand) / 3`, MSE/PVE 측정.
- 샘플별 6,000개 평균.

### Stage 5: Spectral analysis (C4)
- 3D FFT per channel per sample → magnitude squared → radial binning (binned by `sqrt(kx²+ky²+kz²)`).
- 6,000개 평균 → `figs/c4_radial_spectrum.png`.
- 축별 1D FFT (다른 축은 평균) → `figs/c4_axis_spectra.png`.
- Energy in {low: |k|<0.1, mid: 0.1≤|k|<0.3, high: |k|≥0.3} bands.
- Spatial autocorrelation length = first zero crossing of inverse-FFT of power spectrum.

### Stage 6: Train/Valid drift check
- Per-channel KS-statistic on histograms. If max KS > 0.05 → 분리 표 출력.

### Stage 7: REPORT.md 생성
- 각 결정(D1~D4)에 대해 후보 순위표 + 근거 plot 임베드.
- 결정 룰 예시 (스펙 단계의 사전 정의):
  - **D1**: 3-plane mean sum recovery PVE > 90% → triplane keep 1순위. 70~90% → hybrid 1순위. < 70% → abandon 1순위.
  - **D2**: max(channel kurtosis) > 10 → VQ/FSQ 우대. effective rank < 2.5 → quantization 효율 큼.
  - **D3**: high-band energy > 30% → conv/하이브리드 우대. < 15% → axial attention 우대.
  - **D4**: max(|z|>5 fraction) > 1% → L1/Charbonnier 우대. fp16 overflow 비율 > 0.01% → asinh/log warp 우대.

### Stage 8: 재실행성
- `cache/`에 per-sample stats 저장 → 캐시 hit 시 Stage 1~2 skip, Stage 7만 재실행 가능.

## Interview Transcript

<details>
<summary>Full Q&A (4 rounds)</summary>

### Round 0 — Topology
**Q:** 5개 후보 컴포넌트(C1 univariate / C2 inter-channel / C3 axis-projectability / C4 spectrum / C5 inter-sample variability) 중 어떤 걸 활성화?
**A:** C1+C2+C3+C4 활성, C5 deferred (clarification 후 lock).
**Result:** topology locked.

### Round 1 — Goal/Decisions
**Q:** 분석 결과로 triplane 코드의 어떤 디자인 레버를 조정?
**A:** triplane 가설 검증, VQ vs continuous(KL) VAE, conv vs attention, loss 설계 — 연구 초기 단계 architecture-class 결정.
**Ambiguity:** 34.4%.

### Round 2 — Constraints (GPU/walltime)
**Q:** GPU 허락 + walltime?
**A:** GPU 0 OK, 시간 제한 없음.
**Ambiguity:** 25.7%.

### Round 3 — Success Criteria (report style)
**Q:** 리포트가 어떤 형태로 결론?
**A:** 후보 순위 + 근거 (Recommended).
**Ambiguity:** 20.0%.

### Round 4 — Contrarian (candidate set expansion)
**Q:** 결정별 후보 셋이 적절한가? Contrarian 후보 포함?
**A:** "더 넓혀서 탐색" → 확장된 매트릭스 제시 후 "All" 선택.
**Ambiguity:** 17.2% ✅.

</details>
