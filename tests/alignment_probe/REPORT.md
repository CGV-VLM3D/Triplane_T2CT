# Experiment 1 — Alignment Probing: 결과 리포트

자율 실행 완료 (Units 1b–5 + per-organ 4b). **Unit 6 (CKA/mutual-kNN)은 사용자 개념학습 후 재승인 지점에서 멈춤.**
데이터: CT-RATE ctrate_toy_v2 — train 5000 / valid_v2 1304 (Probe B/C/t-SNE는 backfill 전 425, per-organ은 1304).
인코더 5종 freeze: T5(GenerateCT) / Report2CT(3-enc concat) / CT-CLIP / Text2CT 3D-CLIP / fVLM.
latent 정본: Report2CT `_emb.nii.gz` (train=`report2ct_work_dir/image_embeddings`, proxy=`ctrate_toy_v2/valid_v2/latents`).

## TL;DR
- **가설("text가 image space와 정렬될수록 image feature 예측↑")은 지지된다**: image-text contrastive로 학습된
  **CT-CLIP·Text2CT**가, content를 통제해도, 비-contrastive(T5·Report2CT)보다 z②를 잘 예측.
- **CT-CLIP이 정렬도 최강 — global·per-organ 양쪽 모두 최고.** global Probe B 0.393(최고), per-organ도
  whole-report만으로 mean R² 0.339로 fVLM organ-text(0.110)를 모든 organ에서 능가.
- **fVLM의 anatomy-routing은 작동하나(routing 0.645≫0.25, organ "판별" OK) 더 풍부한 per-organ 예측으로는
  이어지지 않음** — per-organ 임베딩이 정보-희박(대부분 정상 템플릿). 즉 fVLM은 global Probe B에서도
  per-organ에서도 image-feature 예측력은 최하. ctgen 조건부 인코더로는 **CT-CLIP이 일관 우세.**

## 결과

### Probe A — content 대조군 (text → 18-label, proxy mean-AUROC)
| encoder | raw (d) | AUROC | macro-F1 | pca256 AUROC |
|---|---|---|---|---|
| report2ct | 2560 | **0.978** | 0.843 | 0.970 |
| ctclip | 512 | 0.972 | 0.814 | 0.968 |
| text2ct | 768 | 0.919 | 0.636 | 0.913 |
| t5 | 768 | 0.910 | 0.654 | 0.901 |
| fvlm | 1024 | 0.890 | 0.658 | 0.897 |

→ **V3 PASS**: 전 인코더 ≥ 0.89. 모든 인코더가 "내용"은 담고 있다 ⇒ Probe B의 격차는
content 부족이 아니라 **alignment 차이**로 해석 가능. (metric = 공식 abnclass 평가
`third_party/ct_clip/scripts/eval.py`와 동일한 per-label `roc_auc_score`.)

### Probe B — image-predictability (text → z②, train→proxy cross-split)
| encoder | R² | Pearson r | image-aligned? | Probe A |
|---|---|---|---|---|
| **ctclip** | **0.393** | **0.623** | ✅ global contrastive | 0.972 |
| **text2ct** | **0.352** | **0.592** | ✅ global contrastive | 0.919 |
| report2ct | 0.305 | 0.554 | ✗ pure text | 0.978 |
| t5 | 0.289 | 0.530 | ✗ pure text | 0.910 |
| fvlm | 0.282 | 0.530 | ✅ anatomy-**local** | 0.890 |

**핵심 해석 (content vs alignment 분리)**: Probe B는 content와 부분 상관(둘 다 report 정보에 의존).
순수 content면 Probe A 순위를 따라야 하는데 — **report2ct는 Probe A 최고(0.978)인데 Probe B는 3위(0.305)**,
**CT-CLIP/Text2CT는 content가 더 적은데도(0.972/0.919) Probe B가 더 높다(0.393/0.352)**.
즉 contrastive 인코더는 **content 대비 image-예측력이 초과** = 그 초과분이 곧 **alignment 신호**. → 가설 지지.

### Probe C — image-side content / z② 천장 (z② 분류기 자체)
- proxy mean-AUROC **0.770** (**V2 PASS** ≥ 0.70). z② = semantic feature로 유효.
- Probe B의 R²(≤0.39)는 이 천장(0.77 AUROC가 담는 semantic) 아래에서 해석.

### t-SNE (V5, 보조) — `results/tsne_p30.png`, `results/tsne_silhouette.json`
- 공통 425 scan, perplexity {5,30,50}, abnormality별 present/absent silhouette. 전반 낮음(보조 근거).

### Per-organ faithful (Unit 4b) — `results/per_organ.json` (proxy 1304, 5-fold OOF)
global Probe B가 과소평가한 fVLM을 anatomy-local하게 재평가. neutral per-organ target `z_organ` =
z② 분류기 pre-pool feature map(256,15,15,8)을 organ occupancy로 가중 풀링(mask는 MAISI와 동일
RAS→Resize(120,120,64) nearest 정렬). 측정 2종:

정합 진단: coverage 4 organ 모두 1.00 ✅(V-PO1); inter-organ cosine **0.954**(V-PO2 미달 — z②가
ReLU라 organ 간 z_organ이 매우 유사 → "organ-o z 예측"의 대부분이 **공통성분** 예측).

**per-organ R² (text → z_organ_o), fVLM organ-text vs global whole-report (V-PO5):**

| encoder (입력) | lung | heart | esoph | aorta | **mean** |
|---|---|---|---|---|---|
| **fvlm_organ** (organ별 text) | 0.291 | 0.066 | −0.001 | 0.085 | **0.110** |
| **ctclip** (whole-report) | 0.391 | 0.273 | 0.344 | 0.349 | **0.339** |
| text2ct (whole-report) | 0.344 | 0.240 | 0.307 | 0.313 | 0.301 |
| t5 (whole-report) | 0.222 | 0.158 | 0.201 | 0.212 | 0.198 |
| report2ct (whole-report) | 0.261 | 0.154 | 0.218 | 0.236 | 0.217 |

| routing top-1 acc (fVLM organ-text가 같은 scan 4 organ 중 o를 맞힘) | **0.645** (chance 0.25) ✅ V-PO4 |

**해석 (중요 — global baseline이 그림을 뒤집음)**:
- **CT-CLIP의 whole-report 텍스트가 fVLM organ-text보다 per-organ image feature를 모든 organ에서 더 잘
  예측**한다(0.339 vs 0.110; t5·report2ct도 fVLM보다 높음). 즉 per-organ **regression은 fVLM에 유리하지
  않다.** 이유: z_organ이 organ 간 0.954로 유사해 예측의 대부분이 공통성분이고, fVLM organ-text는 대부분
  "정상 템플릿"이라 정보가 희박(esoph R²≈0)한 반면 global 텍스트는 전체 report의 풍부한 신호를 가짐.
- **단 routing(0.645≫0.25)은 fVLM organ-text가 "어느 organ인지"는 강하게 구별**함을 보인다(global single-vector는
  애초에 못 받는 시험). 그러나 이는 organ **판별**이지 image content **예측력**은 아니다.
- **종합 결론**: image-feature 예측력(=정렬도)은 **CT-CLIP이 global·per-organ 양쪽 모두 최고**다. fVLM의
  anatomy-routing은 작동하나(판별 OK), 그것이 더 풍부한 per-organ 예측으로 이어지지는 **않는다** —
  fVLM의 per-organ 임베딩은 정보가 희박. ctgen 조건부 인코더로는 **CT-CLIP이 일관되게 우세**하다는
  global Probe B 결론과 합치.

## 핵심 수정 / 교훈
1. **train latent 소스 실수 → 수정 완료**: 처음에 `ctrate_toy_v2/train/.../mu.pt`(raw VAE mu, std 0.67)를
   썼는데, 정본은 `report2ct_work_dir/image_embeddings/*_emb.nii.gz`(std 0.98, proxy와 동일 공간).
   잘못 썼을 때 z② train↔proxy가 1.74σ 어긋나 cross-split이 R²≈−6로 붕괴 → 정본 교체 후 **0.036**으로
   정상화, cross-split 유효. (메모리: [[maisi-latent-source-scale-mismatch]].)
2. **z②가 ReLU feature**(전부 비음수, 평균 pairwise cosine 0.92) → cosine 비변별이라 **폐기**,
   target per-dim 표준화 후 R²/Pearson 사용.
3. **표준 probe 프로토콜**: Probe A `LogisticRegressionCV`(lbfgs/max_iter=1000/L2 C-sweep, Radford CLIP),
   Probe B `RidgeCV`(alpha-sweep) — 임의 epoch/lr 제거, 인코더별 정규화 자동 선택.
4. **fVLM per-organ 재평가(Unit 4b) 결과**: neutral per-organ target(z² feature map organ 가중풀링)으로
   재평가. routing 0.645로 organ **판별**은 입증되나, per-organ **예측 R²는 CT-CLIP whole-report(0.339)이
   fVLM organ-text(0.110)를 압도** → fVLM의 locality가 더 나은 image 예측을 주지는 않음. (z_organ이
   ReLU라 organ 간 cosine 0.954로 유사 — regression은 공통성분 위주; routing이 그나마 locality를 격리.)
5. **proxy latent backfill 완료(425→1304)**. Probe B/C/t-SNE는 425 시점 산출 — 재실행 시 1304로 자동 확대
   (per-organ은 이미 1304).

## 검증 요약
| V1 추출 | V2 z²(ProbeC) | V3 ProbeA | V4 ProbeB | V5 t-SNE | per-organ(4b) |
|---|---|---|---|---|---|
| ✅ | ✅ 0.770 | ✅ ≥0.89 | ✅ contrastive↑; fVLM global 최하 | ✅ | routing 0.645✅ / per-organ R²: CT-CLIP 0.339 ≫ fVLM 0.110 |

## 다음 (Unit 6 — 멈춤)
**CKA / mutual-kNN probe-free 정합** — 네 개념학습 후 재승인하면 진행. probe head 없이 용량 교란을
원천 회피하는 Platonic 계열 표준 지표로 Probe B 결론을 교차검증. (선택) fVLM per-organ faithful
분석용 neutral per-organ target 설계도 같이 논의.

## 파일 (전부 `tests/alignment_probe/`)
`cases.py` · `templates.py` · `text_embed.py` · `latents.py` · `classifier.py` · `probe.py` ·
`tsne.py` · `PLAN.md` · `embeddings/`(npz+logs) · `results/`(probe.json, tsne).
