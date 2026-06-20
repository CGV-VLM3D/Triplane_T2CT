# Experiment 1 — Alignment Probing: 실행 + 검증 계획

**가설**: text condition이 image space와 정렬될수록 text-to-CT 생성이 좋아진다 (mask-free
anatomy-routed VLP의 동기). 이 실험은 그 **precondition**(인코더별 text↔image 정렬도)을 정량화.

**대상 인코더 5종 (전부 freeze)**: T5(GenerateCT) / Report2CT(3-enc concat) / CT-CLIP /
Text2CT 3D-CLIP / fVLM. **데이터**: CT-RATE ctrate_toy_v2 — train 5000 / valid_v2 1304.

---

## 1. 재사용 감사 (중복 구현 회피)

| 필요 기능 | 레포 기존 | 결정 |
|---|---|---|
| valid_v2 케이스 로더 | `src/eval/ct_rate_cases.load_eval_cases` | 패턴 재사용, train+labels 합쳐 `cases.py` |
| 18-label 목록 | `tests/saliency_map/data.ABNORMALITY_LABELS` | **그대로 import** (단일 소스) |
| fVLM organ text | `src/data/fvlm_organ_report.build_organ_text` | 그대로 호출 |
| 각 인코더 텍스트 인코딩 | `src/baselines/*_adapter.py` | 그대로 호출 (rewrite 없음) |
| MAISI latent | `mu.pt`(train) / `_emb.nii.gz`(proxy) | `latents.py` 얇은 로더 |
| 18-label AUROC | `third_party/ct_clip/scripts/eval.py` (per-label `roc_auc_score`) | **동일 metric** = sklearn `roc_auc_score` 직접 (docker는 I/O 계약이 달라 import 안 함) |
| linear probe / PCA / t-SNE / silhouette | **없음** | 신규 (`probe.py`/`tsne.py`), 표준 sklearn 사용 |

## 2. 표준 하이퍼파라미터 (GitHub/논문 관행)

- **Probe A** (content): sklearn `LogisticRegressionCV` per-label — solver=lbfgs,
  `max_iter=1000`, `Cs=10` log-spaced L2 sweep, `cv=3`, `scoring=roc_auc`.
  출처: Radford et al. 2021 CLIP linear-probe appendix (lbfgs + L2 C sweep이 표준).
- **Probe B** (image-predictability): sklearn `RidgeCV(alphas=logspace(-3,3,13))` multi-output,
  **train→proxy cross-split**. feature·target 모두 `StandardScaler`(train 통계).
  지표 = R²(variance-weighted) + 차원별 Pearson r. cosine은 폐기(z②가 ReLU feature라
  전부 비음수 → 평균 pairwise cosine 0.92로 비변별).
  ⚠️ **교훈**: train latent을 처음에 `ctrate_toy_v2/train/.../mu.pt`(raw VAE mu, std 0.67)로
  잘못 써서 z② train↔proxy가 1.74σ 어긋나 cross-split이 R²≈−6로 붕괴. 정본
  `report2ct_work_dir/image_embeddings/*_emb.nii.gz`(std 0.98, proxy와 동일)로 교체 →
  0.036으로 정상화, cross-split 유효. ([[maisi-latent-source-scale-mismatch]])
- **용량 교란 통제**: 모든 인코더 raw + **PCA-256**(공통 차원) 둘 다 보고.
- **z② 분류기** (Unit 2): 최소 3D CNN(1.17M params), Adam `lr=1e-3`, BCE, batch=16, epoch=20,
  random-init(text 인코더와 독립 → 순환논증 방지).
- **t-SNE**: `perplexity ∈ {5,30,50}`(robustness), `init=pca`, primary=30; silhouette 동반.

## 3. 실행 순서 (의존성)

1. **1b** text 추출 (5enc × 2split) → `embeddings/<split>/<enc>.npz`  ✅ 완료
2. **2** z② 분류기 학습 + penultimate 추출 → `embeddings/<split>/z_classifier.npz`  (진행 중)
3. **3+4** `probe.py` → Probe A/B → `results/probe.json`
4. **5** `tsne.py` → `results/tsne_p30.png` + `tsne_silhouette.json`
5. **6** CKA/mutual-kNN — **사용자 개념학습 후 재승인 (여기서 멈춤)**

## 4. 검증 계획 (정량 PASS/FAIL)

| ID | 검증 | 정량 기준 | 상태 |
|---|---|---|---|
| **V1** | 추출 무결성 | dim 정확(t5=768,r2ct=2560,ctclip=512,text2ct=768,fvlm=1024), N=(5000/1304), NaN 0, scan_id 순서 5인코더 동일, CLIP계열 ‖v‖≈1(ctclip/text2ct), fvlm‖v‖≈2(=√4 concat) | ✅ PASS |
| **V2** | z② 분류기 (=Probe C) | proxy mean-AUROC **≥ 0.70** (미달 시 epoch↑/raw-volume/CT-Net로 escalate) | ✅ ep4 0.705 |
| **V3** | Probe A 대조군 | **모든** 인코더 proxy mean-AUROC **≥ 0.75** (text가 내용은 담음). 낮은 인코더는 flag → Probe B 해석 시 분리 | 대기 |
| **V4** | Probe B 가설 | R²/Pearson 순서 **contrastive > 비-contrastive**(content 통제 시); 전 인코더 R²>0 | ✅ 지지: ctclip(0.39)·text2ct(0.35)가 content 대비 image-예측 초과 = alignment 신호. fVLM(0.28)은 global에서 최하(local 강점 미반영) — per-organ faithful은 neutral per-organ target 부재로 future work |
| **V5** | t-SNE 보조 | abnormality별 silhouette 부호가 3 perplexity에서 일관; visual_z silhouette ≥ text 평균(시각 표현이 더 뭉침) — 단독 근거 아님 | 대기 |

**해석 규약**: V3에서 A 모두 높고 V4에서 B가 갈리면 → "내용 부족이 아니라 alignment 차이"로
귀결(가설 지지). z②의 Probe C AUROC가 Probe B cosine/R²의 천장.

## 5. 산출물 (정리 대상)

영구: `cases.py, templates.py, text_embed.py, latents.py, classifier.py, probe.py, tsne.py`,
`embeddings/`, `results/`, 본 PLAN.md. **임시/검증용 스크립트는 남기지 않음** (무결성 체크는
인라인 1회 실행). `--limit`은 부분 재실행용 일반 유틸로 유지.
