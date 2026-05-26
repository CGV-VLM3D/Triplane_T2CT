# Deep Interview Spec: VLM3D 2026 Pivot — Beat Report2CT

## Metadata
- Interview ID: vlm3d-pivot-2026-05-26
- Rounds: 8 (Round 0 topology + Rounds 1-8 socratic; Round 8 added after weight-availability check)
- Final Ambiguity Score: 22% (PASSED w/ noted Compute-feasibility monitoring gate)
- Type: brownfield
- Generated: 2026-05-26
- Threshold: 20% (default)
- Initial Context Summarized: no (user prompt was concise)
- Status: PASSED (with Phase B kickoff compute-measurement gate as conditional)
- Challenge Modes Used: Contrarian (Round 4), Simplifier (Round 6)

## Clarity Breakdown (per-component, brownfield weights: goal 35 / constraints 25 / criteria 25 / context 15)

| Component | Goal | Constraints | Criteria | Context | Weighted | Ambig |
|-----------|------|------|------|------|------|------|
| 1. Repo restructure | 0.95 | 0.70 | 0.95 | 0.75 | 0.86 | 14% |
| 2. Dataset analysis | 0.85 | 0.70 | 0.90 | 0.80 | 0.82 | 18% |
| 3. Baseline repro+diagnose | 0.95 | 0.85 | 0.85 | 0.85 | 0.89 | 11% |
| 4. VLM eval infra | 0.85 | 0.80 | 0.90 | 0.90 | 0.86 | 14% |
| 5. Compute feasibility | 0.85 | 0.70 | 0.70 | 0.90 | 0.78 | 22% ⚠ |
| **Overall (min)** | | | | | **0.78** | **22%** |

> **Component 5 borderline note**: Report2CT 재구현 학습 cost는 impl 전에 정확히 산정 불가. Phase B kickoff(6/1)에서 작은 subset으로 1-epoch wall-clock 실측 후 phase B budget을 재배정하는 모니터링 게이트로 처리.

## Topology

| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| 1. Repo restructure | active | lightning-hydra-template 기반 full migration + 기존 triplane을 `deprecated/`로 이동 | Phase A (5/26-5/31) 안에 LightningModule + Hydra defaults + pytest 통과까지 (Round 6 aggressive choice — schedule risk accepted) |
| 2. Dataset analysis | active | CT-RATE를 표준 의료 AI 분석 protocol로 점검 (HU/spacing/label/report-len/KS-drift) | Minimal EDA notebook + figs/eda/ (Round 3) |
| 3. Baseline repro+diagnose | active | GenerateCT pretrained inference (Phase A) + Report2CT 논문 기반 재구현·학습 (Phase B) + text-conditioning fidelity 진단 4종. TRACE는 다운로드만 | **Weight availability check (post-Round-7)**: GenerateCT 3종 weight 공개 ✅, Report2CT weight 미공개 ❌ → Round 8 decision: 논문 기반 재구현 + CT-RATE로 학습 (MAISI VAE-GAN은 frozen). 4종 diagnostic은 두 baseline에 동일 적용 |
| 4. VLM eval infra | active | VLM3D-Dockers wiring (FVD + CLIPScore + 2.5D-FID) + Phase D submission docker | third_party/vlm3d_dockers submodule + src/eval/vlm3d_runner.py subprocess (Round 7). Submission docker는 hard deliverable |
| 5. Compute feasibility | active | 각 phase의 wall-clock budget 관리 | Deadline 2026-08-20, Phase A 5일, Phase B/C 각 4주, Phase D 3주. 뒤처지면 feature 자르기 (Round 4) |

## Goal

**MICCAI VLM3D 2026 Task 4 (Text-Conditional CT Generation)에서 Report2CT를 이긴다.** 2026-08-20 마감일까지 (1) lightning-hydra-template 기반 깨끗한 코드베이스, (2) **GenerateCT는 공개 weight로 inference 재현, Report2CT는 weight 미공개이므로 논문 기반 재구현 + CT-RATE 학습** + text-conditioning fidelity 4종 진단(cross-attention heatmap, CLIP retrieval, counterfactual pair, token-region attribution), (3) FVD/CLIPScore/2.5D-FID 평가 파이프라인 (VLM3D-Dockers submission docker 형태), (4) CT-RATE 기초 EDA 위에서 baseline의 약점을 데이터-기반으로 진단, (5) 그 진단을 근거로 우리 모델 (MAISI VAE latent diffusion + VLM-focused improvement)을 학습·제출.

## Constraints

- **Hard deadline**: MICCAI VLM3D 2026 Task 4 submission = **2026-08-20**.
- **Phase A (5/26 → 5/31, 5일)**: repo restructure + EDA + baseline pretrained inference + 초기 진단 모두 병렬 완료.
- **Hardware**: ≤ 3× A6000 Blackwell (DeepCGV-Mk7), Docker dev container, Python 3.11, PyTorch 2.x, MONAI, wandb.
- **Baseline reproduction**:
  - **GenerateCT**: pretrained weights (ctvit/transformer/superres .pt 3종, HuggingFace `generatect/GenerateCT`) inference만. Retrain X.
  - **Report2CT**: weight 미공개 → 논문(VLM3D 2025 Task 4 winner write-up) 기반 재구현 + CT-RATE 학습. MAISI VAE-GAN은 frozen 사용. Phase B(6월) 안에 학습 완료 목표.
- **External code**: 수정 금지 (third_party/ submodule + adapter 레이어). Report2CT, GenerateCT, VLM3D-Dockers 세 repo가 third_party/ 안에 들어옴.
- **Hard deliverable**: VLM3D-Dockers 스펙에 맞는 submission docker.
- **Triplane은 잠정 폐기**: 기존 triplane 모델/configs/runs는 deprecated/로 이동, 비교 대상에서 제외.
- **TRACE는 다운로드만**: 비교는 안 할 수도 있음.

## Non-Goals

- Triplane autoencoder 추가 개발 (잠정 폐기).
- Baseline 재학습 또는 paper 수치 재현 (pretrained inference 한정).
- 임상급 QC (motion artifact, beam-hardening, manual audit 등).
- 신규 데이터셋 preprocessing (CT-RATE + 기 precomputed MAISI latents 사용).
- TRACE와의 정량 비교.
- Text encoder/diffusion 외 다른 modality (e.g. 3D-vision-only) 연구.

## Acceptance Criteria

### Phase A — 5/26 → 5/31 (5일)
- [ ] `deprecated/` 폴더에 기존 `src/`, `configs/trial_*.yaml`, `runs/trial_*`, triplane tests 모두 이동.
- [ ] lightning-hydra-template 기반 새 구조: `src/{data,models,baselines,eval}/`, `configs/{data,model,trainer,logger,callbacks,experiment}/`, `tests/`. pytest 전체 통과.
- [ ] `third_party/{report2ct,generatect,vlm3d_dockers}/` submodule 등록 (Report2CT submodule은 reference 코드/논문 참조용; 학습 코드는 우리가 재구현).
- [ ] `src/baselines/generatect_adapter.py` LightningModule wrapper로 GenerateCT pretrained inference (ctvit/transformer/superres .pt 3종) 가능.
- [ ] `notebooks/eda.ipynb` + `figs/eda/`: HU histogram (per kernel/manufacturer), spacing/dim violin, 18 abnormality label freq + co-occurrence matrix, report token-length distribution, KS-test train vs valid.
- [ ] GenerateCT pretrained으로 valid 일부(≥50 샘플) 생성 + 초기 시각 검수 한 라운드.

### Phase B — 6월 (4주)
- [ ] **Phase B kickoff compute-measurement gate (6/1)**: Report2CT 재구현 1-epoch wall-clock을 작은 subset(100 샘플)으로 실측 → 전체 학습 GPU-hour 추정 → phase B budget 재배정. 만약 4주로 부족하면 (a) train subset 축소 (b) Phase C 일부 잠식 (c) Report2CT 단순화 옵션 결정.
- [ ] `src/baselines/report2ct/` 논문 기반 재구현 (multi-encoder LDM, MAISI VAE-GAN frozen, text+spacing 조건). 학습 코드는 우리 LightningModule.
- [ ] Report2CT CT-RATE train_fixed로 학습 → checkpoint 산출. Paper-reported FVD/CLIPScore와 비교 (재구현 sanity check).
- [ ] 4종 진단 풀세트 (GenerateCT + Report2CT 둘 다에 적용):
  - [ ] Cross-attention heatmap (forward hook으로 UNet cross-attn layer 출력, 토큰별 spatial map, sagittal/coronal/axial overlay, PNG grid)
  - [ ] CLIP retrieval recall (CT-CLIP, R@1/R@5/R@10, scatter plot)
  - [ ] Counterfactual generation pair (report 수정 → diff map, finding별 시각화)
  - [ ] Token-region attribution (cross-attn × TotalSegmentator/VISTA-3D anatomy seg → IoU/Dice, per-finding table)
- [ ] `src/eval/vlm3d_runner.py`로 VLM3D-Dockers (FVD + CLIPScore + 2.5D-FID) 두 baseline에 대해 valid set 전체 측정 완료.
- [ ] 우리 모델 v1 설계 (MAISI VAE latent diffusion + 진단에서 발견된 약점을 직접 겨냥하는 개선).

### Phase C — 7월 (4주)
- [ ] 우리 모델 v1 학습 + 동일 4종 진단 통과.
- [ ] Ablation (text encoder choice, conditioning depth, attention layer 위치 등) 최소 3종.
- [ ] FVD/CLIPScore/2.5D-FID에서 Report2CT 대비 유의미한 개선 가능성 확인.

### Phase D — 8/1 → 8/20 (3주)
- [ ] Final training + best checkpoint 선정.
- [ ] Submission docker (VLM3D-Dockers 스펙 준수, `test.sh` + `export.sh` 통과).
- [ ] 제출 + 짧은 method writeup.

### Win condition (전체)
- 우리 모델 FVD/CLIPScore/2.5D-FID > **우리가 재학습한 Report2CT** 수치 (적어도 2/3 메트릭에서 유의미 개선) on CT-RATE valid 1000 split, VLM3D-Dockers로 측정.
- (참고) Report2CT 재학습 수치는 paper-reported 수치와 ±X% 안에 있어야 fair comparison으로 인정 (Phase B sanity check).

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Triplane이 연구의 중심 contribution이다 | 사용자: VLM3D challenge는 VLM 초점이지 latent 압축이 아님 | Triplane 잠정 폐기, deprecated/로 이동 |
| Baseline 재학습이 필요하다 | Round 1: 5일 phase A에서 retrain은 불가능 | Pretrained inference만으로 win condition 정의 (ours > pretrained Report2CT) |
| "Text conditioning 진단"이 단일 산출물이다 | Round 2: 4가지가 서로 보완 | 4종 세트로 확장 (cross-attn / retrieval / counterfactual / token-region) |
| Compute feasibility는 별도 workstream이다 (Contrarian) | Round 4: 마감이 없으면 의미 없는 component | Challenge deadline (2026-08-20) 확정 → wall-clock budget 관리로 명확화 |
| Repo restructure를 sequential phase로 처리 (Simplifier) | Round 6: 5일은 짧음 — full migration vs skeleton-only? | Full migration 선택, schedule slip 리스크 수용 |
| 외부 baseline 코드를 우리 repo에 vendor한다 | Round 7: submodule이 더 깔끔 | third_party/ submodule + src/ 안 adapter LightningModule |
| Pretrained weight가 두 baseline 모두 공개되어 있다 | Round 8 post-Round-7 사실 확인: GenerateCT ✅, Report2CT ❌ | Round 8 decision — Report2CT 논문 기반 재구현 + CT-RATE 학습 (Phase B로 이동). Phase A는 GenerateCT only |

## Technical Context (brownfield)

### Current repo (to deprecate)
- `src/{models,data,evaluation,losses,metrics}/` — triplane 중심
- `configs/{model,train,data,eval,loss}/`, `configs/trial_*.yaml` 9+개
- `runs/trial_*/` — triplane 실험 결과
- `tests/test_{cross_attn_3d,d3t,tier0_overfit,tri_conv,triplane_ae_*}.py` — triplane 테스트
- `reference/`, `maisi_bundle/` — 유지 (MAISI VAE 그대로 사용)
- `datasets/datasets/{CT-RATE, latents}/` — 데이터 (read-only, 변경 없음)

### Target structure (Phase A 끝)
```
.
├── src/                        # lightning-hydra-template 기반 신규
│   ├── data/                   # LightningDataModule (CT-RATE, MAISI latent)
│   ├── models/                 # LightningModule (our LDM 후보)
│   ├── baselines/              # report2ct_adapter.py, generatect_adapter.py
│   ├── eval/                   # vlm3d_runner.py (FVD/CLIPScore/2.5D-FID), diagnostic/{cross_attn,retrieval,counterfactual,token_region}.py
│   ├── train.py / eval.py      # Hydra entrypoints
│   └── utils/
├── configs/
│   ├── data/, model/, trainer/, logger/, callbacks/, experiment/, hparams_search/
│   └── train.yaml, eval.yaml   # default 구성
├── third_party/
│   ├── report2ct/              # submodule
│   ├── generatect/             # submodule
│   └── vlm3d_dockers/          # submodule
├── notebooks/                  # eda.ipynb
├── tests/                      # 새 LightningModule 테스트
├── deprecated/
│   ├── triplane_src/           # 기존 src/
│   ├── triplane_configs/       # 기존 configs/trial_*.yaml
│   ├── triplane_runs/          # 기존 runs/trial_*/
│   └── triplane_tests/
├── results/                    # upper_bound.json 등 유지
└── maisi_bundle/, reference/   # 유지
```

### External resources
- **Report2CT** (https://github.com/sinaamirrajab/report2ct): MAISI VAE-GAN + multi-encoder LDM (text+spacing 조건). VLM3D 2025 1위. 우리와 가장 가까운 setup.
- **GenerateCT** (https://github.com/ibrahimethemhamamci/GenerateCT): CT-ViT autoencoder + MaskGIT + SR diffusion. 최초의 T2CT.
- **TRACE** (https://github.com/VinyehShaw/TRACE): 2.5D slice diffusion + multimodal guidance. 다운만, 비교 안 할 수도.
- **VLM3D-Dockers** (https://github.com/forithmus/VLM3D-Dockers): 3 task 평가 docker. Task 4 = FVD (CT-Net) + CLIPScore + 2.5-D FID.
- **lightning-hydra-template** (https://github.com/ashleve/lightning-hydra-template): 코드 구조 base.

### Compute
- 3× A6000 Blackwell (96GB each), Docker dev container, Python 3.11, PyTorch 2.x, MONAI, wandb.
- Convention: `CUDA_VISIBLE_DEVICES=0` (멀티-GPU는 명시적으로만).
- 데이터: 3D MAISI latent streaming I/O가 종종 bottleneck (3d_latent_io_bottleneck.md memory 참고).

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| CT-RATE | dataset | train_fixed (20k pat, 47k scan), valid_fixed (1.3k pat, 3k scan), labels, reports | source of all volumes; reports are text condition |
| MAISI VAE | encoder | latent shape [4,120,120,64] fp16, sliding-window | encodes CT → latent; frozen in our pipeline |
| Report2CT | baseline (P1) | MAISI VAE-GAN + multi-encoder LDM | 1순위 비교 대상; pretrained ckpt 사용 |
| GenerateCT | baseline (P2) | CT-ViT + MaskGIT + SR diffusion | 시간순 baseline; 자체 autoencoder |
| TRACE | baseline (optional) | 2D slice diffusion + flow/seg/text | 다운만, 비교 안 할 수도 |
| VLM3D-Dockers | eval infra | FVD + CLIPScore + 2.5D-FID, docker subprocess | submission deliverable 형태와 동일 |
| lightning-hydra-template | code structure | configs hierarchy, LightningModule, hydra.utils.instantiate | 새 repo base |
| Submission docker | hard deliverable | Dockerfile + test.sh + export.sh, metrics.json output | Phase D 산출물 |
| Cross-attention heatmap | diagnostic | per-token spatial map, axial/sagittal/coronal overlay | diagnosis 산출물 1 |
| CLIP retrieval recall | diagnostic | R@1/R@5/R@10 via CT-CLIP | diagnosis 산출물 2 |
| Counterfactual pair | diagnostic | report_A vs report_A_minus_finding → diff map | diagnosis 산출물 3 |
| Token-region attribution | diagnostic | cross-attn × anatomy seg IoU/Dice | diagnosis 산출물 4 |
| TotalSegmentator / VISTA-3D | tool | anatomy seg | diagnostic 4의 의존성 |
| CT-CLIP | model | CT-RATE에서 학습된 vision-language model | diagnostic 2의 의존성 |
| third_party/ | code area | git submodules | 외부 repo 통합 패턴 |
| Adapter layer | code pattern | LightningModule wrapper over submodule | src/baselines/ pattern |
| deprecated/ | code area | 옛 triplane 코드 일체 | mv 대상 |
| Phase A-D | timeline | 5/26-5/31, 6월, 7월, 8/1-8/20 | challenge deadline 분할 |
| MICCAI 2026 VLM3D Task 4 | challenge | 마감 2026-08-20 | hard end date |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|--------------|-----|---------|--------|-----------------|
| 1 | 10 | 10 | — | — | N/A |
| 2 | 15 | 5 | 0 | 10 | 67% |
| 3 | 17 | 2 | 0 | 15 | 88% |
| 4 | 18 | 1 | 0 | 17 | 94% |
| 5 | 20 | 2 | 0 | 18 | 90% |
| 6 | 22 | 2 | 0 | 20 | 91% |
| 7 | 24 | 2 | 0 | 22 | 92% |
| 8 | 25 | 1 | 0 | 24 | 96% |

수렴 (>90% 3연속 round). 새 추가는 모두 외부 라이브러리(CT-CLIP, TotalSegmentator)와 코드 패턴(adapter, submodule)이고, 핵심 도메인 entity(CT-RATE, MAISI VAE, Report2CT, GenerateCT, VLM3D-Dockers)는 Round 1부터 안정.

## Interview Transcript

<details>
<summary>Full Q&A (7 rounds + Round 0)</summary>

### Round 0 (Topology Confirmation)
**Q:** 5개 top-level component topology가 맞습니까?
**A:** 5개 그대로 맞음.
**Resolution:** Repo restructure / Dataset analysis / Baseline reproduction+diagnosis / VLM eval infra / Compute feasibility 5개 모두 active로 lock.

### Round 1 (Baseline repro Criteria)
**Q:** Report2CT/GenerateCT 재현 범위를 어디까지로?
**A:** Pretrained 가중치 inference만.
**Ambiguity:** 100% → 53%.

### Round 2 (VLM eval infra Context)
**Q:** Text-conditioning fidelity 진단 산출물 형태?
**A:** Cross-attn heatmap + CLIP retrieval + Counterfactual + Token-region attribution (4종 모두).
**Ambiguity:** 53% → 53% (다른 component가 더 빠르게 가까워짐).

### Round 3 (Dataset analysis Criteria)
**Q:** CT-RATE 분석 완료 시점?
**A:** Minimal exploratory (HU 분포, spacing/dim, label, report-len, KS-drift).
**Ambiguity:** 53% → 43%.

### Round 4 (Contrarian — Compute Constraints)
**Q:** Deadline/budget shape은? (compute가 wall-clock에 끌려가는가 contrarian probe)
**A:** VLM3D 2026 challenge 참가 (특정 날짜).
**Ambiguity:** 43% → 37%.

### Round 5 (Compute Constraints — date)
**Q:** 마감일 대략?
**A:** 2026-08-20, baseline reproduction + 문제 분석은 5월 중.
**Ambiguity:** 37% → 14% (잠시) but threshold 재계산하니 다른 component bottleneck 잔존.

### Round 6 (Simplifier — Repo restructure)
**Q:** Repo restructure done 정의? 5일 phase A 안에서?
**A:** Full migration in 5 days (schedule slip risk 수용).
**Ambiguity:** 14% → 31% (재평가에 따라 다른 component context가 약점 노출).

### Round 7 (Baseline repro ∩ VLM eval infra Context)
**Q:** 외부 repo 통합 전략?
**A:** third_party/ submodule + src/ adapter 레이어.
**Ambiguity:** 31% → 18% — PASSED (잠시).

### Round 8 (Baseline repro Constraints — post-Round-7 weight check)
**사실 검증:** GenerateCT 3종 weight 공개 ✅, Report2CT weight 미공개 ❌ — Round 1 결정 부분 무효화.
**Q:** Report2CT 대응? (재구현+학습 / 저자 contact / 제외 / 다른 baseline 대체)
**A:** 논문 기반 직접 구현 + CT-RATE 학습.
**Ambiguity:** 18% → 22% — **PASSED with Phase B compute-measurement gate**.

</details>

## Next Steps (post-spec)

1. **즉시 (오늘 5/26)**: 본 spec 사용자 승인 → omc-plan consensus 또는 직접 executor로 Phase A 착수.
2. **Phase A 일정 (제안, Round 8 반영)**:
   - Day 1 (5/26): repo restructure 골격 (`src/`, `configs/`, `third_party/`, `deprecated/`), submodule 등록, pytest 골격 잡기. triplane code → `deprecated/` mv.
   - Day 2 (5/27): lightning-hydra-template 패턴으로 base LightningModule/DataModule 스켈레톤 + Hydra entrypoints. 기존 활용 가능한 util(MAISI loader, metrics 일부) 신규 src/로 이동 + 테스트.
   - Day 3 (5/28): GenerateCT adapter (`src/baselines/generatect_adapter.py`) + 3종 pretrained ckpt 다운로드 + 1-sample text→volume inference 통과.
   - Day 4 (5/29): GenerateCT valid 50 샘플 batch inference + EDA notebook 초안 (HU/spacing/label/report-len).
   - Day 5 (5/30-5/31): VLM3D-Dockers subprocess 래퍼 + valid 50-100 샘플로 GenerateCT FVD/CLIPScore/2.5D-FID 첫 측정. EDA notebook 완성 + KS-test 추가. Report2CT 논문 정독 + impl 스펙 draft.
3. **Phase B kickoff (6/1)**:
   - Compute-measurement gate: Report2CT 골격 코드 + 100 샘플 1-epoch wall-clock 실측 → 전체 학습 cost 추정.
   - 4종 진단 구현 시작 (먼저 GenerateCT 대상으로 implementation 검증).
   - 우리 모델 v1 설계 회의 (현 시점까지 발견된 GenerateCT 약점 반영).

## Memory Candidates (post-interview)

저장 권장:
- **project memory**: "MICCAI VLM3D 2026 Task 4 submission deadline = 2026-08-20. Baseline reproduction + diagnosis는 5월 중 완료 (5일 phase A). Win condition = ours > pretrained Report2CT on FVD/CLIPScore/2.5D-FID via VLM3D-Dockers." Why: 모든 phase 결정의 spine. How to apply: 이후 모든 작업 우선순위 판단에서 이 마감/win condition을 reference.
- **feedback memory**: "사용자는 코드 품질을 일정 리스크보다 우선시함 (Round 6에서 full migration in 5 days 선택, schedule slip 수용)." Why: messy code에 대한 불만이 명시적이었음 + 깨끗한 base가 challenge 후반 phase의 속도를 좌우. How to apply: 'quick hack' vs 'clean structure' 선택 시 후자 default.
- **project memory**: "Triplane 잠정 폐기. 기존 triplane 코드/runs/configs는 deprecated/로 이동. 이후 비교 대상 아님." Why: project pivot이므로 다음 세션에서도 혼동 없도록. How to apply: 'triplane' 키워드가 사용자 요청에 나오면 deprecated 맥락 확인부터.
- **reference memory**: "third_party/{report2ct, generatect, vlm3d_dockers}/ git submodule + src/baselines/, src/eval/ adapter pattern." Why: 외부 코드 통합 정책. How to apply: 이후 외부 repo 추가 시 같은 패턴 유지.
