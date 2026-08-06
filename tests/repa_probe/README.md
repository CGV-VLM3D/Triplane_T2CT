# `tests/repa_probe/` — REPA(U-REPA/iREPA/HASTE/VideoREPA) 도입 검증 스터디

`report2ct_wan`에 representation alignment를 붙이기 전에 **"정말 되는가"를 측정으로 판정**하는
self-contained 실험 모음. 계획: `/root/.claude/plans/workspace-paper-pdf-cozy-shore.md`
런북: [docs/repa_runbook.md](../../docs/repa_runbook.md)

**격리 원칙**: 여기 코드는 `src/`·`configs/`를 건드리지 않는다. 검증된 결론만 본 프로젝트로 넘어간다.

## 한 줄 결론

**GO.** frozen UNet + projector만으로 held-out cosine +0.56에 도달한다(U-REPA가 20만 스텝 학습 후
보고한 0.60에 근접). 다만 그 cosine의 **상당 부분은 해부학적 위치 prior**이므로 학습 중에는
`repa_cos`가 아니라 **`repa_cos_gap`**(= 실제 teacher − 섞은 teacher)을 봐야 한다.

## 구조

```
tests/repa_probe/
├── README.md          ← 이 파일 (인덱스 + 결론)
├── _spectre.py        ← 공유: CT/마스크를 wan 그리드로 올리기, 장기·몸통 점유율
├── _metrics.py        ← 공유: LDS / CDS / SRSS / RMSC / crop seam / spatial norm (iREPA Appendix B)
├── u0_smoke/          ← U0  SPECTRE dense token이 wan 그리드에서 나오는가 (기하 정합 증명)
├── u2b_io/            ← U2b teacher 추출 정밀도 + 학습 스텝 오버헤드 → 기본 arm 확정
├── u3_teacher/        ← U3  teacher 적합성: SSL vs VLA, layer, spatial norm, crop seam
├── u4_align/          ← U4  student↔teacher 정렬 가능성 (go/no-go): MLP vs Conv, tap, 해상도, hard vs soft
├── u5_overfit/        ← U5  과적합 + 대조군: 정렬이 실제 대응에서 오는가
├── u6_teachers/       ← U6  teacher 후보 비교: cos-sim 볼륨 · organ×organ 분리 · spatial norm
├── u7_monitor/        ← U7  실제 학습 중 loss/cos/gap/grad 모니터링 (wandb run에서 생성)
├── u8_semantic/       ← U8  표현의 의미적 품질: linear probe / k-NN probe / CKA (REPA §3의 CT판)
└── u9_haste/          ← U9  HASTE 조기종료 진단: 학습 완료된 repa300 체크포인트에서
                            grad(L_diff)·grad(L_repa) 코사인이 후반부에 반전되는지
```

각 단위 폴더는 같은 레이아웃이다: `run.py`(실행) · `REPORT.md`(판정) · `results/*.json` · `figs/*.png`.

## 진행 상황과 결론

| 단위 | 상태 | 한 줄 결론 |
|---|---|---|
| [U0 smoke](u0_smoke/REPORT.md) | ✅ | **6/6.** teacher 32³ grid가 wan latent 64³와 voxel 단위 정합. `window_scan`이 pad가 아니라 **center-crop**이라 z를 253→256으로 먼저 채워야 61 slice가 안 사라진다 |
| [U2b I/O](u2b_io/REPORT.md) | ✅ | teacher 추출은 **TF32**(3.7× 빠르고 cos_min 0.99998). 학습 오버헤드 **16³ +2.5 %** vs 32³ +54.7 % → **16³ 확정** |
| [U3 teacher](u3_teacher/REPORT.md) | ✅ | **spatial norm ON**(12/12 arm 우세), **layer 23**, **SSL 먼저**(VLA 우위 증거 없음). crop seam 실재 — 최종 레이어에서 이웃 cosine **22.9 % 낙차** |
| [U4 align](u4_align/REPORT.md) | ✅ | **GO** (held-out cos +0.5625). **MLP > Conv**(iREPA 주장 미재현), **16³가 32³보다 정렬이 쉽다**, 관계형만으로는 cosine이 **정확히 0** |
| [U5 overfit](u5_overfit/REPORT.md) | ✅ | 파이프라인은 정렬을 학습한다(gaussian 대조군 cos 0 유지). 단 **shuffled 대조군이 0.45** — cosine의 대부분이 위치 prior |
| [U6 teachers](u6_teachers/REPORT.md) | ✅ | 후보 **9종 × 20 스캔 완료**(MedSAM2만 보류). 거리 통제 `SRSSdm`이 순위를 뒤집는다 — CT-FM은 SRSS 1위/SRSSdm 6위(매끄러움), **CT-CLIP 부적합**, RadFinder≈SPECTRE(승률 50 %), fVLM 1위지만 반경 교란 미해결. 학습 없는 HU 바닥값이 전 teacher를 이기므로 SRSSdm은 **상대 비교로만** |
| [U7 monitor](u7_monitor/) | ✅ | **실제 학습이 probe 예측을 검증**: cos 0.604(U4 상한 0.5625 초과), shuffled 0.429(U5 예측 0.45), gap 0.174 |
| [U8 semantic](u8_semantic/REPORT.md) | ✅ | **epoch 커브(11 ckpt × context ON/OFF, 고정 분할)**: baseline은 300 ep 내내 semantic gap을 못 줄이는데(linear 0.63~0.66 평평) **REPA는 ep009에 메운다**(k-NN 0.670 vs teacher 0.675, CKA 0.83→0.89 포화). baseline CKA는 U자(최저 ep079 = FID 바닥). context가 CKA를 끌어내리는 폭이 학습에 따라 소멸하는데 **REPA는 처음부터 없다**. 정렬은 tap에 국소적 |
| [U9 haste](u9_haste/REPORT.md) | ✅ | **HASTE 기각.** repa300(300ep, `repa_stop_step=null`) 30개 체크포인트 전부에서 grad(L_diff)·grad(L_repa) 코사인이 +0.02~+0.16 근방에서 유지, 후반(ep159-299) 평균이 전반(ep9-149)보다 오히려 근소 높음(0.073 vs 0.065). N=8→16+반복3+체크포인트 간 common-random-numbers로 노이즈를 줄인 재확인에서도 동일 — 후반부 conflict 반전은 이 학습에서 재현되지 않는다 |

## 이 스터디가 바꾼 기본값 (논문 그대로가 아니다)

| 항목 | 논문 | 우리 기본값 | 근거 |
|---|---|---|---|
| projector | conv k=3 (iREPA) | **mlp** | U4: cos 0.5625 vs 0.5303, 파라미터 1/5 |
| teacher 해상도 | ViT native (U-REPA Table 6) | **16³ pooled** | U4: 32³가 오히려 정렬이 어렵다(0.4714) + U2b: I/O 8× |
| teacher spatial norm | iREPA 제안 | **ON** | U3: 4개 지표 12/12 arm 우세 |
| 모니터링 지표 | `repa_cos` | **`repa_cos_gap`** | U5: shuffled가 0.45 → raw cosine은 정렬을 과대평가 |
| 정렬 손실 조기 종료(`repa_stop_step`) | HASTE는 채택 권장 | **미채택(`null` 유지)** | U9: 300 epoch 전 구간에서 grad conflict 누적 없음 — 끊어낼 "이전엔 괜찮았던" 지점 자체가 없다 |

## 재현

```bash
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u0_smoke.run
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u2b_io.precision
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u2b_io.steptime --steps 40 --num-workers 12
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u3_teacher.run --n-volumes 24
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u4_align.run --n-volumes 48 --steps 1500
CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u5_overfit.run --steps 200
CUDA_VISIBLE_DEVICES=2 python -m tests.repa_probe.u6_teachers.run
python -m tests.repa_probe.u7_monitor.run        # 학습 중 아무 때나 (wandb run 파싱)
CUDA_VISIBLE_DEVICES=1 python -m tests.repa_probe.u9_haste.run   # 학습된 체크포인트로 사후 진단 (재학습 불필요)
```

`u2b_io/steptime.py`와 `u5_overfit/run.py`는 임시 datalist를 만들어 쓰고 끝나면 지운다.

## 공유 헬퍼

**`_spectre.py`** — 모델 쪽은 전부 [src/baselines/spectre_adapter.py](../../src/baselines/spectre_adapter.py)에 있고, 여기는 데이터 접근만 담당한다.

| 함수 | 역할 |
|---|---|
| `load_volume(scan_id, mask=, pad=)` | CT / ts_seg 마스크를 **wan precompute와 동일한** transform으로 `(1,512,512,256)` (z end-pad 포함) |
| `organ_occupancy(scan_id, organs)` | 장기별 마스크를 teacher token grid로 pool → `(32,32,32)` 점유율 |
| `body_occupancy(scan_id)` | HU > −500 몸통 점유율 — CT는 절반이 공기라 지표를 부풀린다 |
| `ts_labels_with_prefix(prefix)` | 레이블 id를 `third_party/fvlm/data/resize.py` class_map에서 파싱 (손으로 나열 금지) |
| `build_backbone(ckpt, ...)` | frozen SPECTRE 인코더 (어댑터 그대로) |

**`_metrics.py`** — iREPA §4 / Appendix B의 공간 구조 지표를 3D로 옮긴 것.

| 함수 | 역할 |
|---|---|
| `spatial_norm(x, gamma)` | 토큰축 평균 제거 + 분산 정규화 (teacher에만) |
| `lds / cds / srss / rmsc` | Local-vs-Distant Similarity · Correlation Decay Slope · Semantic-Region Self-Similarity · RMS Spatial Contrast |
| `crop_seam(tokens, grid)` | crop 경계 이웃 vs 내부 이웃 cosine (`within` / `across` / `gap` / `drop`) |
| `all_metrics(...)` | 위를 한 번에, `subset=`으로 몸통 토큰만 한정 가능 |
