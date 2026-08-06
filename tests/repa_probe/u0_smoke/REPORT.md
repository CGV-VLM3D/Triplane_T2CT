# U0 스모크 — SPECTRE dense token @ wan 그리드

실행: `CUDA_VISIBLE_DEVICES=2 python -m tests.repa_probe.u0_smoke.run`
(2026-07-28, `valid_1000_a_1`, RTX PRO 6000 Blackwell, spectre 0.2.1)
원자료: [results/smoke_spectre.json](results/smoke_spectre.json) · 그림: [figs/S3_cossim_map_ssl.png](figs/S3_cossim_map_ssl.png)

## 결론: **6/6 통과.** teacher feature를 wan latent와 voxel 단위로 정합된 32³ 그리드에서 뽑을 수 있다.

| # | 검증 | 결과 |
|---|---|---|
| S0 | `crops_to_grid`가 `grid_patch`를 정확히 되돌리는가 (모델 없는 인덱스 왕복) | **exact, max_abs_err 0.0** |
| S1 | z end-pad 후 center-crop으로 잘린 voxel | **0** (253→256, crop grid `(4,4,4)`, crops `(64,1,128,128,64)`, 값 `[0,1]`) |
| S2 | crop당 token / prefix token | **`(64, 513, 1080)`**, `num_prefix_tokens == 1`, backbone 338.5M |
| S3 | CLS 제거 후 32³ 재조립이 해부학적으로 맞는가 | **SRSS +0.150** (lung-lung cos 0.353 vs lung-other 0.203), 폐 마스크와 육안 일치 |
| S4 | `forward_intermediates` 중간 layer 선택 | depth 24, layer `[11,17,23]` 모두 `(N, 512, 1080)` |
| S5 | combiner scan-level global | combiner 출력 `(65, 1080)` = CLS 1 + crop token 64 → 어댑터가 index 0을 꺼내 **`(1080,)`** |
| S6 | 비용 | load+resample **7.0 s**, backbone forward **1.89 s**, peak GPU **2.72 GB** |

## 기하 정합이 확정됐다

```
CT --(wan과 동일 transform)--> (1, 512, 512, 253)  raw HU
   --(z end-pad, air=-1000)--> (1, 512, 512, 256)
   --window_scan(128,128,64)-> crops (64, 1, 128, 128, 64) + grid (4,4,4)   [HU→[0,1] 내부 처리]
   --backbone--------------->  (64, 513, 1080)
   --tokens[:, 1:]---------->  (64, 512, 1080)
   --crops_to_grid---------->  (32, 32, 32, 1080)        ← wan latent 64³의 정확히 2배 거친 격자
```
S0의 인덱스 왕복이 **모델 없이** 순서를 증명한다: crop 인덱스 `n = (h·n_w + w)·n_d + d`, crop 내부 patch도
`(H, W, D)` C-order (rope patch_embed `output_fmt='NHWDC'` → `_pos_embed`의 `x.view(B, -1, C)`). 둘 다 depth가 가장 빨리 변한다.

**z end-pad는 필수다.** 안 하면 `largest_multiple_center_crop`이 `253 // 64 = 3` → 192로 center-crop해서
양 끝 61 slice(24 %)가 조용히 사라진다. 위 표의 `dropped_voxels == 0`이 그 방지선.

## U3로 넘길 발견 2개 (그림에서 바로 보인다)

1. **crop seam이 실제로 보인다.** axial cos-sim map의 y≈16 지점과 sagittal map의 x≈24 지점에 수평/수직
   불연속이 있다 — 128-voxel crop 경계다. SPECTRE의 self-attention이 crop 내부에서만 일어나기 때문에
   당연한 결과이고, 계획의 리스크 표에 적어둔 항목이 확인된 것. U3에서 심각도를 정량화하고,
   필요하면 50 % overlap window + 평균(비용 4×)을 검토한다.
2. **global 성분이 강하다 → iREPA spatial norm이 도움될 가능성이 높다.** cos 값이 거의 전 영역에서
   0.2–0.5이고 음수나 0 근처가 없다. iREPA가 "patch token에 상당한 global 성분이 있어 무관한 토큰끼리도
   꽤 높은 유사도를 보인다"고 지적한 바로 그 패턴이다. SRSS가 +0.150에 머무는 것도 이것과 일관된다.
   U3에서 spatial norm 유/무를 같은 축에서 비교한다.

## 처리량 / 저장 예산 (U2 착수 근거)

- 볼륨당: load+resample 7.0 s + backbone 1.9 s. **teacher 2종을 한 패스로 뽑으면** load를 공유하므로 ≈ 11 s/volume.
- 단일 프로세스로 6,304 scan ≈ **19 h**. peak GPU가 2.72 GB뿐이므로 병목은 GPU가 아니라 **CPU(gzip 해제 + resample)**.
  `--num-shards`로 3–6 프로세스를 한 GPU에 띄우면 **≈ 4–7 h**로 줄어든다 (wan precompute와 같은 전략).
- 저장: scan당 32³ **70.8 MB** + 16³ **8.8 MB** (fp16) → teacher 2종 × 6,304 scan ≈ **1.0 TB** (`/workspace` 여유 5.7 TB).

## 환경 메모

- `spectre`를 import하려면 **`huggingface_hub.load_state_dict_from_file` shim**이 필요하다
  (`spectre/utils/modeling.py:14`가 module-level import; 우리 env는 0.26.3, 그 심볼은 0.30+).
  실제로는 **HF-URL 분기(`modeling.py:77`)에서만 호출**되고 우리는 항상 로컬 `.pt` 경로라 호출되지 않는다 →
  `_spectre.py:install_hf_shim()`. `huggingface_hub`를 올리지 않는다(transformers 4.46 / diffusers 0.31 의존).
- `loralib`을 설치했다 (`spectre/utils/__init__.py`가 `.lora`를 import; spectre-fm의 선언된 의존성).
- 체크포인트는 `strict` 로드에서 **All keys matched successfully** — 아키텍처 불일치 없음.
- 두 backbone 모두 `SpectreImageFeatureExtractor(backbone_name=..., backbone_checkpoint_path_or_url=...)`로
  경로 지정 로드. `presets.py`에는 VLA URL만 등록돼 있어 SSL(`_no_vla`)은 이 경로가 유일한 방법.

## 후속

이 스모크의 로직은 U1에서 [src/baselines/spectre_adapter.py](../../../src/baselines/spectre_adapter.py)
(`SpectreBackbone`)로 승격됐고, 위 수치는 어댑터 경유로 **동일하게 재현**된다(SRSS 0.15026 일치).
결정적 검사 3개(인덱스 왕복 / CLS 미제거 거부 / z=253 center-crop 거부)는
[tests/test_spectre_adapter.py](../../test_spectre_adapter.py)에 회귀 테스트로 고정됐다.
