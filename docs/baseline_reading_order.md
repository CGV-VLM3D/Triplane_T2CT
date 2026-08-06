# Baseline 코드 읽기 순서 가이드

목적: baseline(결과 확인용) 코드를 **적당한 깊이**로 읽기 위한 순서. 각 baseline에서
보는 것은 딱 세 가지 — (1) **encoder / 모델 정의**, (2) **생성·추론 파이프라인**,
(3) **데이터 전처리 파이프라인**. 라인 단위 정독은 불필요.

> **fVLM 제외** — 이미 정독 완료(실연구용). 입력 전처리는 `src/baselines/fvlm_preprocess.py`.

## 권장 순서 (의존성 기준) — 사용자가 적은 순서와 다름

사용자가 적은 순서는 `ctclip → report2ct → text2ct → maisi` 였지만, **MAISI를 먼저**
읽길 권합니다. 이유: Report2CT와 Text2CT **둘 다 MAISI VAE 잠재공간에서 디퓨전**을 돌고
MAISI 디코더로 CT를 복원합니다. 잠재공간 모양(`[B,4,120,120,64]` ↔ `[B,1,480,480,256]`)을
먼저 이해해야 두 생성기가 읽힙니다. CT-CLIP은 두 생성기·세 eval 도커가 공유하는 인코더라
그 다음입니다.

**MAISI → CT-CLIP → Report2CT → Text2CT**

---

## 1. MAISI VAE (공유 잠재공간 — 먼저)

- **모델 정의**: `src/baselines/maisi.py` — `load_frozen()`이 전부. 아키텍처 kwargs를
  재선언하지 않고 `third_party/maisi_bundle/configs/inference.json`의 `autoencoder_def`를
  `monai.bundle.ConfigParser`로 인스턴스화 → 모든 파라미터 `requires_grad=False`.
  `is_fully_frozen()`로 동결 검증.
- **생성·추론**: VAE는 생성기가 아니라 **인코더/디코더**. encode: CT → `[4,120,120,64]`,
  decode: 잠재 → `[1,480,480,256]`. 큰 볼륨 디코드는 sliding-window가 필요할 수 있음
  (eval 샘플러가 `SlidingWindowInferer`로 처리 — 아래 Report2CT 샘플러 참고).
- **전처리**: HU `[-1000,1000]` clip → `[0,1]`, 공간 480×480×256. upper-bound 수치는
  `results/upper_bound.json` (PSNR 30.94).
- **읽을 것**: `maisi.py` 1개 파일 + inference.json의 `autoencoder_def` 블록만.
  나머지 bundle 내부는 skip.

## 2. CT-CLIP (공유 인코더)

- **encoder 정의**: `src/baselines/ctclip_adapter.py` — `CTCLIPBackbone`. 핵심 메서드
  `encode_image(vol) -> (B,512)`, `encode_text(ids,mask) -> (B,512)`, `tokenize(text)`.
  contrastive forward를 미러: `visual_transformer(..., return_encoded_tokens=True)` →
  temporal mean → flatten → `to_visual_latent` + L2 / 텍스트는 `last_hidden_state[:,0]` →
  `to_text_latent` + L2.
- **upstream**: `third_party/ct_clip/CT_CLIP/ct_clip/ct_clip.py` — forward 계약만 확인
  (내부 transformer 구현은 skip).
- **전처리**: 어댑터 입력은 `(1,1,240,480,480)`, 강도 HU/1000 ≈ `[-1,1]`. upstream
  전처리 레퍼런스는 `third_party/ct_clip/scripts/data_inference_nii.py` (resample
  1.5/0.75/0.75 → clip → /1000 → center pad/crop). *주의: 우리 쪽 노트북 전처리
  `load_ct_for_ctclip`는 viz 정리 때 삭제됨 — 필요하면 upstream 스크립트가 단일 출처.*
- **config**: `configs/model/vlm_backbone/ctclip.yaml`.
- **읽을 것**: 어댑터 1개 + upstream forward 계약. CT-CLIP은 마스크 불필요(볼륨+텍스트만).

## 3. Report2CT (생성기 #1 — MAISI 잠재 디퓨전)

- **모델 정의 (LIVE)**: `src/models/report2ct_module.py` — `Report2CTModule`
  (LightningModule). config `configs/model/report2ct.yaml`의 `_target_`가 가리키는 진짜
  경로. UNet은 `monai...DiffusionModelUNetMaisi`(233M), 스케줄러는
  `src/baselines/rflow.RFlowScheduler`. UNet/scheduler 모두 코드로 새로 안 짜고
    `report2ct.yaml`의 `_target_`로 MONAI 클래스를 instantiate하며, `config_maisi_2560.json`과의
    1:1 패리티 + bit-exact forward는 `tests/test_report2ct_parity.py`가 강제한다.
- **encoder**: `src/baselines/report2ct_text_encoder.py` (3개 텍스트 인코더 concat,
  2560-d 컨디셔닝) + `src/baselines/report2ct_image_encoder.py` (CT → MAISI 잠재).
- **생성·추론**: `src/eval/samplers/report2ct.py` — 학습된 Lightning 체크포인트 로드,
  RFlow denoising 루프, MAISI 디코드(`SlidingWindowInferer`), HU 저장. **audit checklist
  #2(샘플러 루프 `→0` 종료)·#5(타일 디코더)·#7(bf16 autocast)** 가 여기서 중요.
- **전처리·데이터**: `src/data/report2ct_datamodule.py` + 사전계산 스크립트
  `scripts/precompute_report2ct_{image,text}_embeddings.py`. 전체 48k 잠재 set 사용
  (CLAUDE.md Dataset reference 참고).
- **읽을 것**: module → text/image encoder → sampler 순. upstream JSON
  `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json`의 `diffusion_unet_def`는
  참고만.

## 4. Text2CT (생성기 #2 — MAISI 잠재 rectified-flow)

- **모델 정의**: `src/baselines/text2ct_adapter.py` — `Text2CTAdapter`. FrozenCLIP3D
  텍스트 컨디셔닝 + rectified-flow 샘플러. inference-only (upstream
  `scripts.diff_model_demo.run_inference` 재사용).
- **스케줄러**: `src/baselines/_vendored/rectified_flow.py` — **실제 vendored RFlow
  스케줄러**(MONAI ≥1.5 전용이라 1.4 핀 환경에 vendor). `src/baselines/rflow.py`는
  헷갈리지 말 것 — 그건 MAISI 번들 스케줄러를 re-export 하는 **별개의 shim**.
- **생성·추론**: `src/eval/samplers/text2ct.py` (`Text2CTSampler`). 샘플 생성은
  `scripts/generate_text2ct_valid.py`.
- **전처리·데이터**: Report2CT와 같은 MAISI 잠재 family. 출력 spacing 규약 유의
  (text2ct 0.75/0.75/3.0 — `ctgen-eval-spacing-convention` 메모리).
- **config**: `configs/model/text2ct.yaml`. upstream: `third_party/text2ct/scripts/diff_model_demo.py`.

---

## Eval / 결과 확인 진입점 (공통)

baseline "결과 확인"은 여기서 돈다:

- `src/eval.py` — Hydra `@main` eval 엔트리.
- `src/eval/samplers/` — `base.py`(추상) + `report2ct.py` / `text2ct.py` / `generatect.py`.
- `src/eval/tasks/ctgen.py` + `_fid_runner.py` — 2.5D-FID / CLIPScore / FVD (docker 스크립트 직접 호출).
- `scripts/run_eval.py` — 실제 실행 엔트리포인트(Hydra).

## 한눈에 보기

| baseline | 모델 정의 | 추론/샘플러 | 전처리·데이터 | config |
|---|---|---|---|---|
| MAISI | `src/baselines/maisi.py` | (enc/dec) | HU clip→[0,1], 480³ | inference.json |
| CT-CLIP | `src/baselines/ctclip_adapter.py` | encode_image/text | data_inference_nii.py | vlm_backbone/ctclip.yaml |
| Report2CT | `src/models/report2ct_module.py` | `src/eval/samplers/report2ct.py` | report2ct_datamodule.py + precompute_* | model/report2ct.yaml |
| Text2CT | `src/baselines/text2ct_adapter.py` | `src/eval/samplers/text2ct.py` | MAISI 잠재 family | model/text2ct.yaml |

> 베이스라인을 읽을 때 silent-bug 체크리스트(활성 범위 핸드오프 / 샘플러 `→0` /
> config-vs-hardcode / 축 순서 / 타일 디코더 / HU 저장 / mixed precision / class-label)는
> CLAUDE.md 끝의 **"Baseline / model-clone audit checklist"** 참고.
