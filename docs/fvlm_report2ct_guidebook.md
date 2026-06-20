# fVLM & Report2CT 가이드북 — I/O와 코드 리딩 맵

> **목적**: 두 모델의 **입출력 텐서(shape/dtype)** 와 **"어느 파일을 어떤 순서로 읽을지"** 를 한 파일로 빠르게 파악한다.
> 본문 설명은 한국어, 코드·경로·shape는 원문 유지. 모든 shape/라인은 현재 코드에 직접 대조해 기재함(2026-06-03 기준).
> 학습 절차·다운로드 레시피 같은 운영 문서는 §5 인덱스에서 포인터로 연결한다.

상위 문서: 프로젝트 전체 투어는 [GUIDE.md](../GUIDE.md), 규칙·아키텍처는 [CLAUDE.md](../CLAUDE.md).

---

## 0. 30초 요약

| | **Report2CT** | **fVLM** |
|---|---|---|
| 방향 | **text → 3D CT 생성** (Task 4 / ctgen) | **CT → per-organ feature** (VLM 백본, abnclass·정렬용) |
| 한 줄 | report+spacing 조건의 **latent diffusion** | **anatomy-aware** BLIP, 장기별 대조정렬 |
| 입력 | report(findings+impression) + voxel spacing + (학습 시) CT latent | CT volume **+ organ segmentation mask** |
| 출력 | CT volume `(B,1,480,480,256)` (추론) / velocity `(B,4,120,120,64)` (학습 1-step) | 장기별 image feature `(1,256)` + 정상/비정상 logits `(1,2)` |
| 핵심 클래스 | `Report2CTModule` (실사용) / `build_unet` (shape 참고) | `FVLMBackbone` → `.model` (`BlipPretrain`) |
| 가중치 | **미공개** (사용자가 직접 학습) | gdrive 공개 ckpt (`pretrained_ct_rate.pt`) |
| nn.Module? | ✅ LightningModule | ❌ plain class (`.model`이 실제 Module) |
| latent 공간 | **MAISI VAE** `(4,120,120,64)` 사용 | **자체 ViT** 인코더 (MAISI 안 씀) |
| 우리 어댑터 | [src/models/report2ct_module.py](../src/models/report2ct_module.py) | [src/baselines/fvlm_adapter.py](../src/baselines/fvlm_adapter.py) |

> 가장 큰 함정 두 가지를 먼저 박아둔다:
> 1. **Report2CT**: 실 학습/추론 경로는 [report2ct_module.py](../src/models/report2ct_module.py)뿐이다. UNet/scheduler는 코드로 새로 안 짜고 [configs/model/report2ct.yaml](../configs/model/report2ct.yaml)의 `_target_`로 instantiate하며, `config_maisi_2560.json`과의 패리티는 [test_report2ct_parity.py](../tests/test_report2ct_parity.py)가 강제한다. *(옛 `report2ct_adapter.py` skeleton은 2026-06-09 삭제 — parity 테스트와 완전 중복.)*
> 2. **fVLM**: CT-RATE에는 **organ mask가 없다.** TotalSegmentator로 미리 만들어야 forward가 돈다(§2.3). 아직 미생성 상태.

---

## 1. Report2CT — text → 3D CT 생성

### 1.1 한눈에 보는 파이프라인

```
findings / impression (text)        voxel spacing (mm)        CT volume (학습 시 GT)
        │                                  │                        │
        ▼ 3×BERT 풀링→concat               ▼ ×100                   ▼ MAISI VAE encode (frozen)
   context (B,2,2560)              spacing_tensor (B,3)        latent (B,4,120,120,64)
        │                                  │                        │  + noise (RFlow)
        └──────────────┬───────────────────┘                       ▼
                       ▼                                     noisy_latent (B,4,120,120,64)
        DiffusionModelUNetMaisi (233M, cross-attn on context, class_labels=1=CT)
                       │
                       ▼ (학습) velocity 예측 → MSE(output, images−noise)
                       ▼ (추론) RFlow 샘플링 루프 → MAISI VAE decode
                                                  → CT volume (B,1,480,480,256)
```

- **텍스트 조건 2560-d** = `MedEmbed-large`(1024) + `ClinicalBERT`(768) + `BiomedVLP-CXR-BERT`(768), 각 모델 mean-pooling 후 concat. findings·impression 각각 `(2560,)` → `(B,2,2560)`.
- **latent diffusion**: 픽셀이 아니라 **MAISI VAE latent** `(4,120,120,64)` 위에서 denoise. scheduler는 `RFlowScheduler`(rectified flow, V_PREDICTION).
- **추론 디코드 경로는 아직 미구현** — 학습 모듈만 존재(§1.3).

### 1.2 I/O 텐서 레퍼런스

| 단계 | 입력 | 출력 | 코드 |
|---|---|---|---|
| 텍스트 인코딩 | `findings:str`, `impression:str` | 각 `(2560,)` float32 → 배치 `(B,2,2560)` | [report2ct_text_encoder.py:72](../src/baselines/report2ct_text_encoder.py#L72) `encode` / [report2ct_module.py:89](../src/models/report2ct_module.py#L89) `_prepare_context` |
| 이미지 인코딩 | CT `(B,1,480,480,256)`, HU clip[-1000,1000]→[0,1] | latent `(B,4,120,120,64)` | [maisi.py:27](../src/baselines/maisi.py#L27) `load_frozen` + [report2ct_image_encoder.py](../src/baselines/report2ct_image_encoder.py) |
| UNet 1-step | `x:(B,4,120,120,64)`, `timesteps:(B,)`, `context:(B,2,2560)`, `spacing_tensor:(B,3)`, `class_labels:(B,)`=1 | velocity `(B,4,120,120,64)` | [report2ct_module.py:139](../src/models/report2ct_module.py#L139) `unet_inputs` |
| 학습 손실 | `model_output`, `images−noise` | scalar | [report2ct_module.py:166](../src/models/report2ct_module.py#L166) `F.mse_loss` (V_PREDICTION) |
| 디코드(추론) | latent `(B,4,120,120,64)` | CT `(B,1,480,480,256)` | [maisi.py](../src/baselines/maisi.py) (`decode`, 추론 경로 미구현) |

- 상수: latent `(4,120,120,64)` `(C,H,W,D)`, context dim `2560` — [report2ct.yaml `in_channels:4` / `cross_attention_dim:2560`](../configs/model/report2ct.yaml).
- spacing은 DataModule에서 이미 ×100 되어 들어온다([report2ct.yaml `spacing_multiplier: 100.0`](../configs/data/report2ct.yaml)).
- CFG 학습 드롭: 학습 중 확률 0.15로 `context = zeros_like(context)` ([report2ct_module.py:107](../src/models/report2ct_module.py#L107)).
- scale_factor: 첫 배치 std의 역수로 latent 정규화([report2ct_module.py:77](../src/models/report2ct_module.py#L77)).

### 1.3 ⚠️ 어댑터 vs 모듈 — 헷갈리면 안 되는 부분

| 파일 | 정체 | 언제 보나 |
|---|---|---|
| [report2ct_module.py](../src/models/report2ct_module.py) | **실사용** `Report2CTModule` (LightningModule). 업스트림 학습 루프를 Lightning으로 포팅, 각 줄에 `upstream :line` 주석 | 실제 학습 forward(scale_factor, context concat, CFG, RFlow, loss) 전부 |

UNet/scheduler 정의는 둘 다 코드로 새로 짜지 않고 [configs/model/report2ct.yaml](../configs/model/report2ct.yaml)의 `_target_`로 MONAI 클래스를 인스턴스화한다. 이 YAML이 `config_maisi_2560.json`과 1:1인지는 [tests/test_report2ct_parity.py](../tests/test_report2ct_parity.py)가 강제한다.

### 1.4 읽는 순서 (리딩 맵)

1. **[configs/model/report2ct.yaml](../configs/model/report2ct.yaml)** — 한 화면에 UNet 채널/attention/cross-attn=2560/scheduler/하이퍼파라미터 전부. 전체 overview로 최적.
2. **[report2ct_module.py:99-168](../src/models/report2ct_module.py#L99-L168)** — `_shared_forward`. 실제 학습 한 스텝의 모든 것. 여기가 핵심.
3. **[report2ct_text_encoder.py:72-93](../src/baselines/report2ct_text_encoder.py#L72-L93)** — 3개 BERT mean-pool → 2560 concat.
4. **[maisi.py:27-68](../src/baselines/maisi.py#L27-L68)** + **[report2ct_image_encoder.py](../src/baselines/report2ct_image_encoder.py)** — CT ↔ latent 변환(§3).
5. (선택) **[diff_model_train_vlm3D_2560_multi_text.py](../third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py)** — 업스트림 원본. `module.py` 주석의 `upstream :line` 매핑으로 대조하며 본다.

### 1.5 학습은 user-owned

가중치 미공개이므로 멀티데이 학습은 **사용자가 직접** 돌린다(프로젝트 합의: 학습은 user-owned). 업스트림 런처는 [third_party/report2ct/train.sh](../third_party/report2ct/train.sh)(`torchrun --nproc_per_node=2 ...`). 우리 쪽 Hydra 실행 진입점은 `python src/train.py experiment=report2ct_repro`. 절차 상세는 §5의 runbook 참고.

---

## 2. fVLM — CT → per-organ feature (anatomy-aware VLM 백본)

### 2.1 한눈에 보는 구조

```
CT volume (B,1,112,256,352) ──ViT(patch 16×16×32)──► image_embeds (B,1232,768)
                                                  └► hidden_image_embeds = 4개 멀티스케일 [block3,6,9,final]
organ mask (B,1,112,256,352)  (id 1=lung,2=heart,3=esophagus,4=aorta)
        │ max_pool3d(16,16,32) → 장기별 token 선택 (1232개 중)
        ▼
 query_token[organ] ──MultiheadAttention(선택 토큰 key/value)──► (1,768)
        └► vision_projs[organ] → (1,256) → L2-norm = image_feat
                                              │
 text prompt (neg,pos) ─BERT(CXR-BERT)→ text_proj → (2,256) = text_feat
                                              ▼
                       logits = image_feat @ text_feat.T / temp = (1,2)
                                              ▼  softmax = [P(정상), P(비정상)]
```

- **장기 4개**: lung / heart / esophagus / aorta. 각 장기마다 학습된 `query_token`(768)과 `vision_projs`(768→256)가 따로 있다.
- **멀티스케일**: ViT가 block 3·6·9·최종 norm의 4개 hidden을 모두 내보내고, ROI 추출 시 선택 토큰을 4 레벨에서 concat해 attention의 key/value로 쓴다.
- **anatomy-aware**라서 `encode_image(vol)->(B,dim)` 같은 균일 계약이 **없다**. 반드시 `(volume, mask)` 쌍이 필요하고 장기별 feature가 나온다.

### 2.2 I/O 텐서 레퍼런스

| 메서드 (어댑터 passthrough) | 입력 | 출력 | 코드 |
|---|---|---|---|
| `visual_encoder(images)` (ViT.forward) | `images:(B,1,112,256,352)` float32 | `(image_embeds:(B,1232,768), hidden_image_embeds: list[4] of (B,1232,768))` | [vit.py:129-146](../third_party/fvlm/lavis/models/blip_models/vit.py#L129-L146) |
| `.prepare_text_feat(test_items)` | `test_items: list[(organ, pathology, neg_prompt, pos_prompt)]` | `dict{(organ,pathology,neg,pos): (2,256)}` float32 L2-norm — **row0=neg/정상, row1=pos/비정상** | [blip_pretrain.py:496-516](../third_party/fvlm/blip_pretrain.py#L496-L516) |
| `.forward_test_win(images, masks, organ_logits, test_organs, text_feat_dict, organ_feat_dict, whole_organ_sizes, skip_organ=None)` | `images:(B,1,112,256,352)`, `masks:(B,1,112,256,352)` 정수 organ id | `organ_logits[item].append(probs)`, `probs:(1,2)` softmax `[P(정상),P(비정상)]`; `organ_feat_dict[organ]=image_feat:(1,256)` | [blip_pretrain.py:396-494](../third_party/fvlm/blip_pretrain.py#L396-L494) |

> ⚠️ **흔한 오해 교정 (코드 직접 대조함)**
> - `prepare_text_feat`의 값은 `(1,256)`이 **아니라 `(2,256)`**다. prompt가 `(neg, pos)` 쌍이라 토크나이저가 2개 시퀀스를 만든다 → `image_feat(1,256) @ text_feat.T(256,2) = (1,2)` 로 정상/비정상 확률.
> - ViT 토큰 수는 1233이 **아니라 `1232`**다. `_VIT_KWARGS`는 `classification=False`(기본값)라 **cls_token이 생성되지 않는다** ([vit.py:131](../third_party/fvlm/lavis/models/blip_models/vit.py#L131)). 7×16×11 = 1232 = (112/16)·(256/16)·(352/32).
> - `test_items` 튜플 = `(organ, pathology, neg_prompt, pos_prompt)`. neg/pos는 `_get_prompt`가 `item[2]`,`item[3]`에서 뽑는다 ([blip_pretrain.py:518-531](../third_party/fvlm/blip_pretrain.py#L518-L531)).

ViT 스펙: hidden 768, 12 layers, 12 heads, patch `(16,16,32)`, img `(112,256,352)`. kwargs는 [fvlm_adapter.py:91-98](../src/baselines/fvlm_adapter.py#L91-L98) `_VIT_KWARGS`(`DUPLICATION INTENTIONAL`).

### 2.3 ⚠️ organ-mask 전제조건

CT-RATE는 volume·report·18-label만 제공하고 **organ mask는 없다.** fVLM은 `(volume, mask)`가 둘 다 있어야 forward가 돈다.

- 마스크는 **TotalSegmentator로 1회 사전계산** → `/workspace/data/preprocessed/ctrate_{train,valid}_masks/` (현재 **미생성**, 후속 task-module plan에서 진행).
- 마스크 정수 id: `0=bg, 1=lung, 2=heart, 3=esophagus, 4=aorta`. spatial은 `(112,256,352)`로 volume과 정합되어 `max_pool3d(16,16,32)` 후 1232 토큰 grid와 맞아야 한다.
- 비용(A6000): valid-only ≈ 3–5h, train ≈ 40–80h. 어댑터는 TotalSegmentator를 **호출하지 않는다** — 마스크는 데이터 파이프라인 입력으로 취급. 레시피는 [vlm_baselines_runbook.md](vlm_baselines_runbook.md) §6.

### 2.4 어댑터가 풀어주는 업스트림 함정 5가지

[fvlm_adapter.py](../src/baselines/fvlm_adapter.py) `_ensure_built()`([:129-246](../src/baselines/fvlm_adapter.py#L129-L246))은 lazy 빌드 한 번에 upstream의 깨진 부분들을 우회한다. 이게 가장 헷갈리는 코드라 마지막에 본다.

1. **lavis 모듈 충돌 스크럽** ([:138-142](../src/baselines/fvlm_adapter.py#L138-L142)) — 같은 프로세스에 다른 `lavis` 패키지가 먼저 import되면 fVLM의 `lavis`를 shadow. `sys.modules`의 `lavis*` 제거 후 fVLM repo를 `sys.path[0]`로.
2. **heavy subpackage 스터빙** ([:144-153](../src/baselines/fvlm_adapter.py#L144-L153)) — `lavis.datasets/processors/tasks`가 decord·medspacy·`transformers==4.25`를 끌어와 우리 4.46과 충돌. 빈 stub로 막음.
3. **`BlipBase.__init__`의 `transformers<4.27` guard 패치** ([:171](../src/baselines/fvlm_adapter.py#L171)) — 2022년 LAVIS 가드. fVLM이 쓰는 경로(`forward_text`)엔 무관해서 no-op로 교체.
4. **`init_tokenizer` 리다이렉트** ([:177-179](../src/baselines/fvlm_adapter.py#L177-L179)) — 로컬 디렉토리 대신 HF id로.
5. **registry 슬롯 비우기** ([:187-189](../src/baselines/fvlm_adapter.py#L187-L189)) — Salesforce 원본 `blip_pretrain_vit`와 fVLM 서브클래스 이름 충돌 해소.

> 왜 `BlipPretrain.from_config()`를 안 쓰나? upstream이 개발자 로컬 MAE 경로 `/storage/guoruizhe/.../mae_pretrain_vit_base.pth`를 하드코딩하기 때문 ([blip_pretrain.py:342-345](../third_party/fvlm/blip_pretrain.py#L342-L345)). 그래서 컴포넌트를 수동 조립하고 공개 ckpt를 strict=False로 덮어쓴다.

### 2.5 읽는 순서 (리딩 맵)

1. **[fvlm_adapter.py 모듈 docstring (1-32)](../src/baselines/fvlm_adapter.py#L1-L32)** + **[공개 API 250-270](../src/baselines/fvlm_adapter.py#L250-L270)** — 무엇을 노출하는지, 왜 nn.Module이 아닌지, organ-mask 전제.
2. **[blip_pretrain.py:63-114](../third_party/fvlm/blip_pretrain.py#L63-L114)** — `BlipPretrain.__init__`: 4 organs, `query_tokens(4,768)`, `vision_projs`(768→256 ×4), `text_proj`, `attention`(MHA 768/4heads).
3. **[blip_pretrain.py:496-531](../third_party/fvlm/blip_pretrain.py#L496-L531)** — `prepare_text_feat` + `_get_prompt`: test_items 4-튜플 구조와 `(2,256)` 출력.
4. **[blip_pretrain.py:396-494](../third_party/fvlm/blip_pretrain.py#L396-L494)** — `forward_test_win`: mask→intact organ 판정→멀티스케일 토큰 선택→query attention→장기별 logits. 추론 핵심.
5. **[vit.py:129-146](../third_party/fvlm/lavis/models/blip_models/vit.py#L129-L146)** — ViT forward: 멀티스케일(block 3/6/9 + norm), 토큰 1232, cls 없음.
6. **[third_party/fvlm/eval.py](../third_party/fvlm/eval.py)** — 실제 end-to-end 호출 예시(어댑터 docstring이 `eval.py:270`,`:318`을 가리킴).
7. (선택) **[fvlm_adapter.py:129-246](../src/baselines/fvlm_adapter.py#L129-L246)** — `_ensure_built` 빌드 함정 5가지(§2.4).

---

## 3. Report2CT의 latent 토대: Frozen MAISI VAE

> ⚠️ MAISI VAE는 **Report2CT(그리고 우리 ctgen 생성기)** 가 사는 latent 공간이다. **fVLM은 MAISI를 쓰지 않고 자체 ViT 인코더**를 쓴다 — 둘을 latent 차원에서 직접 연결하지 말 것.

- 로더: [maisi.py:27](../src/baselines/maisi.py#L27) `load_frozen(device=...)`. 아키텍처 kwargs를 재선언하지 않고 [inference.json](../third_party/maisi_bundle/configs/inference.json)의 `autoencoder_def`를 `ConfigParser`로 인스턴스화.
- **전 파라미터 `requires_grad=False`** — [tests/test_maisi_frozen_load.py](../tests/test_maisi_frozen_load.py)가 강제. 큰 볼륨 디코드는 `SlidingWindowInferer`(roi 80³, overlap 0.4) 사용([maisi.py:23-24](../src/baselines/maisi.py#L23-L24)).
- 변환: CT `(B,1,480,480,256)` ↔ latent `(B,4,120,120,64)`. HU는 `[-1000,1000]`→`[0,1]`.
- round-trip 상한선(우리 생성 품질의 천장): **PSNR 30.94 ± 2.97 dB, SSIM 0.7195**([results/upper_bound.json](../results/upper_bound.json)).

---

## 4. 빠른 검증 커맨드

```bash
# fVLM 어댑터 스모크 (가중치 불필요 — 빌드/conflict-resolution 경로만)
pytest tests/test_fvlm_adapter.py -k "not requires_weights" -v

# fVLM Hydra config가 올바른 _target_으로 compose되는지
python -c "from hydra import initialize, compose; \
           initialize(version_base='1.3', config_path='configs'); \
           print(compose('train', overrides=['model=vlm_backbone/fvlm']).model._target_)"

# Report2CT 모듈/파리티 (UNet forward shape + 업스트림 JSON 일치)
pytest tests/test_report2ct_module.py tests/test_report2ct_parity.py -v

# MAISI VAE frozen 로드 (전 param requires_grad=False)
pytest tests/test_maisi_frozen_load.py -v

# Report2CT 실험 config compose 정상성 (학습 없이)
python src/train.py experiment=report2ct_repro --cfg job --resolve

# fVLM 가중치까지 (docs/vlm_baselines_runbook.md 다운로드 후)
CUDA_VISIBLE_DEVICES=0 pytest tests/test_fvlm_adapter.py -k requires_weights -v
```

---

## 5. 관련 문서 인덱스

| 문서 | 내용 |
|---|---|
| [report2ct_training_runbook.md](report2ct_training_runbook.md) | Report2CT 5단계 실행 커맨드(text emb → image emb → datalist → train), 스모크 vs 풀런, 트러블슈팅 |
| [report2ct_external_components.md](report2ct_external_components.md) | 텍스트 인코더 HF 모델 id, MAISI 버전, 스케줄러/CFG 등 외부 의존 핀 |
| [report2ct_training_handoff.md](report2ct_training_handoff.md) | 하드웨어 비용(2×A6000 ≈ 150–250 GPU-h), sanity 스키마, decision gate |
| [vlm_baselines_runbook.md](vlm_baselines_runbook.md) | fVLM 가중치 다운로드(gdrive) + **organ-mask 사전계산 레시피**(§6) |
| [GUIDE.md](../GUIDE.md) / [CLAUDE.md](../CLAUDE.md) | 프로젝트 전체 투어 / 규칙·아키텍처·baseline-clone 감사 체크리스트 |

---

### 부록: 정확성 노트
이 문서의 모든 shape/라인은 작성 시 다음 파일을 직접 열어 대조했다 — `report2ct_module.py`, `report2ct_text_encoder.py`, `maisi.py`, `fvlm_adapter.py`, `blip_pretrain.py`, `vit.py`, 그리고 4개 config. 특히 fVLM의 `prepare_text_feat` 출력 `(2,256)`과 ViT 토큰 수 `1232`(cls 없음)는 코드 대조로 교정한 값이다. 업스트림이 핀 SHA에서 바뀌면 라인 번호가 어긋날 수 있으니, 의심되면 해당 함수명으로 재확인할 것.
