# 연구 방향 로드맵 — 텍스트 조건화 → U-DiT / U-REPA

> **상태 (2026-08-02 갱신)**: Phase 0(`report2ct_wan_repa` 300 epoch) **완료**
> (`epoch_299.ckpt` 저장, 크래시 없이 정상 종료). Phase 1은 1a(`c_ctx` 토큰 시퀀스 사이드카,
> toy 6,304개 전체 precompute)·1b(`Report2CTSeqDataModule`)·1c(pooled 조건화 UNet 서브클래스 +
> 패딩 마스킹 cross-attn)·eval-side train/eval parity(`Report2CTWanSeqLatentSampler`)까지
> 코드·커밋 완료. **남은 것은 1e(toy 5k A/B/C 게이트) 실행뿐** — 사용자 시간 될 때 착수.
> Phase 2a(더미 벤치마크)도 실행 완료 — 결과는 해당 절 참조.
> ⚠ **개명(2026-08-02)**: pooled 텍스트 조건화 경로를 "AdaLN"이라 불렀으나 실제로는
> scale/shift/gate 없는 단순 additive 조건화라 `text_pooled_cond`로 코드 전체 개명함
> (이전 이름 `text_pooled_adaln`). 이 문서도 전체 갱신.
> 이 문서는 대화 중 코드·데이터로 직접 확인한 사실 위에 세운 것이며,
> 확인되지 않은 항목은 본문에 ⚠로 표시했다.

## Context

VLM3D 2026 ctgen(Task 4) 제출 마감 **2026-08-20** (작성일 07-31 기준 20일). headline은
`report2ct_wan_mask_v2`로 이미 확보돼 있어 안전망이 있는 상태다.

이 계획은 두 가지를 동시에 노린다:
1. **챌린지 성적 개선** — 1차 지표인 CLIPScore-T2I를 텍스트 조건화 개선으로 올린다
2. **논문 기여 확보** — 3D 의료 볼륨 생성에 U자형 DiT(U-REPA 계열) + MM-DiT를 적용

### 조사에서 확인된 사실 (이 계획의 근거)

**(1) 현재 텍스트 조건은 pooled 2토큰이고, 정보가 실제로 파괴된다.**
[src/baselines/report2ct_text_encoder.py](../src/baselines/report2ct_text_encoder.py)의
`Report2CTTextEncoder`는 3개 생의학 인코더(MedEmbed-large 1024 + ClinicalBERT 768 +
BiomedVLP-CXR-BERT-specialized 768)의 `last_hidden_state`를 mean-pooling해 `(B, 2, 2560)`을 만든다.
세 인코더를 직접 돌려 측정한 pooled 코사인 유사도:

| 문장 쌍 | pooled cos |
|---|---|
| "No pleural effusion" vs "Pleural effusion" | 0.899 |
| "No pneumothorax detected" vs "Pneumothorax detected" | 0.935 |
| "No nodule observed" vs "A nodule observed" | 0.900 |
| "right **lower**" vs "left **upper**" lobe nodule | 0.868 |
| *(기준) 무관한 문장* | *0.663* |

"무관=0.66, 동일=1.0" 척도에서 **의미가 정반대인 문장이 동일 쪽 71% 지점**. 부정문은
리포트의 **99.4%** 에 등장한다. 실제 파이프라인은 findings 전체(중앙값 184단어, ~15문장)를
벡터 하나로 뭉개므로 실측보다 더 나쁘다. **있음/없음과 위치가 곧 CLIPScore-T2I가 재는 축이다.**

**(2) 토큰 시퀀스는 이미 계산되고 버려지고 있다.**
upstream 노트북이 `c_ctx = torch.cat(token_list, dim=1)  # [B, num_models*512, max_dim]`을
만들지만 저장하지 않는다. 우리 포팅 코드가 이 폐기를 명시적으로 기록해 두었다
([report2ct_text_encoder.py:93-94](../src/baselines/report2ct_text_encoder.py#L93-L94)):
*"returning only the pooled (c_vec) component (the c_ctx token-level is discarded)"*.
→ **복구 비용 = precompute 재실행. 새 모델도 학습도 불필요.**

**(3) 모델 코드 변경이 거의 없다.**
`Report2CTModule._shared_forward`는 이미 conditioner-agnostic이다 —
datamodule이 `context` 키로 `(B, N, D)`를 내보내면 그대로 cross-attention에 들어간다
(fVLM이 `(B, 4, 256)`으로 이미 사용 중). **cross-attention 경로는 모델 수정 0.**

**(4) pooled 텍스트 조건화 주입 지점이 깨끗하다.**
`DiffusionModelUNetMaisi.forward`는 `emb`를 두 헬퍼로 조립한다:
```python
emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
```
→ **`_get_input_embeddings`만 오버라이드하는 서브클래스로 충분** (MONAI 패치 불필요,
`third_party` 무손상). 그리고 `emb`는 **모든 레벨**의 ResBlock에 들어가므로,
`attention_levels: [false, false, true, true]` 때문에 cross-attn이 닿지 못하는
**고해상도 레벨(64³, 32³)에 텍스트가 도달하는 유일한 통로**가 된다.
⚠ **AdaLN이 아니다** — 실제 메커니즘은 `DiffusionUNetResnetBlock.forward`에서
`h = h + time_emb_proj(emb)[...,None,None,None]`(채널별 bias를 더하기만 함, scale/gate 없음).
DiT의 AdaLN-Zero(`x*(1+scale)+shift`, 게이트 포함)보다 약한 additive 조건화라서
`text_pooled_cond`로 부른다(2026-08-02 개명, 이전 `text_pooled_adaln`). `_get_input_embeddings`가
spacing/region 벡터를 붙이는 방식은 **진짜 concat**(`torch.cat((emb, _emb), dim=1)`)이 맞음 —
새 텍스트 pooled 벡터도 이 패턴을 따른다.

**(5) ⚠ "선행연구는 pooled를 쓴다"는 주장은 성립하지 않는다.**
vendoring된 4개 모델을 코드로 확인한 결과 2:2로 갈린다.

| pooled | 시퀀스 |
|---|---|
| **Report2CT** (2토큰), **text2ct** (`pooler_output` → `(B,1,D)`) | **GenerateCT** (`t5_encode_text` → `last_hidden_state` + `text_mask`), **Foundation-VAE** (`MedBERTEmbedder`, `return_embeddings=True` → `(B,77,768)`) |
| 둘 다 **MAISI-latent 계열** | 둘 다 **SD/Imagen 계열** |

분야 전체의 맹점이 아니라 **계보의 문제**다 (MAISI의 UNet은 원래 class label + region index
조건용이라 텍스트가 작은 cross-attn 슬롯에 얹힌 구조). 따라서 시퀀스 조건화는
**논문 기여가 아니라 성능 개선 + 후속 작업의 전제**로 취급한다.
논문 주장은 **MM-DiT(joint attention)와 3D U-DiT 설계**가 진다 — 이 둘은 text→CT에서 미청구 상태
(GenerateCT·Foundation-VAE 모두 plain cross-attention).

**(6) U-REPA가 U-DiT 하이퍼파라미터의 published 레퍼런스를 준다.**
`third_party/repa_refs/u_repa/` (NeurIPS'25, arXiv 2503.18414, *"Aligning Diffusion U-Nets to ViTs"*):
`models_urepa.py`가 `patch_size=2`, `hidden_size=1152`, `depth=[10,16,10]` 대칭,
1단계 `Downsample`/`Upsample`, U-ViT식 long skip(`skip_linear = Linear(2*hidden, hidden)`)이다.
`loss.py`에 **manifold loss**(자기유사도 행렬 D vs D̃ 비교, `manifold_tp='cos-l2'`, weight 3)가
추가돼 있고, 이것이 "U-Net용으로 REPA를 재단했다"는 핵심이다.

⚠ **갭**: 2D 전용이라 stage-1에 국소 attention이 없다. 256px/patch2면 stage-1이 256토큰이지만
3D로 옮기면 32³ = **32,768토큰**이 되어 global attention이 forward 6.6 TFLOP를 먹는다.
**3D 이식에는 window attention 또는 추가 다운샘플이 필수.**

---

## Phase 0 — 진행 중 (조치 불필요)

`report2ct_wan_repa` 학습이 작성 시점 **Epoch 278/300** 진행 중 (3 GPU).
완료 후 eval하여 REPA 효과를 확정한다. Phase 1 준비 작업은 이 학습과 병행 가능
(코드 작업 + 텍스트 precompute는 `CUDA_VISIBLE_DEVICES=3` — GPU 2는 사용자 작업이므로 접근 금지).

---

## Phase 1 — 텍스트 조건화 (챌린지 + 전제 작업)

**목표**: pooled 2토큰 → 토큰 시퀀스(cross-attn) + pooled(전역 additive 조건화, `text_pooled_cond`) 이중 경로.
SD3의 설계에서 joint attention만 뺀 형태.

### 1a. `c_ctx` 복구 — 사이드카 방식

- `src/baselines/report2ct_text_encoder.py`: `encode_tokens()` 추가.
  기존 `encode()` / `encode_pair()` 는 **손대지 않는다** (parity 테스트 보호).
- ⚠ 3개 인코더의 hidden dim이 다르다 (1024/768/768). upstream `c_ctx`는 `max_dim=1024`으로
  **제로 패딩** 후 시퀀스 축 concat — 절반이 0인 벡터가 생긴다.
  **제로 패딩 대신 인코더별 학습 선형 투영으로 공통 차원에 맞춘다.**
- `scripts/precompute_report2ct_text_embeddings.py`: `--save-tokens` 플래그 추가 →
  `<id>_emb.nii.gzmulti_2560_ctx.npz` 사이드카 별도 생성.
  **기존 `*multi_2560.json`은 불변** → 현재 돌고 있는 학습들 전부 무영향.
- 대상: toy 5k train + valid_v2 1304 먼저. full 46k는 게이트 통과 후.

### 1b. 시퀀스를 내보내는 datamodule 변형

- `src/data/report2ct_datamodule.py`의 `_build_transforms` / `_transforms` 훅을 사용
  (이미 *"overridable hook for subclass variants"* 로 설계돼 있음).
- 사이드카 `.npz`를 읽어 `context` `(B, N, 2560)` + `context_pooled` `(B, 2, 2560)` 을 emit.
- **길이 처리**: 인코더별 고정 길이 패딩. 실측(500개 npz 샘플): findings p50 285 / p90 439–468 /
  512 상한 걸림, impression p50 33–47 / p90 88–125. **채택**: findings 384, impression 128
  (인코더 3개 합산 1536토큰 — upstream의 원래 총 예산(512×3)과 같은 총량을 findings/impression에
  똑똑하게 재배분한 것; findings ~90%, impression ~95%+ 커버).
- **⚠ 패딩 마스킹 — 지금 구현한다** (기존 계획은 "나중에"였으나 조사 후 뒤집음). 코드로 확인:
  `monai.networks.blocks.crossattention.CrossAttentionBlock.forward(x, context)`에 **mask 인자가
  아예 없다**. 실제 T2I/T2V 모델 조사 결과 마스킹은 "CLIP vs T5"가 아니라 **아키텍처 계열**로 갈린다 —
  SD1.x/SDXL/SD3/Flux(U-Net 계열 + Flux DiT)는 마스킹 안 함(SD 계열은 `encoder_attention_mask` 인자
  자체는 있으나 파이프라인이 안 씀), **PixArt-α(DiT + 명시적 mask 배선)는 마스킹함**
  (`encoder_attention_mask=prompt_attention_mask`). 우리 패딩 비율(1536토큰 중 짧은 impression은
  상당수가 패딩)이 SD3/Flux(77–512토큰, 패딩 비율 낮음)보다 PixArt-α의 "가변 길이 T5, 긴 꼬리" 상황에
  더 가깝다 — **가장 가까운 선례가 마스킹하는 쪽**이라 지금 넣는다. `CrossAttentionBlock`은 pip
  설치된 MONAI라(third_party 아님) 서브클래싱 자유로움.

### 1c. Pooled 전역 조건화 주입 — UNet 서브클래스 ✅ 완료

- `src/models/components/maisi_unet_text_pooled.py`: `DiffusionModelUNetMaisiTextPooled`
  (`DiffusionModelUNetMaisi` 상속), pooled 벡터의 선형 투영(zero-init)을 `emb`에 더함
  (concat 아님 — 모든 ResBlock의 `time_emb_proj` 폭이 고정이라 concat은 shape 크래시).
  같은 클래스가 패딩 마스크 cross-attention(`masked_cross_attention.py`)도 함께 활성화.
- `Report2CTModule`에 `text_pooled_cond` 플래그로 pooled 벡터를 UNet에 전달하는 경로 추가
  (이전 이름 `text_pooled_adaln` — 진짜 AdaLN이 아니라서 2026-08-02 개명).
- ⚠ **CFG**: 두 경로를 **함께 드롭** — 구현 완료, `cfg_drop_prob=0.15` / `cfg_per_sample` 의미 보존 확인.

### 1d. ⭐ 20일 안에 넣는 핵심 트릭 — `cross_attention_dim: 2560` 유지

시퀀스를 **2560으로 투영**해서 넣으면 `ep299` 체크포인트의 cross-attention K/V 가중치가
그대로 전이된다 → **from-scratch가 아니라 fine-tune**으로 갈 수 있다.
(1a의 인코더별 투영 레이어가 이 역할을 겸해 제로 패딩 문제도 동시에 해소.)

⚠ **설계는 완료, 실행 경로엔 gap 있음 (2026-08-02 확인, 미수정)**: 이 트릭을 실제로 쓰는
`Report2CTModule.init_from_ckpt` → `_load_weights_only`(report2ct_module.py:213)가
`strict=True`. arm B/C의 UNet(`DiffusionModelUNetMaisiTextPooled`)은 arm A 체크포인트엔 없는
`text_pooled_proj.weight/bias`를 갖고 있어서, arm A 체크포인트를 그대로 `init_from_ckpt`로
로드하면 지금 코드로는 missing-key `RuntimeError`. **1e 게이트 자체는 안 막힘**(arm A도
스크래치 학습이라 B/C도 스크래치로 하면 여전히 apples-to-apples) — 다만 1f(46k 파인튜닝,
23시간 추정)는 이 트릭이 실제로 작동해야 그 추정이 성립. 수정은 작음: `strict=False` +
missing key가 정확히 `text_pooled_proj.*`인지 확인하는 assert(~10줄, 다른 아키텍처 drift는
여전히 loud하게 에러나도록). **1e를 파인튜닝으로 돌리기로 하면 그 직전에 하면 됨.**

### 1e. toy 5k 통제 A/B — **게이트 G1** ⬜ 다음 할 일 (미착수)

기존 `report2ct_wan`이 toy 5k(625 it × batch 8, 체크포인트
`outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt`)로 학습돼 있어
**동일 조건 통제 비교**가 된다. 데이터·스케줄·latent 전부 고정, 조건화만 변경 —
data/model config는 다 준비됨, experiment yaml만 아직 없음(직접 CLI override로도 실행 가능):

| arm | 텍스트 표현 | 주입 | 상태 |
|---|---|---|---|
| **A** (기존, 재학습 불필요) | pooled 2토큰 | cross-attn | ✅ 체크포인트 존재 |
| **B** | 시퀀스 | cross-attn | `configs/model/report2ct_wan_seq.yaml` 준비, 학습 안 함 |
| **C** | 시퀀스 + pooled | cross-attn + `text_pooled_cond` | `configs/model/report2ct_wan_seq_pooled.yaml` 준비, 학습 안 함 |

**게이트 기준**: B 또는 C가 A 대비 **CLIPScore-T2I** 개선.
eval은 `task.n_samples=300`, step 100, `fid_profile=docker`, **신규 `out_dir`**
(명명 규칙 `eval_ep<NNN>_sp0.75_1.3_cfgt5...` 준수 — 기존 dir 재사용 금지).

### 1f. 게이트 통과 시 — full 46k

- 46k `c_ctx` precompute (수 시간)
- `ep299`에서 fine-tune 30–50 epoch. A6000 Pro 3장 기준 epoch당 약 27.5분 → **50 epoch ≈ 23시간**
- valid_v2 n300 eval → 이기면 headline 교체, 지면 `report2ct_wan_mask_v2` 유지

### 1g. 여유가 남으면 — 리포트 노이즈 ablation

`validation_reports.csv` 3039건 실측: 선행검사 비교 **13.8%**(모델이 본 적 없는 이전 검사 참조
— 학습 불가능한 노이즈), 권고 25.5%, 불확실 표현 28.5%, 촬영기법 45.4%.
**"prior/recommendation 문장을 제거하면 더 오르는가"** 를 재면
"pooling은 노이즈 대응이었다"는 가설을 직접 검증하는 실험이 된다.
결과가 어느 쪽이든 논문에 들어간다.

### 1h. 후속 ablation (1e/1f 이후, 지금 착수 안 함) — 인코더 교체 + 역할 재배치 ⬜

**계기**: 현재 3개 텍스트 인코더 중 `BiomedVLP-CXR-BERT-specialized`가 실제로 CLIP류
대조학습을 거친 VLA(vision-language-alignment) 모델임을 HF 모델카드 원문으로 확인했다
("CXR-BERT is trained in a multi-modal contrastive learning framework, similar to the CLIP
framework" / "[CLS] token is utilized to align text/image embeddings") — **단, 정렬 대상
이미지가 CT가 아니라 흉부 X-ray(MIMIC-CXR)** 다. `third_party/spectre/`(CVPR 2026,
arXiv:2511.17209)에 **CT 리포트에 정렬된 동일 성격의 텍스트 타워**가 이미 있다
(`Qwen3-Embedding-0.6B` + LoRA, SigLIP loss, `text_embed_dim=1536`).

**변수 분리 원칙(사용자 지적)**: 1e의 A/B/C는 "pooled→시퀀스"라는 **단일 변수**만 바꾼다.
인코더 교체를 같이 넣으면 CLIPScore 변화가 어느 쪽 때문인지 분리가 안 된다 →
**인코더 교체는 1e 결과가 나온 뒤, 별도 라운드로 분리한다.**

**후속 ablation 2개** (같은 precompute로 라벨만 바꿔 재사용 가능, 추가 비용 거의 없음):
1. **D**: `CXR-BERT` 시퀀스 → `SPECTRE-VLA` 시퀀스로 교체(역할은 그대로 cross-attn).
   "CXR 정렬 vs CT 정렬"만 분리 측정.
2. **E**: 비전 정렬 인코더(D의 승자)를 cross-attn이 아니라 **pooled 전역 조건화
   (`text_pooled_cond`)** 로 이동. 근거: SPECTRE-VLA·CXR-BERT·(HiDream-I1의 CLIP) **셋 다
   대조학습의 정렬 대상이 pooled/[CLS] 토큰 하나**였다 — 토큰별 표현은 직접 학습 신호를 받은
   적이 없다. HiDream-I1(arXiv:2505.22705)이 실제로 이 역할 분리(pooled CLIP → AdaLN 전역 조건 /
   T5+Llama 시퀀스 → cross-attn)를 쓴다(HiDream 자체는 진짜 AdaLN-Zero, 우리는 additive라는
   차이는 있음). **다만 이건 문헌이 아니라 추론이다** — HiDream이 "왜" 그렇게 했는지 논문이
   직접 설명하진 않고, 이 프로젝트 세팅에서 검증된 적도 없다. E는 그 추론을 직접 테스트하는
   실험으로 취급한다.

**기각한 대안**: RadFM(arXiv:2308.02463/Nat.Commun. 2025, MedMD, 3D ViT+Perceiver+MedLLaMA-13B) —
텍스트 임베딩 API 없이 VQA/리포트생성 전용 생성모델이라 통합 비용 대비 가치가 낮음, 기각.
RadFinder(huggingface.co/collections/lmb-freiburg/radfinder)는 사용자 확인상 SPECTRE의
fine-tune이라 D/E 결과가 나온 뒤 SPECTRE 자리에 바로 교체해볼 수 있는 저비용 후속 — 별도 조사 불필요.

**타 문헌**: SD3 논문(arXiv:2403.03206) §5.3.3 — 인코더 46.3% 드롭아웃 학습 결과 "CLIP 2개만 써도
성능 저하 제한적, T5는 세부 장면·타이포그래피에서만 확실히 기여" → 동일 역할 인코더 간 중복이
실제로 있다는 증거(단, 우리 3개 인코더의 중복도는 별도 실측 필요). "같은 텍스트를 여러 인코더로
인코딩할 때의 중복 제거" 자체를 다룬 논문은 조사에서 찾지 못했다(=이 설계 공간이 비어 있다는 뜻이기도
하다). ⚠ 이 절의 논문 인용은 워크플로우 조사 결과이며 원문을 직접 재확인하지 않았다 — 실제 사용 전
1차 출처 확인할 것.

---

## Phase 2 — U-DiT / U-REPA (챌린지 이후 주력)

### 2a. 먼저 실측 — 더미 벤치마크 ✅ 완료 (2026-07-31)

`tests/udit_bench/`(model.py + bench.py + README.md)에서 3-level 계층(L0 2enc+2dec
윈도우8³ / L1 4enc+4dec / L2 12-block 보틀넥, patch2, **텍스트 조건 없음** — cross-attn
비용은 2c에서 이미 분석적으로 계산돼 있어 실측 대상이 아니었음)을 실제로 구현해
RTX PRO 6000 Blackwell(A6000 Blackwell) 1장에서 batch 4/8/16 × checkpoint on/off로 측정:

| 항목 | 사전 추정 | **실측** | 결과 |
|---|---|---|---|
| 샘플당 활성화 (checkpoint 없음) | ~3.2 GB | **3.9 GB** (batch8) | 22% 높음 — "30–50% 증가 가능" 캐비어트 안 |
| U-Net 대비 절감 | −45% (5.8GB 대비) | **−33%** | 방향 맞음, 폭은 과장돼 있었음 |
| 처리량/MFU | 근거 없는 가정 (25–30%) | **U-Net의 2배 이상** (21.49 vs 10.4 samp/s/GPU, batch8) | **추정이 너무 비관적이었다** |
| checkpoint 필요성 | "batch8까지 필요" 가정 | **불필요** (batch16도 69.5/97GB) | 이 스케일에선 항상 off로 단순화 |

⚠ **GPU 경합 상태에서 측정** — 사용자의 `generate_wan_latents.py`가 4-GPU를 쓰는 도중
GPU 3 한 장을 나눠 썼다(사용자 명시 승인). 메모리 수치는 경합 무관, **처리량은 이 경합 때문에
과소평가돼 있다** — 즉 아래 학습 시간은 이미 DiT에 불리한 조건에서 나온 하한값이다.

**학습 시간 재계산**(batch8, checkpoint off, A6000 Blackwell ×3 선형 스케일링 가정,
full 46,393 볼륨/300epoch): Wan U-Net(toy 실측 10.4 samp/s/GPU 외삽) **~5.2–5.7일** vs
**Wan U-DiT(이번 실측) ~2.5일**. 이전 "5–8일 밴드, 대략 U-Net과 동률"이라는 결론은
**폐기** — 실측 기준으론 U-DiT가 명확히 더 빠르다. 세부 표·재현 커맨드·캐비어트 전체는
[tests/udit_bench/README.md](../tests/udit_bench/README.md) 참조.

### 2b. 두 구조를 비교

| | 기반 | 3D stage-1 처리 |
|---|---|---|
| **U-REPA-3D** (baseline) | `models_urepa.py` 구조 이식: patch 2, `depth=[10,16,10]`, 1단계 다운샘플, long skip, manifold loss(`cos-l2`, w=3) | window attention 추가 (아래) |
| **3레벨 계층** (ablation) | L0 32³/d384/4블록, L1 16³/d768/8블록, L2 8³/d1152/12블록 | 동일 |

**stage-1 attention 구현**: NATTEN 3D는 피한다(미설치 + 3D 커널 최적화 부족).
**Swin식 non-overlapping window(8³) + shifted window**를 `reshape` + torch 2.7 내장 SDPA로 구현 —
커스텀 커널 0. 참고 구현이 `monai.networks.nets.swin_unetr`에 있다
(`WindowAttention`이 `len(window_size)==3` 분기를 가짐, MONAI 1.4에서 확인 완료).

**3D 고유 관찰(측정 대상)**: 3D는 레벨당 **8배**로 붕괴한다(2D는 4배). 그 결과 깊은 레벨은
파라미터를 늘려도 연산·메모리가 거의 안 는다 (3레벨 설계에서 L2는 파라미터 75% / 연산 15%).
→ **"3D 생성 모델은 2D보다 용량을 아래로 밀어야 한다"** 는 설계 원칙을 측정으로 뒷받침한다.
2D 논문에서는 나올 수 없는 관찰이라 독립적 기여가 될 수 있다.

### 2c. MM-DiT 얹기 — 논문 주장의 중심

Phase 1에서 만든 토큰 시퀀스가 전제. **L1/L2에만** 적용(L0는 이미지 전용 — 가장 비싼 레벨에서
텍스트 스트림 가중치를 복제하지 않는다).

블록당 파라미터 회계: cross-attn DiT 블록 = `20d² + 2·d_ctx·d` /
MM-DiT 블록(양 스트림 동폭) = `36d²`.

| 구성 | 파라미터 | forward FLOPs/샘플 |
|---|---|---|
| cross-attn DiT (기준) | 465M | 2.39 T |
| MM-DiT (텍스트=이미지 폭) | 755M (1.62×) | 2.52 T (**1.05×**) |
| MM-DiT (텍스트 1/2 폭) | 496M (1.07×) | < 2.52 T |

**파라미터 +62%인데 연산 +5%** — 텍스트 스트림이 ~300토큰만 처리하기 때문.
증가분은 L2에 몰린다(이미지 512토큰 vs 텍스트 300토큰으로 텍스트가 시퀀스의 37%).

⚠ **효율성 주장의 정확한 표현**: 파라미터는 U-Net 233M 대비 **2–3배 늘어난다**.
"메모리를 덜 쓴다"고만 쓰면 "모델을 줄였으니 당연"으로 오독된다. 정확한 형태:
> **3.2배 많은 파라미터를 45% 적은 활성화 메모리로 학습한다** —
> 3D에서는 용량을 저해상도 레벨로 재배치하면 이것이 가능하다

---

## 검증

### 회귀 방지 (필수)
- `pytest tests/ -q` — 특히 `tests/test_report2ct_parity.py::test_text_encoder_parity`.
  기존 pooled 경로는 **bit-identical**이어야 한다 (신규 코드는 전부 opt-in 경로).
- 기존 `*multi_2560.json` 미변경 확인 → 진행 중 학습 무영향.

### 신규 경로
- 1샘플 smoke: `python src/train.py experiment=<new> trainer.max_epochs=1 trainer.limit_train_batches=50`
- shape 검증: `context` `(B, N, 2560)`, `context_pooled` `(B, 2, 2560)`, UNet 출력 `(B,16,64,64,64)`
- **grad sanity ⚠**: `DiffusionModelUNetMaisi`는 `zero_module` 출력 conv 때문에
  step 0에서 모든 입력의 grad가 정확히 0이다. grad 확인 전 출력 conv를 non-zero로 만들 것
  (첫 옵티마이저 스텝 후 자동 해소).
- eval 1샘플로 출력 크기 실측 후 n300 본 eval.
  `model.spacing_mm` / `model.cfg_scale` 은 CLI에서 **명시 필수** (Hydra MISSING).

### 게이트
| 게이트 | 기준 | 실패 시 |
|---|---|---|
| **G1** (1e) | B 또는 C가 A 대비 CLIPScore-T2I 개선 | 1g 노이즈 필터링 시도 → 그래도 안 되면 negative result로 기록하고 Phase 2로 |
| **G2** (1f) | full 46k에서 `wan_mask_v2` 초과 | headline 유지, 챌린지 리스크 0 |
| **G3** (2b) | U-DiT가 동일 계산량에서 U-Net 이상 | "3D 의료 데이터 규모에서는 conv 귀납편향이 유효" — 프론티어 흐름에 반하는 결과라 그 자체로 보고 가치 |

---

## 리스크

| 리스크 | 근거 | 대응 |
|---|---|---|
| **리포트 노이즈로 시퀀스가 역효과** | 선행검사 비교 13.8% 등 학습 불가능한 노이즈 실재 | 1g 문장 필터링. 단 부정문 99.4%는 오히려 시퀀스 유리 근거 |
| **MONAI cross-attn이 mask 미지원** | 미확인 | zero-padding (SD 선례). 구현 전 확인 |
| **DiT가 47k 데이터 규모에서 열세** | conv 귀납편향, DiT는 수렴 느림 | REPA/U-REPA가 정확히 이 문제의 완화책. G3가 negative여도 보고 가치 |
| **U-REPA 3D 이식의 미검증 선택** | 윈도우 크기, 레벨 수/폭 배분은 2D 논문에 근거 없음 | 5k toy 짧은 스윕으로 결정 |
| ~~비용 추정이 가정 위에 있음~~ | ~~MFU 25–30%, 활성화 3.2GB 모두 미실측~~ | ✅ 2a에서 실측 완료(07-31) — 활성화는 방향 확인, 처리량은 추정보다 좋았음 |

---

## 읽을 논문

**1순위 (MM-DiT 근거)**
- **SD3** (*Scaling Rectified Flow Transformers…*, 2024) — MMDiT 정의 + cross-attn/UViT 대비 ablation,
  pooled→AdaLN & 시퀀스→joint attention 이중 경로 설계
- **DiT** (Peebles & Xie 2023) — patch 2/4/8 ablation, adaLN-Zero

**2순위 (U자형 + REPA)**
- **U-REPA** (arXiv 2503.18414 — 이미 vendoring, manifold loss가 핵심)
- **HDiT** (ICML 2024) — hourglass + 고해상도 neighborhood attention
- **U-DiT** (NeurIPS 2024) — 토큰 다운샘플 attention
- **REPA** 원논문

**3순위 (video diffusion — 3D 설계 근거)**
- **Latte** — spatio-temporal 블록 4변형 비교 (3D attention 분해에 가장 직접적)
- **CogVideoX** — 분리형 → full 3D attention 전환 근거
- **HunyuanVideo** — video에서의 MMDiT(dual-stream) 사용례
- **Wan2.1** — 우리 substrate. VAE + DiT 설계, 3D RoPE
- **LTX-Video** — 계층 대신 VAE 압축으로 푸는 대안 노선

**4순위 (3D 계층 attention)**: Video Swin Transformer · MViTv2(pooling attention) · Swin UNETR · DiT-3D

**5순위 (포지셔닝)**: **Foundation-VAE**(가장 가까운 동시기 연구 — 이미 vendoring, 필독) ·
GenerateCT · Report2CT · MedSyn · MAISI

⚠ 위 목록은 대화 중 기억에서 정리한 것. **U-REPA 외에는 제목·저자·연도를 확인하고 인용할 것.**

### 참고: video 아키텍처를 그대로 옮기면 안 되는 이유
video의 3축은 비대칭(T ≠ H,W)이고 그 비대칭이 Video Swin의 H,W-only 병합,
Wan의 patch (1,2,2), spatial↔temporal 분리 attention을 낳았다.
**CT는 3축이 전부 공간이고 유효 spacing 0.74/0.74/1.3mm로 거의 등방**이라
(patch 2면 토큰 하나가 약 12×12×10mm) 축 분리 기법은 의미가 없는 대신,
**등방 3D 계층은 video에서 늘 어색했던 구조인데 CT에서는 자연스럽다.**

---

## 실행 순서 요약

```
Phase 0  REPA ep300 완료 → eval                                    ✅완료
Phase 1  1a c_ctx 사이드카 ✅─┐
         1b datamodule       ✅├→ 1e toy 5k A/B [G1] ⬜다음 할 일 → 1f full 46k [G2] → 챌린지 제출
         1c pooled-cond 서브클래스✅┘                              └→ 1g 노이즈 ablation (여유 시)
                                        1e/1f 이후 별도로 → 1h 인코더 교체 ablation
──────────────────────────── 08-20 마감 ────────────────────────────
Phase 2  2a 더미 벤치마크 ✅완료 → 2b U-REPA-3D vs 3레벨 [G3] → 2c MM-DiT
```

**Phase 1은 Phase 2의 전제**(MM-DiT는 토큰 시퀀스가 있어야 의미가 있음)이므로 G1이
실패해도 작업이 버려지지 않는다. 챌린지 headline은 `report2ct_wan_mask_v2`로 확보돼 있어
모든 게이트 실패 시에도 제출 성적 손실은 없다.

## 규약 (프로젝트 [CLAUDE.md](../CLAUDE.md))

- `main` 브랜치에서만 작업, worktree 금지
- `third_party/` 읽기 전용 — MONAI는 서브클래스로 확장
- 산출물 `/workspace/data/`, 실행 로그 `/workspace/logs/`, eval `out_dir`은 `/workspace` (md0)
- ad-hoc GPU 작업은 `CUDA_VISIBLE_DEVICES=3` (GPU 2는 사용자 작업, 접근 금지)
- 유닛 하나씩 구현 → 설계 의도 + 핵심 로직 설명 → 승인 후 다음 유닛
