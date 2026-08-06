---
description: Encoder/decoder I/O + preprocessing cheat-sheet (MAISI VAE, CT-CLIP, fVLM, GenerateCT, Report2CT UNet). Auto-injected when editing model/eval/saliency/precompute code.
paths:
  - "src/baselines/**"
  - "src/eval/**"
  - "src/models/**"
  - "src/diagnostics/**"
  - "tests/saliency_map/**"
  - "scripts/precompute_*"
---

# Encoder / Decoder I/O cheat-sheet

One-stop contract reference for every CT encoder/decoder we wrap, so consumer code
(saliency, retrieval, samplers, precompute) doesn't re-derive shapes / ranges / call
signatures from the adapter every time. **The adapter source is ground truth — this is
a verified summary; check the cited `file:line` before trusting a detail.**

**How this file reaches you:** the OMC `rules-injector` hook auto-injects it (once per
session, dedup by content hash) when you Read/Edit/Write a file matching `paths:` above.
Deeper *per-adapter* rules sit alongside and fire on their own globs:
[fvlm.md](fvlm.md) (`src/baselines/fvlm_*.py`) and
[baseline-clone.md](baseline-clone.md) (the 8-point silent-bug audit; `src/baselines/**` + `src/eval/**`).

**Cross-encoder trap (baseline-clone #1 — activation-range handoff):** input intensity
convention is **not** the same across encoders — **MAISI VAE wants `[0,1]`, CT-CLIP wants
`[-1,1]` (HU/1000)**. Feeding one encoder the other's range produces plausible-looking but
biased features/HU with no crash. Always check the per-encoder "Input" row below.

> Status: MAISI VAE ✅ · CT-CLIP ✅ · FrozenCLIP3D ✅ · fVLM ✅ · GenerateCT ✅ · Report2CT ✅ · SPECTRE ✅ — all covered.

---

## MAISI VAE — frozen latent space · [src/baselines/maisi.py](../../src/baselines/maisi.py)

The fixed `[B,4,120,120,64]` latent space that every generator (Report2CT / text2ct)
diffuses in. Encode (CT→latent) and decode (latent→CT) only — no training.

**Build**
```python
from src.baselines.maisi import load_frozen
vae = load_frozen(device="cuda:0")     # AutoencoderKlMaisi; ALL params requires_grad=False, .eval()
```
- Never re-declare architecture kwargs — `load_frozen` instantiates `autoencoder_def` from
  `third_party/maisi_bundle/configs/inference.json` via `ConfigParser` + freeze + eval ([maisi.py:49-63](../../src/baselines/maisi.py#L49-L63)).
  Frozenness is enforced by `tests/test_maisi_frozen_load.py`.

**Input** (encode) — CT volume `(B, 1, 480, 480, 256)`, float, intensity **`[0,1]`**
(HU clipped `[-1000,1000]` → scaled `[0,1]`; `results/upper_bound.json`).
⚠️ `train_fixed/valid_fixed` already have HU baked in — **never re-apply metadata
RescaleSlope/Intercept** (intercept −8192 → all-air; memory `ctrate-fixed-hu-no-rescale`).

**Output** — latent `(B, 4, 120, 120, 64)` on encode; volume `(B, 1, 480, 480, 256)` ~`[0,1]` on decode.
Note encode returns **axis order `(B, C, H, W, D)`** (D last); ~7.2 MB/sample fp32 (the I/O bottleneck, memory `3d-latent-i-o-bottleneck`).

**Calls** (verified)
```python
# encode: volume -> latent          (report2ct_image_encoder.py:104-110)
with torch.autocast("cuda", dtype=torch.float16):   # REQUIRED — MaisiGroupNorm3D emits fp16 (baseline-clone #7)
    z = vae.encode_stage_2_inputs(x)                 # x (B,1,480,480,256) -> z (B,4,120,120,64)

# decode: latent -> volume          (src/eval/samplers/report2ct.py:65)
with torch.autocast("cuda", dtype=torch.float16):
    vol = vae.decode_stage_2_outputs(z)              # z (B,4,120,120,64) -> vol (B,1,480,480,256)
```

**Preprocessing / gotchas**
- **scale_factor is diffusion-only.** Training multiplies latents by `sf = 1/latent.std()`
  before the UNet ([report2ct_module.py:102,147](../../src/models/report2ct_module.py#L102)) and
  decode divides by it ([report2ct.py:65](../../src/eval/samplers/report2ct.py#L65) `_ReconModel`).
  A pure **encode→decode round-trip uses NO scale_factor** (identity).
- **Tiled decode for full volumes** (baseline-clone #5): wrap decode in a MONAI
  `SlidingWindowInferer` to avoid OOM / boundary-statistic drift — see `_ReconModel` +
  `_dynamic_infer` in [report2ct.py:41-78](../../src/eval/samplers/report2ct.py#L41-L78). A direct
  `decode_stage_2_outputs` is fine only for small/test latents.
- **Always under `autocast(fp16)`** on CUDA (both encode and decode) — `norm_float16: true`
  layers crash otherwise (baseline-clone #7).

**Saliency / grad note** — params are frozen (no *param* gradients) but the encode/decode
methods are **not** `@torch.no_grad()`, so **input/activation gradients DO flow** through the
VAE — usable for input-space saliency. Caveat: `autocast(fp16)` degrades gradient precision;
for saliency run fp32 (autocast off, CPU or GPU) at the cost of memory.

---

## CLIP-family text encoders — CT-CLIP vs text2ct FrozenCLIP3D

⚠️ **Two different "3D CLIP" models live in this repo — do not conflate them** (easy mistake):

| | **CT-CLIP** (our backbone) | **FrozenCLIP3D** (text2ct only) |
|---|---|---|
| adapter | [ctclip_adapter.py](../../src/baselines/ctclip_adapter.py) | inside [text2ct_adapter.py](../../src/baselines/text2ct_adapter.py) → `third_party/text2ct` |
| role | contrastive backbone for all 3 tasks | text→context conditioner for text2ct generation |
| text encoder | BiomedVLP-**CXR-BERT**-specialized | **CLIP-ViT-L/14** text_model |
| tokenizer / max_len | BertTokenizer / **512** | CLIP BPE / **77** |
| text output | `(B, 512)` L2, from **CLS** token | `(B, 1, 768)` L2, from **pooler + projection** |
| image encoder | CTViT 480/20/10 → `(B,512)` | CTViT 512/16/16 (vision side, unused in text→CT) |
| image input range | HU/1000 **[-1,1]**, H=W=480, D%10==0 | n/a for our path |
| weights | `CT-CLIP_v2.pt` | `CLIP3D_Finding_Impression_30ep.pt` |
| how we call it | `encode_image`/`encode_text`/`tokenize` directly | NOT direct — upstream `run_inference` consumes the module |

### CT-CLIP — contrastive backbone · [src/baselines/ctclip_adapter.py](../../src/baselines/ctclip_adapter.py)

**Build**
```python
from src.baselines.ctclip_adapter import CTCLIPBackbone
bb = CTCLIPBackbone(device_str="cuda:0")     # lazy: weights load on 1st encode_*/tokenize
```
- `image_dim == text_dim == 512`. Weights `data/checkpoints/ctclip/CT-CLIP_v2.pt`, loaded `strict=False`
  (benign transformers drift) (ctclip_adapter.py:54,186-189).
- **Device**: CT-CLIP hardcodes `cuda` in attention/ctvit — keep it on cuda:0; pick a physical GPU with
  `CUDA_VISIBLE_DEVICES=N`, **never** `device_str='cuda:1'` (ctclip_adapter.py:155-164).

**Input** — image `(B,1,D,480,480)`, **HU/1000 ∈ [-1,1]**, `D % 10 == 0` (CTViT patch divisibility);
text `str | list[str]` (ctclip_adapter.py:246-269).

**Output** — image & text latents both `(B,512)`, **L2-normalized** (cosine-ready) (ctclip_adapter.py:276,305).

**Calls** (verified)
```python
tok = bb.tokenize(reports)                                    # {input_ids, attention_mask} each (B,512)
img = bb.encode_image(vol)                                    # (B,1,D,480,480) -> (B,512) L2
txt = bb.encode_text(tok["input_ids"], tok["attention_mask"]) # (B,512) L2
sim = img @ txt.T                                             # cosine similarity
```
- encode_image: `visual_transformer(vol, return_encoded_tokens=True)` → temporal mean → flatten
  `(B,294912)` → `to_visual_latent` → L2 (:272-276). `return_encoded_tokens=True` is **mandatory** —
  without it CTViT runs the decoder path (weights absent in `CT-CLIP_v2.pt` → AttributeError).
- encode_text: **CLS** `last_hidden_state[:,0]` (NOT `pooler_output`) → `to_text_latent` → L2 (:300-305).

**Preprocessing / gotchas** — intensity **[-1,1]** is the inverse of MAISI's [0,1] (cross-encoder trap).
Pad/crop depth to a multiple of 10 before encode. Adapter asserts shape only — caller owns normalization (:263-269).

**Saliency / grad** — `encode_image`/`encode_text` are `@torch.no_grad()` → no backprop to input. For
Grad-CAM, use the grad-enabled module via the `bb.clip` property and drive `clip.visual_transformer(...)`
yourself (ctclip_adapter.py:206-217). This is the hook `tests/saliency_map/runners/ctclip_runner.py` uses.

### FrozenCLIP3D — text2ct conditioning encoder · `Text2CTAdapter` in [src/baselines/text2ct_adapter.py](../../src/baselines/text2ct_adapter.py)

text2ct's report→context encoder; **not the same as CT-CLIP**. Wraps `third_party/text2ct`
`core.models.encoders.clip.FrozenCLIP3D` (config-bank key `clip_3D`).

**Build** — constructed inside `Text2CTAdapter._ensure_built` (not standalone):
```python
cfgm = model_cfg_bank()("clip_3D"); clip = get_model()(cfgm).to(device)   # under cwd = text2ct root
clip.load_state_dict(torch.load("CLIP3D_Finding_Impression_30ep.pt"), strict=True)
```
(text2ct_adapter.py:295-313). **CUDA-only** — `FrozenCLIP3D.__init__` calls `self.model.cuda()` (clip.py:208).

**Input** — report **string**; text2ct conditions on the **impression only** (not findings+impression like
Report2CT). Base `openai/clip-vit-large-patch14`; CLIP BPE tokenizer, `max_length=77` (clip.py:174-182).

**Output** — `(B, 1, 768)`, **L2-normalized**: `pooler_output` → `text_projection` → L2 → `unsqueeze(1)`
(clip.py:262-273). Used directly as the UNet cross-attn `context`; CFG null context = `zeros_like(context)`
(diff_model_demo.py:130,167-168).

**Calls** — we do **not** call `encode_text` directly; the adapter hands the module to upstream
`run_inference`, which calls `clip([impression_text], "encode_text")`:
```python
vol = Text2CTAdapter(...).inference(prompt)   # -> (1,1,512,512,128) int16 HU; @torch.no_grad (text2ct_adapter.py:337-365)
```

**Preprocessing / gotchas** — output is **768-dim**, not 512 — wiring it where a CT-CLIP 512 context is
expected silently mismatches the UNet `cross_attention_dim`. The vision side (CTViT 512/16/16,
clip.py:191-205) is built but unused in the text→CT path.

**Saliency / grad** — used under `@torch.no_grad()` (inference only); not wired for gradients in our path.
For text-saliency, drive `FrozenCLIP3D.encode_text` outside the no_grad wrapper.

---

## fVLM — anatomy-aware organ encoder · [src/baselines/fvlm_adapter.py](../../src/baselines/fvlm_adapter.py)

> Deep rules auto-load from [fvlm.md](fvlm.md) when you edit `src/baselines/fvlm_*.py` — this is the
> I/O summary; fvlm.md carries the preprocessing + per-organ windowing detail.

Anatomy-aware: there is **no `encode_image`**. The forward needs `(volume, organ_mask)` and returns
**per-organ** ROI features. `FVLMBackbone` is a plain class (NOT `nn.Module`); the real trainable module
is `.model` (`BlipPretrain`) (fvlm_adapter.py:109-115).

**Build**
```python
from src.baselines.fvlm_adapter import FVLMBackbone
bb = FVLMBackbone(device_str="cuda:0")     # lazy; weights data/checkpoints/fvlm/, strict=False
```
Text encoder = `XBertEncoder` over `BiomedVLP-CXR-BERT-specialized`; image encoder = `ViT`
(fvlm_adapter.py:195,209-216). Organs = `(lung, heart, esophagus, aorta)`; all per-organ feats are
**256-dim, L2-normalized**.

**Inputs** — image `(1,1,D,H,W)` float + mask `(1,1,D,H,W)` int multilabel
`{0:bg, 1:lung, 2:heart, 3:esophagus, 4:aorta}`, both from `load_ct_and_mask_for_local`
([fvlm_preprocess.py](../../src/baselines/fvlm_preprocess.py)): resample 1×1×3 mm →
**`ScaleIntensityRange(-1150, 350)→[0,1]`** → bbox-crop → pad 112×256×352. Masks come from
`datasets/CT-RATE/dataset/ts_seg/` (TotalSegmentator v2, remapped via `resize.py` class_map — never hand-list).

**Outputs**
- `encode_organs(image, mask) -> {organ: (256,)}` — per-organ ROI **image** feat; organ-absent → skipped
  (fvlm_adapter.py:315-382). Per-organ `center_crop_organ(..., (112,288,352))` (per-organ ROI crop —
  distinct from the whole-volume preprocessing pad 112×256×352) + `divisible_pad_end` +
  `forward_test_win(skip_organ=id)`, once per organ — **not** sliding-window (dead code; see fvlm.md).
- `encode_organ_texts(organ_text: dict[str,str]) -> (len(organs), 256)` — per-organ **text** feat:
  CLS `last_hidden_state[:,0]` → `text_proj` → L2 (fvlm_adapter.py:274-313; `max_txt_len`=175, from upstream `BlipPretrain`).

**Preprocessing / gotchas**
- **Intensity window is (-1150, 350)→[0,1]** — NOT MAISI's [-1000,1000]→[0,1] nor CT-CLIP's HU/1000 (cross-encoder trap).
- all-zero (organ-absent) mask = 0 loss / 0 grad — skip it (fvlm.md); `encode_organs` already skips (`whole==0 → continue`).

**Saliency / grad** — `encode_organs`/`encode_organ_texts` are `@torch.no_grad()`. For grad-CAM, drive a
grad-enabled copy of `.model` (`forward_test_win`) outside the no_grad wrapper — see
`tests/saliency_map/runners/fvlm_runner.py`.

---

## GenerateCT — text→CT generator (CTViT + MaskGIT + super-res) · [src/baselines/generatect_adapter.py](../../src/baselines/generatect_adapter.py)

`GenerateCTAdapter(LightningModule)`, inference-only. Text is conditioned through an internal **T5**
encoder — **not** a CT-CLIP / CLIP3D.

**Build**
```python
from src.baselines.generatect_adapter import GenerateCTAdapter
gen = GenerateCTAdapter(device_str="cuda:0", load_super_resolution=False)   # lazy build
```
- text encoder: **`google/t5-v1_1-base`**, `MAX_LENGTH=256`, encoded dim **768** (= MaskGIT `dim_context`)
  (transformer_maskgit/t5.py:16-18; `_MASKGIT_KWARGS` dim_context 768, generatect_adapter.py:80-86).
- CTViT: image_size 128, patch 16, temporal_patch 2, dim 512 (generatect_adapter.py:66-76).
- 3 ckpts under `data/checkpoints/generatect/`: `ctvit_pretrained.pt`, `transformer_pretrained.pt`, `superres_pretrained.pt`.

**Input** — `prompt: str | list[str]` (free-text report).

**Output**
- `text_to_volume(prompt, num_frames=201, cond_scale=5.0) -> (1,1,201,128,128)` low-res (generatect_adapter.py:290-315).
- `text_to_volume_hires(prompt) -> (1,1,201,512,512)` per-2D-frame super-res; needs `load_super_resolution=True` (:317-354).
- `inference(prompt, ...)` dispatches on the SR flag (:356-375).

**Preprocessing / gotchas**
- **Voxel spacing** `output_spacing = (1.5, in_plane, in_plane)` in squeezed `(D,H,W)`; in_plane = 0.75 (hires) /
  3.0 (low-res) (generatect_adapter.py:257-273). The eval sampler stamps the SAME geometry in ITK `(x,y,z)`
  order — do NOT reconcile the two (audit #4; memory `ctgen-eval-spacing-convention`).
- HU rescale + int16 save live in `src/eval/samplers/generatect.py`, not in the adapter.
- Super-res `Unet` cross-attn: `cast_tuple` wraps the whole ListConfig so all layers get cross-attn —
  reproduced intentionally to match the checkpoint (generatect_adapter.py:213-219; baseline-clone #3).

**Saliency / grad** — `text_to_volume*` / `inference` are `@torch.no_grad()`; the adapter exists partly for
cross-attention / token-region diagnostics — drive the transformer / CTViT directly for grad.

---

## Report2CT — text encoder (2560-d) + MAISI UNet denoiser · [src/models/report2ct_module.py](../../src/models/report2ct_module.py)

Two pieces: a frozen **text encoder** (report → 2560-d context) and the **DiffusionModelUNetMaisi** denoiser
that diffuses in the MAISI latent space. (Training is user-owned; memory `report2ct-training-is-user-owned`.)

**Text encoder** — [src/baselines/report2ct_text_encoder.py](../../src/baselines/report2ct_text_encoder.py) `Report2CTTextEncoder`
- **3 frozen HF biomedical encoders, concatenated**: `abhinand/MedEmbed-large-v0.1` (1024) +
  `medicalai/ClinicalBERT` (768) + `microsoft/BiomedVLP-CXR-BERT-specialized` (768) = **2560**
  (report2ct_text_encoder.py:20-24,79). **Mean-pooled**, `max_seq_len=512`, all params frozen.
- `encode_pair(findings, impression) -> ((2560,), (2560,))` CPU float32 (:112-124). Training promotes each
  to `(B,1,2560)` and concatenates → context **`(B, 2, 2560)`** (report2ct_module.py:110-127 `_prepare_context`).

**UNet denoiser** — `DiffusionModelUNetMaisi` (Hydra-injected; kwargs in
[configs/model/report2ct.yaml](../../configs/model/report2ct.yaml), mirrored at report2ct.py:81-104)
- in/out channels **4**, `cross_attention_dim=2560`, `num_class_embeds=128`, `include_spacing_input=True`, `with_conditioning=True`.
- Input: latent `(B,4,120,120,64)` × `scale_factor`, context `(B,2,2560)`, spacing `(B,3)` (×1e2),
  modality class **1** (CT). Output: predicted target `(B,4,120,120,64)` (report2ct_module.py:129-182).

**Sampling / decode** — `src/eval/samplers/report2ct.py`: RFlow denoise → `_ReconModel` divides by
`scale_factor` → MAISI `decode_stage_2_outputs` via `SlidingWindowInferer` (see the **MAISI VAE** section).

**Saliency / grad** — `Report2CTTextEncoder.encode*` is `@torch.no_grad()` + frozen; the UNet is a trainable
denoiser (grad flows in training). No standalone image-encoder contract here.

---

## SPECTRE — CT 3D ViT foundation model (frozen REPA teacher) · [src/baselines/spectre_adapter.py](../../src/baselines/spectre_adapter.py)

`report2ct_wan` REPA 학습의 **정렬 타깃**. 이미지 전용 인코더라 CT-CLIP/fVLM의 `encode_image`/`encode_text`
계약을 공유하지 않는다 — 산출물은 **dense 3D token grid**다. 런북: [docs/repa_runbook.md](../../docs/repa_runbook.md)

**Build**
```python
from src.baselines.spectre_adapter import SpectreBackbone, CKPT_VLA
bb = SpectreBackbone(device_str="cuda")                      # SSL-only 백본 (기본)
bb = SpectreBackbone(ckpt_path=CKPT_VLA, with_combiner=True) # SSL+VLA + scan-level combiner
```
- ViT-L 338M: 24 layer, dim **1080**, patch **16×16×8**, crop **128×128×64** → crop당 8³=512 patch token (+CLS).
- 아키텍처 kwargs는 `spectre.presets`에서 읽는다 (하드코딩 안 함 → config drift 없음). `strict=True` 로드.
- 환경: `pip install loralib` + `huggingface_hub.load_state_dict_from_file` shim(`_install_hf_shim`).
  **`huggingface_hub`를 업그레이드하지 말 것** (transformers 4.46 / diffusers 0.31 의존).

**Input** — `(1, H, W, D)` channel-first **raw HU**. SPECTRE가 내부에서 `[-1000,1000] → [0,1]`을 적용한다
(cross-encoder trap: MAISI `[0,1]` / CT-CLIP HU÷1000 `[-1,1]` / fVLM `(-1150,350)→[0,1]` / Wan `[-1,1]`과 전부 다르다).
⚠ **모든 축이 crop_size의 배수여야 한다.** 아니면 upstream `window_scan`이 **center-crop**해 버린다
(253 slice → 192, 61 slice 소실). `window()`는 자르지 않고 `ValueError`를 던진다 — pad는 호출자 책임.

**Output**
- `encode_dense(vol_hu, layer=None, pool_to=None) -> (Gh, Gw, Gd, 1080)` — 512×512×256이면 `(32,32,32,1080)`.
  내부에서 **CLS를 제거**하고(`num_prefix_tokens == 1`) crop별 token을 전역 격자로 재조립한다.
- `encode_global(vol_hu) -> (1080,)` — combiner 경유 scan-level (VLA에서만 의미 있음; combiner가 VLA 위에서 학습됨).
- `window(vol_hu) -> (crops, grid)` / `encode_crops(crops, grid, layer=, want_global=)` — 한 번의 backbone
  pass로 dense와 global을 함께 얻는 저수준 경로 (precompute가 쓴다).

**Preprocessing / gotchas**
- token 순서는 crop·patch 둘 다 **C-order, depth가 가장 빨리** 변한다 (`grid_patch`: `n=(h·n_w+w)·n_d+d`;
  rope patch_embed `output_fmt='NHWDC'`). `tests/test_spectre_adapter.py`가 **모델 없는 인덱스 왕복**으로 고정한다.
- **TF32를 켤 것** — fp32 대비 3.7× 빠르고 token cosine 최저 0.99998 (`tests/repa_probe/u2b_io/`).
- teacher token에는 **iREPA spatial norm**을 걸어 쓴다 (`src/models/components/repa.py:spatial_norm`).
  raw token은 global 성분이 커서 무관한 토큰끼리도 cos 0.24가 나온다.
- crop 경계에 **seam**이 있다: 최종 레이어에서 이웃 cosine이 22.9 % 떨어진다. token-wise cosine 손실은
  영향 없고 **관계형 손실만** 오염된다 → `RepaAligner(rel_scope="within_crop")`으로 공짜 제거 가능.
- 가중치 라이선스 **CC-BY-NC-SA** (코드는 MIT).

**Saliency / grad** — `encode_*`는 `@torch.no_grad()`이고 파라미터는 전부 frozen. grad가 필요하면
`SpectreBackbone.model`(property)로 하위 모듈을 직접 구동한다 (CT-CLIP `.clip`과 같은 관례).

<!-- all encoders covered — keep entries in sync with the cited adapters when they change -->
