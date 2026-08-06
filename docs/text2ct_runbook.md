# Text2CT baseline — runbook

Text2CT (danielemolino/Text2CT, Apache-2.0) is a report → 3D-CT generator built on the **same
MAISI latent-diffusion stack as Report2CT**: a frozen MAISI VAE, a `DiffusionModelUNetMaisi`
sampled with a rectified-flow scheduler, and a `FrozenCLIP3D` text encoder for cross-attention
conditioning. We add it as a generation baseline to compare against `ours_final` and
`report2ct_our_repro` on the VLM3D-Dockers metrics.

- Submodule: `third_party/text2ct` (pinned — see [submodule_pins.md](submodule_pins.md)).
- Adapter: [src/baselines/text2ct_adapter.py](../src/baselines/text2ct_adapter.py) — inference-only,
  reuses upstream `scripts.diff_model_demo.run_inference` verbatim.
- Config: [configs/model/text2ct.yaml](../configs/model/text2ct.yaml) — swap in via `model=text2ct`.
- Tests: [tests/test_text2ct_adapter.py](../tests/test_text2ct_adapter.py).

## 0. MONAI version note (already handled in-repo)

Text2CT samples with `RFlowScheduler`, which only exists in **MONAI ≥ 1.5**; this repo is
pinned to **MONAI 1.4** (the Report2CT adapter and the frozen MAISI VAE round-trip are
validated against 1.4). Rather than bump the global dependency, we vendor the single MONAI-1.5
scheduler file at [src/baselines/_vendored/rectified_flow.py](../src/baselines/_vendored/rectified_flow.py)
and register it under the `monai.networks.schedulers` namespace at build time. No action needed.

Similarly, Text2CT vendors a `transformers==4.30` copy of the CLIP tokenizer, which breaks under
our `transformers` 4.46 (`get_vocab()` is called before `self.encoder` is set). The adapter
swaps in the real `transformers.CLIPTokenizer`/`CLIPProcessor` at build time — they are the same
`openai/clip-vit-large-patch14` tokenizer, so token ids are identical (verified BOS=49406). The
vendored `CLIP3DModel`/`CLIPConfig` (the architecture the checkpoint expects) are kept as-is. No
action needed; both shims live in `src/baselines/text2ct_adapter.py`.

## 0b. Extra Python deps

Two packages from Text2CT's `requirements.txt` are not in the base image and are needed by the
inference build path (`easydict` → `core.cfg_helper`; `decord` → pulled in transitively when
CTViT's package `__init__` imports `videotextdataset`). Install once:

```bash
pip install "easydict==1.13" "decord==0.6.0"
```

## 1. Weights (user-owned download)

Three checkpoints from HuggingFace `dmolino/text2ct-weights` land under
`/workspace/data/checkpoints/text2ct/`:

| File | Role | Built via |
|---|---|---|
| `autoencoder_epoch273.pt` | MAISI VAE | `autoencoder_def` |
| `unet_rflow_200ep.pt` | diffusion UNet (+ `scale_factor` in the dict) | `diffusion_unet_def` |
| `CLIP3D_Finding_Impression_30ep.pt` | FrozenCLIP3D text/vision encoder | `model_cfg_bank("clip_3D")` |

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="dmolino/text2ct-weights",
    repo_type="model",
    local_dir="/workspace/data/checkpoints/text2ct",
)
PY
ls -lh /workspace/data/checkpoints/text2ct/   # expect the 3 .pt files above
```

> The adapter loads all three with `strict=True` (UNet/CLIP3D) — if HF/transformers/MONAI
> version drift trips a key mismatch, surface it; do **not** patch `third_party/` (read-only).

### HF base encoder cache

`FrozenCLIP3D` builds a `CLIP3DModel` from `openai/clip-vit-large-patch14` (tokenizer + config),
so that model must be reachable (network or `HF_HOME` cache). The repo's shared cache is
`/workspace/data/checkpoints/hf_cache` (`export HF_HOME=...`).

## 2. (Fast path) evaluate their 1,000 released synthetic scans first

The authors released ~1,000 pre-generated chest-CT volumes for the VLM3D challenge. Scoring
those directly with `CTGenEvaluator` ([src/eval/tasks/ctgen.py](../src/eval/tasks/ctgen.py))
yields a first Text2CT number with zero new model code — do this before/while wiring the adapter
to de-risk. Confirm the exact HF
repo id/path for the *scans* at download time (it differs from the weights repo) and place them
under `/workspace/data/checkpoints/text2ct/released_scans/`.

## 3. Smoke tests

```bash
# CI-safe (no weights, no GPU): construct + vendored-scheduler + shim checks
pytest tests/test_text2ct_adapter.py -k "not requires_weights" -v

# Hydra compose sanity
python -c "from hydra import initialize, compose; \
           initialize(version_base='1.3', config_path='configs'); \
           print(compose('train', overrides=['model=text2ct']).model._target_)"

# With weights, on a GPU (FrozenCLIP3D is CUDA-only):
CUDA_VISIBLE_DEVICES=0 pytest tests/test_text2ct_adapter.py -k requires_weights -v
```

## 4. Generate on our split + evaluate

```bash
# Generate volumes for our prompts (uses model.inference(text) -> (1,1,512,512,128) int16 HU)
CUDA_VISIBLE_DEVICES=0 python src/inference.py model=text2ct \
    output_dir=/workspace/data/text2ct_preds   # + prompts=... per configs/inference.yaml

# Score the generated dir against valid_v2 GT (CTGenEvaluator runs the docker scripts directly)
python -c "from pathlib import Path; from src.eval.tasks.ctgen import CTGenEvaluator; \
    CTGenEvaluator(gt_dir='/workspace/data/ctrate_toy_v2/valid_v2').evaluate( \
        Path('/workspace/data/text2ct_preds'), Path('/workspace/data/text2ct_eval'))"
```

> Spacing: upstream saves with affine `diag(0.75, 0.75, 3.0)`. `src/inference.py` currently
> saves with `eye(4)`; if the eval is spacing-sensitive, set the spacing on the saved NIfTI to
> match upstream (`out_spacing = (0.75, 0.75, 3.0)`).

## Out of scope

Text2CT **training** (user-owned, like Report2CT). Reusing their `FrozenCLIP3D` as a VLM
backbone is a possible future follow-up, not part of this baseline.
