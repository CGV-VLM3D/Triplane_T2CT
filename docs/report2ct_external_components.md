# Report2CT External Components Pinned (Day 2)

Day 2 (2026-05-27) — confirmed by reading
`third_party/report2ct/vlm3d_inference.ipynb` (cell 0) and
`third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py`.

## Text encoders (3 of them)

The notebook does this once at preprocessing time:

```python
model_names = [
    "abhinand/MedEmbed-large-v0.1",
    "medicalai/ClinicalBERT",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
]
for name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).eval().to(device)
```

| HF model id | Hidden dim | Loader |
|---|---|---|
| `abhinand/MedEmbed-large-v0.1` | 1024 | `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` |
| `medicalai/ClinicalBERT` | 768 | `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` |
| `microsoft/BiomedVLP-CXR-BERT-specialized` | 768 | `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` |

Per-section pooling: `(weighted mean by attention mask)` → 1024 + 768 + 768 = **2560** per section.
Concatenated across `findings` + `impression` → conditioning tensor `(B, 2, 2560)`.

> Resolves Day-1 Open Question #3 (encoder ids) — RESOLVED.

## Text embeddings are precomputed, not in the training loop

The training script `diff_model_train_vlm3D_2560_multi_text.py` does NOT load text encoders.
It reads per-sample JSON files with precomputed embeddings:

```python
monai.transforms.Lambdad(keys="context_f", func=lambda x: _load_data_from_file(x, "findings_embeddings"))
monai.transforms.Lambdad(keys="context_i", func=lambda x: _load_data_from_file(x, "impression_embeddings"))
```

JSON filename convention (per train script L476):
`<image_filename>multi_2560.json` — contains `findings_embeddings` + `impression_embeddings` + spacing.

Implication: precompute step must materialize text embeddings BEFORE training. The
`vlm3d_inference.ipynb` cell 0 does this for inference; the training prerequisite uses
the same model bundle but iterates over CT-RATE train+valid reports.

## CFG dropout

`diff_model_train_vlm3D_2560_multi_text.py:297`: `context = torch.zeros_like(context)` triggered with prob 0.15. Matches paper text.

## MAISI image embeddings (precomputed too)

Format: `<volume_name>_emb.nii.gz` under `args.embedding_base_dir`. NOT `mu.pt` like our
existing `/workspace/datasets/datasets/latents/` collaborator-prepared cache. Two options
for Day 3+:

1. **Use submodule's `vlm3d_image_embedding.py`** to regenerate `_emb.nii.gz` on CT-RATE train_fixed (writes under `/workspace/data/` per our storage convention). Slower but matches submodule expectations exactly.
2. **Write a thin adapter** that converts `mu.pt` → `_emb.nii.gz` shape on the fly. Faster but couples to our cache format.

Day 3 decision: probably (1) for fidelity; revisit if disk space becomes an issue.

## UNet

Confirmed from `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json:diffusion_unet_def`:

- `_target_: monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi`
- `num_channels: [64, 128, 256, 512]` ✅ (paper matches)
- `attention_levels: [false, false, true, true]` ✅ (cross-attn at last 2 levels — matches paper)
- `cross_attention_dim: 2560` ✅
- `num_head_channels: [0, 0, 32, 32]`
- `with_conditioning: true`, `include_spacing_input: true`, `use_flash_attention: true`
- `num_class_embeds: 128`, `resblock_updown: true`

## Scheduler

`config_maisi_2560.json:noise_scheduler`:

- `_target_: monai.networks.schedulers.rectified_flow.RFlowScheduler` ✅
- `num_train_timesteps: 1000`, `scale: 1.4` ✅
- `use_timestep_transform: true`, `use_discrete_timesteps: false`, `sample_method: uniform`

## Training schedule

`config_maisi_diff_model_vlm3D.json:diffusion_unet_train`:

- `batch_size: 2`, `lr: 0.0001`, `n_epochs: 100`, `cache_rate: 0`

## License & non-commercial note

- All 3 text encoders are licensed for research use (MedEmbed / ClinicalBERT / BiomedVLP-CXR-BERT — check each card for exact terms).
- CT-CLIP for retrieval diagnostic: CC-BY-NC-SA (non-commercial research) — see `docs/ct_clip_check.md`.
- MAISI bundle: Apache 2.0 (MONAI Consortium).
- Report2CT submodule: license file present (`third_party/report2ct/LICENSE`), check Day 3 before adapter ships.

## Pin commit SHAs

| Component | Pinned at |
|---|---|
| Submodule SHAs | `docs/submodule_pins.md` |
| Text encoder HF revisions | TBD — Day 3 will record `model.config._commit_hash` after first `from_pretrained` call. |
| MAISI bundle | local copy at `maisi_bundle/` (no upstream sync needed) |
