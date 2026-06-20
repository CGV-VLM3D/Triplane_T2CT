# VLM Baseline Weights — Download Runbook

User-owned download recipe for the two VLM baselines added in
[`/root/.claude/plans/baseline-models-are-recursive-manatee.md`](../../root/.claude/plans/baseline-models-are-recursive-manatee.md).
The assistant prepared the adapter code and Hydra configs; this document is
where the actual multi-GB weight downloads happen. Mirrors the
"user-owned training" convention used by Report2CT.

Target tree after running every section below:

```
/workspace/data/checkpoints/
├── ctclip/
│   ├── CT-CLIP_v2.pt           # symlink to /workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt
│   └── CT-VocabFine_v2.pt      # via huggingface-cli
├── fvlm/
│   ├── pretrained_ct_rate.pt   # via gdown
│   └── anatomy_descriptions/   # supplementary jsonl via gdown
└── hf_cache/                   # shared HF_HOME — prevents BiomedVLP-CXR-BERT re-downloads
```

## 0. Shared setup (run once)

```bash
# Shared HF cache so BiomedVLP-CXR-BERT-specialized is only downloaded once.
export HF_HOME=/workspace/data/checkpoints/hf_cache
echo 'export HF_HOME=/workspace/data/checkpoints/hf_cache' >> ~/.bashrc

# Tools the recipes below need.
pip install --quiet huggingface_hub gdown
```

Verify submodules are checked out (added by the assistant's plan step):

```bash
git submodule status third_party/ct_clip
# Should print one line with a SHA prefix (fVLM is a plain directory, not a submodule).
```

## 1. CT-CLIP

`CT-CLIP_v2.pt` is **already on disk** (used by the VLM3D-Dockers eval). We
just symlink it under the canonical training-side path so the adapter's
default `ckpt_path` resolves.

```bash
cd /workspace/data/checkpoints/ctclip
ln -sf /workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt CT-CLIP_v2.pt
ls -lh CT-CLIP_v2.pt   # should be a symlink to a ~1.4 GB file
```

Optional — vocab-finetuned variant used by zero-shot classification:

```bash
huggingface-cli download ibrahimhamamci/CT-RATE \
    models/CT-CLIP-Related/CT_VocabFine_v2.pt \
    --repo-type dataset \
    --local-dir /workspace/data/checkpoints/ctclip/ \
    --local-dir-use-symlinks False
# Note: lives in the CT-RATE *dataset* repo, not a separate model repo.
# File is CT_VocabFine (underscore), not CT-VocabFine (hyphen).
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/test_ctclip_adapter.py -v
```

Expected: `test_adapter_constructs_without_weights` and
`test_default_checkpoint_path_points_to_workspace_data` pass unconditionally;
`test_encode_image_and_text_with_real_weights` passes only after the symlink.

## 2. fVLM

Pretrained CT-RATE weights live in two gdrive folders (per
`third_party/fvlm/README.md`):

```bash
# Main pretrained weights (folder id from README).
cd /workspace/data/checkpoints/fvlm
gdown --folder https://drive.google.com/drive/folders/15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD

# Supplementary anatomy-wise decomposed descriptions (used by downstream tasks).
mkdir -p anatomy_descriptions
cd anatomy_descriptions
gdown --folder https://drive.google.com/drive/folders/10bz2UFxqxDPzl2P9NohESSNyBuld_Iek
```

After download, ensure the main checkpoint file is named
`pretrained_ct_rate.pt`. If gdown produced a differently-named file (the
upstream sometimes renames releases), rename or symlink:

```bash
cd /workspace/data/checkpoints/fvlm
ls -lh                                                  # find the actual .pt file
ln -sf <actual_filename>.pt pretrained_ct_rate.pt        # only if needed
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/test_fvlm_adapter.py -v
```

## 3. End-to-end Hydra compose check (no weights needed)

After section 0, even before downloading weights, the configs must compose:

```bash
cd /workspace
python -c "from hydra import initialize, compose; \
           initialize(version_base='1.3', config_path='configs'); \
           [print(v, '->', compose('train', overrides=[f'model=vlm_backbone/{v}']).model._target_) \
            for v in ('ctclip','fvlm')]"
```

Expected output (one line per backbone, no errors):

```
ctclip -> src.baselines.ctclip_adapter.CTCLIPBackbone
fvlm -> src.baselines.fvlm_adapter.FVLMBackbone
```

## 4. After both pass

Report the measured `image_dim` / `text_dim` per backbone from the
`requires_weights` test outputs to the assistant; those numbers are the
input contract for the follow-up plan that wires task modules
(`src/models/vlm/{classification,reportgen,retrieval}_module.py`).

## 5. NOT YET — organ-mask prerequisite (fVLM only)

fVLM is **anatomy-aware** — its forward pass requires per-volume organ
segmentation masks (lung / heart / esophagus / aorta). CT-RATE does **not**
ship masks; the upstream authors generated theirs with TotalSegmentator.

Those masks **already exist** under
`/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/ts_total/` (one
multilabel NIfTI per scan) — no precompute job is needed. The in-memory
preprocessing that reads them and remaps to `{lung:1, heart:2, esophagus:3,
aorta:4}` is `src/baselines/fvlm_preprocess.py:load_ct_and_mask_for_local`.

CT-CLIP does **not** need masks — it operates on volume + text only,
unchanged from the upstream zero-shot pipeline.
