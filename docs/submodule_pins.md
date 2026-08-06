# third_party Submodule Pins

Pinned at Phase A Day 1 (2026-05-26). SHAs are "latest at clone time".
If upstream HEAD has broken in the intervening days, document and re-pin.

| Submodule | Path | URL | Pinned SHA |
|---|---|---|---|
| Report2CT | `third_party/report2ct` | https://github.com/sinaamirrajab/report2ct | `7b483a856ef159cfd0dada249b110d8f8eebf502` |
| GenerateCT | `third_party/generatect` | https://github.com/ibrahimethemhamamci/GenerateCT | `2a811356de351c67f89b2929c8bc9f2390797d9c` |
| VLM3D-Dockers | `third_party/vlm3d_dockers` | https://github.com/forithmus/VLM3D-Dockers | `a94590095847824d664e3a86f357924207f777fb` (re-pinned 2026-06-09; was `c73fe07`) |
| CT-CLIP | `third_party/ct_clip` | https://github.com/ibrahimethemhamamci/CT-CLIP | `a2a155c601987820433c01db69b64d701d3d229d` |
| fVLM | `third_party/fvlm` | https://github.com/alibaba-damo-academy/fvlm | `723a1f978a37c4dcce52b3f0562b926c0dc1c5c1` (**converted to regular directory 2026-06-08 — no longer a submodule**) |
| Text2CT | `third_party/text2ct` | https://github.com/danielemolino/Text2CT | `4fa286a64f128b71f1dddf24f9ad3b447241634e` |
| pytorch-grad-cam | `third_party/pytorch_grad_cam` | https://github.com/jacobgil/pytorch-grad-cam | `4e73d451ba0562ca134f5fb4f500a315a8b884b2` (added 2026-06-09; CAM engine for saliency Experiment 3; needs `pip install ttach`) |
| SPECTRE | `third_party/spectre` | https://github.com/cclaess/SPECTRE | `c05af47fe1d5f3d19836493508c8c9b1291671d4` (added 2026-07-28; CT 3D-ViT foundation model — REPA teacher. Code MIT, **weights CC-BY-NC-SA**. Needs `pip install loralib` + a `huggingface_hub.load_state_dict_from_file` shim, see below) |
| Foundation-VAE | `third_party/foundation_vae` | https://github.com/qic999/Foundation-VAE | `f891011e087bd0ce8cc5603baf8d5c333b54122e` (added 2026-07-14; reference for report2ct_wan — Wan CT VAE recon. `Reconstruction/Wan/` vendors the official Wan repo + `recon_ct_window_wan2.1.py` (native `Wan2_1_VAE`). We use diffusers `AutoencoderKLWan` instead; see [wan_latent_runbook.md](wan_latent_runbook.md)) |

## ⚠ Re-pinning VLM3D-Dockers: diff the *call sites*, not just the tree

When bumping `third_party/vlm3d_dockers`, compare `ct_challenges/ctgen_evaluation/evaluation.py`
(the container ENTRYPOINT) against the previous pin — **the arguments it passes to the metric
scripts are part of the metric definition**, and a changed argument is invisible in a normal
"did the code move?" review.

```bash
git -C third_party/vlm3d_dockers diff <old>:ctgen_evaluation/evaluation.py \
                                      <new>:ct_challenges/ctgen_evaluation/evaluation.py
```

This is not hypothetical: the `c73fe07 → a945900` bump (2026-06-09) added
`--model_name squeezenet1_1` to the 2.5D-FID call. We re-pinned for the path reorg and did not
notice, so our FID silently kept using the script's *default* feature network for seven weeks.
Found 2026-07-29; see [ctgen_local_eval.md](ctgen_local_eval.md) §2.5D-FID 프로파일.

## REPA-family reference clones (read-only, never imported)

`third_party/repa_refs/` holds shallow clones of the four REPA papers' code, kept locally only as
reading material while implementing `RepaAligner` — **nothing in `src/` imports them**. Gitignored
(`.gitignore: third_party/repa_refs/`); re-clone with `git clone --depth 1 <url> <path>`.

| Repo | Path | URL | SHA (2026-07-28) |
|---|---|---|---|
| REPA | `third_party/repa_refs/repa` | https://github.com/sihyun-yu/REPA | `67f714503e3892f993844aab088ffc5791c92613` |
| U-REPA | `third_party/repa_refs/u_repa` | https://github.com/YuchuanTian/U-REPA | `376c1df54255c955c9d0499edd9a36c615522a0d` |
| iREPA | `third_party/repa_refs/irepa` | https://github.com/end2end-diffusion/irepa | `99ad4ac234efe8de52ce157120f72856e836d09f` |
| VideoREPA | `third_party/repa_refs/videorepa` | https://github.com/aHapBean/VideoREPA | `8d581dc301e18546a1808436c724f59d259d09f9` |

## SPECTRE (added 2026-07-28) — REPA teacher encoder

CT-only 3D ViT foundation model (arXiv:2511.17209). Two published backbones under
`data/checkpoints/spectre/`: `..._no_vla.pt` (SSL only) and `..._patch16_128.pt` (SSL + vision-language),
plus `spectre_combiner_feature_vit_large.pt` (scan-level combiner). We load by **explicit path** —
`presets.py` registers only the VLA URL, so the SSL checkpoint has no preset route.

Two environment adaptations, both outside `third_party/` (P2 keeps the tree unpatched):
- `pip install loralib` — declared dependency of `spectre-fm`; `spectre/utils/__init__.py` imports `.lora`.
- a `huggingface_hub.load_state_dict_from_file` shim — `spectre/utils/modeling.py:14` imports it at module
  level and our `huggingface_hub` is 0.26.3 (added in 0.30+). It is only *called* on the HF-URL branch
  (`modeling.py:77`), which our local-path loads never take. Shim lives in `tests/repa_probe/_spectre.py`
  (`install_hf_shim`), same spirit as the numpy-2.x shim in `src/eval/tasks/_fid_runner.py`.
  Do **not** upgrade `huggingface_hub` — transformers 4.46 / diffusers 0.31 pin against it.

Verified end-to-end in [tests/repa_probe/u0_smoke/REPORT.md](../tests/repa_probe/u0_smoke/REPORT.md).

**Text2CT** (added 2026-06-03) is a report→3D-CT generator baseline on the same MAISI
latent-diffusion stack as Report2CT. Its sampler (`RFlowScheduler`) needs MONAI ≥ 1.5; we keep
MONAI pinned at 1.4 and vendor that one scheduler file at
`src/baselines/_vendored/rectified_flow.py` (see [text2ct_runbook.md](text2ct_runbook.md)).
Adapter: `src/baselines/text2ct_adapter.py` (inference-only; reuses upstream
`scripts.diff_model_demo.run_inference`).

**CT-CLIP duplication note**: the VLM3D eval dockers under
`third_party/vlm3d_dockers/ct_challenges/{reportgen_example_docker,abnclass_example_docker,ctgen_evaluation}/CT-CLIP/`
(a945900 moved everything under `ct_challenges/`; `ctgen_evaluation/CT-CLIP/` is empty in-repo and
populated only at docker build, so local CLIPScore uses a sibling docker's vendored copy via
`src/eval/_vlm3d_paths.py:ctclip_pkg_parents()`) each vendor their own copy of CT-CLIP, pinned to whatever SHA the docker
authors chose. We keep `third_party/ct_clip` separate so training-side
code (`src/baselines/ctclip_adapter.py`) can be re-pinned independently
of the eval-side copies. Do not collapse them — the eval copies are part
of the official eval container and must not drift from VLM3D-Dockers.

## Pin / restore

Restore exact pinned SHA after `git submodule update --init`:

```bash
git -C third_party/report2ct       checkout 7b483a856ef159cfd0dada249b110d8f8eebf502
git -C third_party/generatect      checkout 2a811356de351c67f89b2929c8bc9f2390797d9c
git -C third_party/vlm3d_dockers   checkout a94590095847824d664e3a86f357924207f777fb
git -C third_party/ct_clip         checkout a2a155c601987820433c01db69b64d701d3d229d
# third_party/fvlm is NOT a submodule anymore (converted 2026-06-08)
git -C third_party/text2ct         checkout 4fa286a64f128b71f1dddf24f9ad3b447241634e
git -C third_party/spectre         checkout c05af47fe1d5f3d19836493508c8c9b1291671d4
```

## fVLM modifications (regular directory, not a submodule)

`third_party/fvlm` was converted from a submodule to a plain tracked directory (2026-06-08)
so local patches can be committed freely. Original upstream SHA: `723a1f978a37c4dcce52b3f0562b926c0dc1c5c1`.
The following changes from upstream are in-tree:

### `lavis/models/blip_models/blip.py`

**Lines 22-23** — delete the `transformers<4.27` version assert (our env is 4.46; the actual
API breakage that guard was protecting against is in BLIP tokenizer/decoder paths fVLM
doesn't use):
```python
# DELETE:
transformers_version = version.parse(transformers.__version__)
assert transformers_version < version.parse("4.27"), "BLIP models are not ..."
```

**Line 27** — change tokenizer path from local dir to HF id:
```python
# BEFORE:  BertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")
# AFTER:   BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
```

### `blip_pretrain.py`

**Lines 344-364** — delete the MAE pretrain ckpt load block inside `from_config`
(`/storage/guoruizhe/...mae_pretrain_vit_base.pth`). The released CT-RATE checkpoint
overrides every ViT weight anyway, so this init is a no-op in practice. Keep
`image_encoder = model` on the line after.

**`from_config` method body, first line** — add `if cfg is None: cfg = {}` so the method
can be called without a cfg object.

### `lavis/models/med.py`

**Lines 1395-1406 (`XBertEncoder.from_config`)** — replace the `med_config_path` local-file
dance with HF id + four lavis-custom attrs that BertEmbeddings/BertSelfAttention read
unconditionally:
```python
@classmethod
def from_config(cls, cfg, from_pretrained=False):
    med_config = BertConfig.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    med_config.add_type_embeddings = False
    med_config.add_cross_attention = False
    med_config.encoder_width = med_config.hidden_size
    med_config.fusion_layer = med_config.num_hidden_layers

    if from_pretrained:
        model = cls.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            config=med_config,
            add_pooling_layer=False,
        )
    else:
        model = cls(config=med_config, add_pooling_layer=False)
```

### Adapter simplifications unlocked

After applying the three patches above, remove from `src/baselines/fvlm_adapter.py`:
- `_blip.BlipBase.__init__ = _LavisBase.__init__` monkey-patch
- `_blip.BlipBase.init_tokenizer = classmethod(...)` monkey-patch
- `_VIT_KWARGS` dict + manual `ViT(**_VIT_KWARGS)` construction
- Manual `BertConfig` + four attrs + `XBertEncoder.from_pretrained(...)` block (~20 lines)

Replace with a minimal cfg dict and `BlipPretrain.from_config(cfg)` call.

## Policy

- third_party/ is **read-only** (project Principle P2) with the exception of the fVLM patches
  documented above. Code adaptation otherwise happens in `src/baselines/*_adapter.py`
  (LightningModule wrappers) and `src/eval/` (samplers + `tasks/ctgen.py`), not by modifying submodule sources.
- Re-pin only after a deliberate upstream sync; update this file in the same commit.
