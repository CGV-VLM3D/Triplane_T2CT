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
  (LightningModule wrappers) and `src/eval/vlm3d_runner.py`, not by modifying submodule sources.
- Re-pin only after a deliberate upstream sync; update this file in the same commit.
