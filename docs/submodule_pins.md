# third_party Submodule Pins

Pinned at Phase A Day 1 (2026-05-26). SHAs are "latest at clone time".
If upstream HEAD has broken in the intervening days, document and re-pin.

| Submodule | Path | URL | Pinned SHA |
|---|---|---|---|
| Report2CT | `third_party/report2ct` | https://github.com/sinaamirrajab/report2ct | `7b483a856ef159cfd0dada249b110d8f8eebf502` |
| GenerateCT | `third_party/generatect` | https://github.com/ibrahimethemhamamci/GenerateCT | `2a811356de351c67f89b2929c8bc9f2390797d9c` |
| VLM3D-Dockers | `third_party/vlm3d_dockers` | https://github.com/forithmus/VLM3D-Dockers | `c73fe07308c9393c94f412a57e874d25792813bf` |
| CT-CLIP | `third_party/ct_clip` | https://github.com/ibrahimethemhamamci/CT-CLIP | `a2a155c601987820433c01db69b64d701d3d229d` |
| fVLM | `third_party/fvlm` | https://github.com/alibaba-damo-academy/fvlm | `723a1f978a37c4dcce52b3f0562b926c0dc1c5c1` |
| ViSD-Boost | `third_party/visd_boost` | https://github.com/alibaba-damo-academy/ViSD-Boost | `42d18a9c3d389284ae003326ec32c245f49c3608` |
| Text2CT | `third_party/text2ct` | https://github.com/danielemolino/Text2CT | `4fa286a64f128b71f1dddf24f9ad3b447241634e` |

**Text2CT** (added 2026-06-03) is a report→3D-CT generator baseline on the same MAISI
latent-diffusion stack as Report2CT. Its sampler (`RFlowScheduler`) needs MONAI ≥ 1.5; we keep
MONAI pinned at 1.4 and vendor that one scheduler file at
`src/baselines/_vendored/rectified_flow.py` (see [text2ct_runbook.md](text2ct_runbook.md)).
Adapter: `src/baselines/text2ct_adapter.py` (inference-only; reuses upstream
`scripts.diff_model_demo.run_inference`).

**CT-CLIP duplication note**: the VLM3D eval dockers under
`third_party/vlm3d_dockers/{reportgen_example_docker,abnclass_example_docker,ctgen_evaluation}/CT-CLIP/`
each vendor their own copy of CT-CLIP, pinned to whatever SHA the docker
authors chose. We keep `third_party/ct_clip` separate so training-side
code (`src/baselines/ctclip_adapter.py`) can be re-pinned independently
of the eval-side copies. Do not collapse them — the eval copies are part
of the official eval container and must not drift from VLM3D-Dockers.

## Pin / restore

Restore exact pinned SHA after `git submodule update --init`:

```bash
git -C third_party/report2ct       checkout 7b483a856ef159cfd0dada249b110d8f8eebf502
git -C third_party/generatect      checkout 2a811356de351c67f89b2929c8bc9f2390797d9c
git -C third_party/vlm3d_dockers   checkout c73fe07308c9393c94f412a57e874d25792813bf
git -C third_party/ct_clip         checkout a2a155c601987820433c01db69b64d701d3d229d
git -C third_party/fvlm            checkout 723a1f978a37c4dcce52b3f0562b926c0dc1c5c1
git -C third_party/visd_boost      checkout 42d18a9c3d389284ae003326ec32c245f49c3608
git -C third_party/text2ct         checkout 4fa286a64f128b71f1dddf24f9ad3b447241634e
```

## Policy

- third_party/ is **read-only** (project Principle P2). Code adaptation happens in
  `src/baselines/*_adapter.py` (LightningModule wrappers) and `src/eval/vlm3d_runner.py`,
  not by modifying submodule sources.
- Re-pin only after a deliberate upstream sync; update this file in the same commit.
