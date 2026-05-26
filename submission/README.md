# VLM3D 2026 Task 4 submission (Phase A stub)

Reference: `third_party/vlm3d_dockers/ctgen_example_docker/README.md`.

## Contract

- **Input**: `/input/<anything>.json` — list of `{"input_image_name": str, "report": str}`.
- **Output**:
  - `/output/<name>.mha` per prompt (later zipped + removed)
  - `/output/predictions.zip` — final submission archive.
- **Volume shape**: 512 × 512 × 256, HU range `[-1000, 1000]`, isotropic spacing `(1.0, 1.0, 1.0)` mm.

## Local test (Phase A acceptance path, no docker daemon required)

```bash
bash submission/test_local.sh
```

Runs `process.py` directly via Python on the 5-prompt fixture at `submission/test/`.
Writes outputs to `/tmp/submission_test_out/`. Verifies `predictions.zip` exists with 5 entries.

## Docker test (Phase D production path)

```bash
bash submission/test.sh    # build + run + verify
bash submission/export.sh  # save as .tar.gz for upload
```

## Stub vs production

- **Stub (current)**: `process.py::generate_volume` emits a deterministic placeholder
  (gaussian noise mapped to HU range). Phase D will replace this with sampling from
  the trained `ours/final/best.ckpt` checkpoint.
- The I/O contract (paths, JSON schema, .mha shape/spacing/range, predictions.zip)
  is final. The generator is the only thing that swaps in Phase D.
