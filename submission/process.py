#!/usr/bin/env python3
"""VLM3D 2026 Task 4 submission entrypoint.

Implements the official ctgen contract documented at
`third_party/vlm3d_dockers/ctgen_example_docker/README.md`:

  /input/<file>.json    JSON list of {input_image_name, report}
  /output/<name>.mha    one .mha per prompt
  /output/predictions.zip   archive of all .mha files (originals removed)

Phase A Day 5 status: **stub** — emits deterministic placeholder volumes
(zeros, HU range, spacing (1, 1, 1)) so the I/O contract is proven end-to-end
before we have a trained `ours_final` model. Phase D production: swap the
`generate_volume` call to load `ours/final/best.ckpt` and sample text→volume.

Run standalone (no docker required, Phase A acceptance path):
    INPUT_DIR=submission/test OUTPUT_DIR=/tmp/sub_out \\
        python submission/process.py

Run inside docker (Phase D production path):
    bash submission/test.sh
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Paths (overridable via env so the same script works in dev + container).
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/opt/app/models"))

# Final output volume specification (from VLM3D-Dockers ctgen contract).
FINAL_SHAPE: tuple[int, int, int] = (512, 512, 256)  # (H, W, D)
FINAL_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)
HU_RANGE: tuple[int, int] = (-1000, 1000)


def find_prompts_json(input_dir: Path) -> Path:
    """Find the first .json file in /input (per ctgen contract)."""
    try:
        return next(input_dir.glob("*.json"))
    except StopIteration as exc:
        raise SystemExit(f"No JSON found in {input_dir}") from exc


def load_prompts(json_path: Path) -> list[dict]:
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(
            f"{json_path} must contain a JSON list, got {type(data).__name__}"
        )
    for i, item in enumerate(data):
        for k in ("input_image_name", "report"):
            if k not in item:
                raise SystemExit(f"Prompt {i} missing '{k}'")
    return data


def generate_volume(prompt: str, seed: int) -> np.ndarray:
    """Stub generator: return a deterministic placeholder volume.

    Phase A: zeros with mild gaussian noise so SimpleITK and metric containers
    have something non-degenerate to score. Phase D replaces this with the
    trained `ours/final/best.ckpt` sampler.
    """
    rng = np.random.default_rng(seed)
    vol = rng.normal(loc=0.5, scale=0.02, size=FINAL_SHAPE).astype(np.float32)
    vol = np.clip(vol, 0.0, 1.0)
    # Map [0, 1] -> HU range
    vol = vol * (HU_RANGE[1] - HU_RANGE[0]) + HU_RANGE[0]
    return vol.astype(np.int16)


def save_mha(volume: np.ndarray, out_path: Path) -> None:
    itk = sitk.GetImageFromArray(volume)
    itk.SetSpacing(FINAL_SPACING)
    itk.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(itk, str(out_path))


def zip_predictions(output_dir: Path) -> Path:
    zip_path = output_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mha in sorted(output_dir.glob("*.mha")):
            zf.write(mha, arcname=mha.name)
            mha.unlink()
    return zip_path


def main() -> None:
    prompts_path = find_prompts_json(INPUT_DIR)
    prompts = load_prompts(prompts_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(prompts)} prompts in {prompts_path}")
    for i, item in enumerate(prompts):
        name = Path(item["input_image_name"]).stem
        out_mha = OUTPUT_DIR / f"{name}.mha"
        vol = generate_volume(item["report"], seed=i)
        save_mha(vol, out_mha)
        print(f"  [{i + 1}/{len(prompts)}] wrote {out_mha}")

    zip_path = zip_predictions(OUTPUT_DIR)
    print(f"\nZipped {zip_path}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
