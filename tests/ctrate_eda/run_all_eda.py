#!/usr/bin/env python
"""Sequential runner for the CT-RATE EDA suite.

Executes each scripts/NN_*.py module in order via subprocess, prints progress,
and continues past a failing module (recording it) so one broken step does not
abort the whole run. The GPU-dependent latent module (A_latent) is skipped when
no CUDA device is visible unless --force-gpu is given.

Usage:
    python run_all_eda.py                 # run all (auto-skip GPU if none)
    python run_all_eda.py --root /workspace/tests/ctrate_eda
    python run_all_eda.py --only 03_label 04_report
    python run_all_eda.py --force-gpu     # run A_latent even if GPU probe fails
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Execution order: structure -> metadata -> label -> report -> headers ->
# voxel -> multimodal -> recon -> (GPU) latent -> spectrum.
MODULE_ORDER = [
    "01_validate_structure.py",
    "02_metadata.py",
    "03_label.py",
    "04_report.py",
    "05_nifti_header.py",
    "06_nifti_voxel.py",
    "07_multimodal.py",
    "08_recon.py",
    "A_latent.py",  # GPU (MAISI decode); skipped when no CUDA visible
    "B_spectrum.py",
]

# Modules that need a GPU (skipped when none is available).
GPU_MODULES = {"A_latent.py"}


def gpu_available() -> bool:
    """Best-effort CUDA probe: nvidia-smi present AND torch sees a device."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the CT-RATE EDA modules in order.")
    ap.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="EDA root dir (contains scripts/). Default: this file's dir.",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these module filenames (e.g. 03_label.py 04_report.py).",
    )
    ap.add_argument(
        "--force-gpu",
        action="store_true",
        help="Run GPU modules even if the CUDA probe fails.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        print(f"[FATAL] scripts dir not found: {scripts_dir}", file=sys.stderr)
        return 2

    has_gpu = args.force_gpu or gpu_available()
    print(f"[info] EDA root: {root}")
    print(f"[info] GPU available: {has_gpu} (force_gpu={args.force_gpu})")

    order = args.only if args.only else MODULE_ORDER
    results = []  # (module, status, seconds)

    for i, mod in enumerate(order, 1):
        script = scripts_dir / mod
        tag = f"[{i}/{len(order)}] {mod}"

        if not script.is_file():
            print(f"{tag}  -> SKIP (not found: {script})")
            results.append((mod, "missing", 0.0))
            continue

        if mod in GPU_MODULES and not has_gpu:
            print(f"{tag}  -> SKIP (needs GPU, none available)")
            results.append((mod, "skipped-gpu", 0.0))
            continue

        print(f"{tag}  -> RUN")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd="/workspace",  # scripts self-insert /workspace on sys.path
        )
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
        print(f"{tag}  -> {status} in {dt:.1f}s")
        results.append((mod, status, dt))

    print("\n=== EDA run summary ===")
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    for mod, status, dt in results:
        print(f"  {mod:26s} {status:14s} {dt:7.1f}s")
    print(f"  {n_ok}/{len(results)} modules OK")

    # Non-zero exit if any module actually failed (missing/skip are tolerated).
    any_fail = any(s.startswith("fail") for _, s, _ in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
