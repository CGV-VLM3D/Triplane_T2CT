"""Pack per-patient ``mu.pt`` files into a single fp16 memmap shard.

Reads
-----
    <src_root>/<split>/<patient_id>/mu.pt   fp16, shape [4, H, W, D]
    <src_root>/stats.json                   (optional; copied through if found)

Writes
------
    <out_root>/<split>.npy                  fp16 memmap, shape [N, 4, H, W, D]
    <out_root>/<split>.index.json           patient_ids in row order
    <out_root>/stats.json                   (copied if present at src_root)

The dataset loader (src/data/maisi_latent_dataset.py) auto-detects the shard
layout: if ``<root>/<split>.npy`` is present it reads from the memmap;
otherwise it falls back to the per-file patient-dir layout.

Why
---
6,000 individual mu.pt files dominate epoch time via filesystem syscall
overhead (see CLAUDE.md "3D latent I/O bottleneck"). A single memmap-able
.npy lets workers share the OS page cache and skip per-file open/close.

Usage
-----
    # 2mm toy (small, fast): build both splits
    python scripts/build_latent_shard.py \\
        --src-root /workspace/data/latents_2mm \\
        --out-root /workspace/data/latents_2mm_shard

    # 1mm full (heavy: ~43 GB total)
    python scripts/build_latent_shard.py \\
        --src-root /workspace/datasets/datasets/latents \\
        --out-root /workspace/data/latents_1mm_shard

    # Single split only
    python scripts/build_latent_shard.py \\
        --src-root /workspace/data/latents_2mm \\
        --out-root /workspace/data/latents_2mm_shard \\
        --split valid
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def _list_patients(split_dir: Path) -> list[Path]:
    return sorted(
        p for p in split_dir.iterdir() if p.is_dir() and (p / "mu.pt").is_file()
    )


def build_split(
    src_root: Path, out_root: Path, split: str, dtype: str = "float16"
) -> None:
    split_dir = src_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} does not exist")
    patients = _list_patients(split_dir)
    if not patients:
        raise RuntimeError(f"No patient dirs with mu.pt under {split_dir}")

    sample_shape = tuple(
        torch.load(patients[0] / "mu.pt", map_location="cpu", weights_only=True).shape
    )
    n = len(patients)
    full_shape = (n,) + sample_shape
    np_dtype = np.dtype(dtype)
    nbytes = int(np.prod(full_shape)) * np_dtype.itemsize
    print(
        f"[shard] split={split}  n={n}  shape={sample_shape}  dtype={dtype}  "
        f"size~{nbytes / 1e9:.2f} GB"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    npy_path = out_root / f"{split}.npy"
    tmp_path = npy_path.with_suffix(".npy.tmp")
    idx_path = out_root / f"{split}.index.json"
    if tmp_path.exists():
        tmp_path.unlink()

    # open_memmap writes a proper .npy header so np.load(mmap_mode="r") works.
    arr = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np_dtype, shape=full_shape
    )
    index: list[str] = []
    for i, pdir in enumerate(patients):
        t = torch.load(pdir / "mu.pt", map_location="cpu", weights_only=True)
        if tuple(t.shape) != sample_shape:
            raise RuntimeError(
                f"shape mismatch at {pdir.name}: {tuple(t.shape)} != {sample_shape}"
            )
        arr[i] = t.to(torch.float16).numpy()
        index.append(pdir.name)
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"[shard]   {split} {i + 1}/{n}")
    arr.flush()
    del arr  # release the writable mmap before rename
    tmp_path.replace(npy_path)

    with open(idx_path, "w") as f:
        json.dump({"patient_ids": index, "shape": list(full_shape), "dtype": dtype}, f)
    print(f"[shard]   wrote {npy_path}  ({npy_path.stat().st_size / 1e9:.2f} GB)")

    src_stats = src_root / "stats.json"
    dst_stats = out_root / "stats.json"
    if src_stats.is_file() and not dst_stats.is_file():
        shutil.copy2(src_stats, dst_stats)
        print(f"[shard]   copied {src_stats} -> {dst_stats}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument(
        "--splits",
        default="train,valid",
        help="comma-separated splits (default: train,valid)",
    )
    p.add_argument("--split", default=None, help="single split (overrides --splits)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = p.parse_args()

    splits = (
        [args.split]
        if args.split
        else [s.strip() for s in args.splits.split(",") if s.strip()]
    )
    for split in splits:
        build_split(args.src_root, args.out_root, split, dtype=args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
