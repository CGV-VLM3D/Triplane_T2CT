#!/usr/bin/env python3
"""Precompute frozen SPECTRE teacher features for report2ct_wan REPA training.

Twin of `scripts/precompute_wan_image_embeddings.py` (Wan VAE latents) on the teacher side.
Each CT is resampled to the **exact same fixed geometry** the Wan latents were built on, so a
teacher token and a Wan latent voxel address the same physical tissue:

  CT (X, Y, Z) --Resized-> (512, 512, 253) --z end-pad-> (512, 512, 256)
     --window_scan(128,128,64)-> 4×4×4 = 64 crops --SPECTRE ViT-L-> per-crop (1+512, 1080)
     --drop CLS + reassemble--> dense token grid (32, 32, 32, 1080)

⚠ The z end-pad is load-bearing. `spectre/windowing.py` CENTER-CROPS to the largest whole
multiple of the crop size; a 253-slice volume would silently lose 61 slices (24 %) and break
the correspondence with the latent. `SpectreBackbone.window` raises rather than crop, and this
script does the pad. See tests/repa_probe/u0_smoke/REPORT.md.

Two teachers are extracted in ONE pass over each volume because loading + resampling the CT is
~80 % of the cost:
  ssl — spectre_backbone_vit_large_patch16_128_no_vla.pt  (DINO + iBOT + KoLeo)
  vla — spectre_backbone_vit_large_patch16_128.pt         (+ SigLIP report alignment)

Both resolutions are written: 32³ is the paper-faithful target (U-REPA Table 6 shows
downscaling the teacher costs FID) and 16³ is the I/O-cheap fallback; which one training uses
is decided by measurement in tests/repa_probe/u2b_io/. 16³ is an average-pool of 32³, so
writing both costs no extra compute.

Intensity: raw HU is handed to SPECTRE, which windows [-1000, 1000] → [0, 1] itself. CT-RATE
`_fixed` inputs already have HU baked in (never re-apply slope/intercept,
[[ctrate-fixed-hu-no-rescale]]).

Cache hygiene: a `meta.json` (geometry + checkpoint + provenance) is written into each output
dir on the first run and VERIFIED on later runs; a mismatch aborts BEFORE writing.

Usage (GPU 2, 4 shards to hide the CPU-bound load):
    for S in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=2 nohup python scripts/precompute_spectre_features.py \\
        --ids-file /workspace/data/ctrate_toy_v2/train/ids.json \\
        --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \\
        --out-root /workspace/data/report2ct_wan \\
        --num-shards 4 --shard $S \\
        > logs/spectre_precompute_s$S_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    done
Run again with the valid_v2 ids-file + valid_fixed into the SAME --out-root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.spectre_adapter import (  # noqa: E402
    CKPT_COMBINER,
    CKPT_VLA,
    DEFAULT_CKPT,
    SpectreBackbone,
)

#: teacher tag → backbone checkpoint. `vla` additionally gets the scan-level combiner, which
#: was trained on top of the VLA backbone — running it on SSL features would be off-distribution,
#: so the global vector is a VLA-only product.
TEACHERS: dict[str, Path] = {"ssl": DEFAULT_CKPT, "vla": CKPT_VLA}
GLOBAL_TEACHERS: frozenset[str] = frozenset({"vla"})

IN_PLANE = 512
DEPTH_RESAMPLE = (
    253  # matches the Wan latent grid (causal ×4 compression is lossless here)
)
DEPTH_PAD = 256  # multiple of the SPECTRE crop depth (64)
AIR_HU = -1000.0


def id_to_nifti_path(vol_id: str, ct_rate_dir: Path) -> Path:
    """``train_10000_a_1`` → ``<ct_rate_dir>/train_10000/train_10000_a/train_10000_a_1.nii.gz``."""
    parts = vol_id.split("_")
    return ct_rate_dir / "_".join(parts[:2]) / "_".join(parts[:3]) / f"{vol_id}.nii.gz"


def build_resample_transforms():
    """Resample-only MONAI pipeline — transcribed from the Wan precompute so the grids match.

    Source: `scripts/precompute_wan_image_embeddings.py` `build_resample_transforms`. No
    intensity scaling: SPECTRE consumes raw HU and applies its own [-1000, 1000] → [0, 1]
    window inside `window_scan`.
    """
    import monai  # noqa: PLC0415
    from monai.transforms import Compose  # noqa: PLC0415

    return Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
            monai.transforms.Resized(
                keys="image",
                spatial_size=(IN_PLANE, IN_PLANE, DEPTH_RESAMPLE),  # (X, Y, Z)
                mode="trilinear",
            ),
        ]
    )


def read_ids(ids_file: str) -> list[str]:
    """Load volume IDs from a `.json` (`{"ids": [...]}` or a bare list) or a `.txt`."""
    path = Path(ids_file)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return list(data["ids"] if isinstance(data, dict) else data)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", "/workspace", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


def build_meta(teacher: str, ckpt: Path, grid: tuple[int, int, int]) -> dict:
    """Provenance written once per output dir and verified on every later run."""
    return {
        "teacher": teacher,
        "checkpoint": ckpt.name,
        "checkpoint_bytes": ckpt.stat().st_size,
        "in_plane": IN_PLANE,
        "depth_resample": DEPTH_RESAMPLE,
        "depth_pad": DEPTH_PAD,
        "pad_value_hu": AIR_HU,
        "crop_size": list(SpectreBackbone.crop_size),
        "patch_size": list(SpectreBackbone.patch_size),
        "token_grid": list(grid),
        "embed_dim": SpectreBackbone.dense_dim,
        "cls_dropped": True,
        "layer": "final (forward_features)",
        "dtype": "float16",
        "matmul_precision": "tf32" if torch.backends.cuda.matmul.allow_tf32 else "fp32",
        "token_order": "C-order (H, W, D), depth fastest",
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
    }


def check_or_write_meta(out_dir: Path, meta: dict) -> None:
    """Write `meta.json` on first use; abort on any geometry/checkpoint drift afterwards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    volatile = {"created", "git_sha"}
    if not path.is_file():
        path.write_text(json.dumps(meta, indent=2))
        return
    old = json.loads(path.read_text())
    diff = {
        k: (old.get(k), meta[k])
        for k in meta
        if k not in volatile and old.get(k) != meta[k]
    }
    if diff:
        raise SystemExit(
            f"meta.json mismatch in {out_dir}: {diff}\n"
            f"Features built at a different geometry/checkpoint must not be mixed. "
            f"Use a fresh --out-root or delete the directory."
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ids-file", required=True, help="Volume IDs (.json or .txt)")
    p.add_argument(
        "--ct-rate-dir", required=True, help="CT-RATE {train,valid}_fixed/ directory"
    )
    p.add_argument(
        "--out-root",
        default="/workspace/data/report2ct_wan",
        help="Parent of the spectre_<teacher>_<grid>/ output dirs",
    )
    p.add_argument(
        "--teachers",
        default=",".join(TEACHERS),
        help=f"Comma-separated subset of {sorted(TEACHERS)}",
    )
    p.add_argument(
        "--resolutions",
        default="32,16",
        help="Comma-separated token-grid sizes to write (32 = native, 16 = avg-pooled)",
    )
    p.add_argument(
        "--device", default="cuda:0", help="Torch device (cuda:0 under CVD=N)"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N volumes (smoke)"
    )
    p.add_argument(
        "--num-shards", type=int, default=1, help="Split ids into N interleaved shards"
    )
    p.add_argument("--shard", type=int, default=0, help="This process's shard index")
    p.add_argument(
        "--max-crops-per-forward",
        type=int,
        default=16,
        help="Backbone memory cap (crops)",
    )
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument(
        "--no-tf32",
        dest="tf32",
        action="store_false",
        help="Disable TF32 matmul (3.7x slower). Default on: measured cos_mean 0.9999999 / "
        "cos_min 0.99998 and SRSS 0.128084 vs fp32 0.128078 on valid_1000_a_1 — an order of "
        "magnitude below any threshold that matters, and better-behaved than bf16 "
        "(cos_min 0.9956). See tests/repa_probe/u2b_io/results/precision.json.",
    )
    p.set_defaults(skip_existing=True, tf32=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # Teacher extraction is GPU-bound at fp32 (ViT-L matmuls never reach the tensor cores).
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    if bad := [t for t in teachers if t not in TEACHERS]:
        raise SystemExit(f"unknown teacher(s) {bad}; choose from {sorted(TEACHERS)}")
    grids = sorted((int(r) for r in args.resolutions.split(",")), reverse=True)
    if 32 not in grids:
        raise SystemExit(
            "--resolutions must include 32 (the other sizes are pooled from it)"
        )

    ids = read_ids(args.ids_file)[args.shard :: args.num_shards]
    if args.limit:
        ids = ids[: args.limit]
    ct_rate_dir = Path(args.ct_rate_dir)
    out_root = Path(args.out_root)

    native = SpectreBackbone.token_grid((IN_PLANE, IN_PLANE, DEPTH_PAD))  # (32, 32, 32)
    out_dirs = {(t, g): out_root / f"spectre_{t}_{g}" for t in teachers for g in grids}
    for (teacher, g), d in out_dirs.items():
        meta = build_meta(teacher, TEACHERS[teacher], native if g == 32 else (g, g, g))
        meta["pooled_from"] = list(native) if g != 32 else None
        check_or_write_meta(d, meta)

    print(
        f"[shard {args.shard}/{args.num_shards}] {len(ids)} volumes, teachers={teachers}, grids={grids}"
    )

    # Build every teacher up front — 338M params each, ~2.5 GB peak activation, so all of them
    # plus the combiner fit comfortably next to 3 sibling shards on one 96 GB card.
    backbones = {
        t: SpectreBackbone(
            ckpt_path=TEACHERS[t],
            device_str=args.device,
            with_combiner=t in GLOBAL_TEACHERS,
            combiner_ckpt_path=CKPT_COMBINER,
        )
        for t in teachers
    }
    transforms = build_resample_transforms()

    n_done = n_skipped = n_failed = 0
    for vol_id in tqdm(
        ids, desc=f"shard{args.shard}", file=sys.stdout, mininterval=10.0
    ):
        targets = {
            (t, g): out_dirs[(t, g)] / f"{vol_id}.npy" for t in teachers for g in grids
        }
        if args.skip_existing and all(p.is_file() for p in targets.values()):
            n_skipped += 1
            continue

        nifti = id_to_nifti_path(vol_id, ct_rate_dir)
        if not nifti.is_file():
            print(f"  [miss] {vol_id}: {nifti}", flush=True)
            n_failed += 1
            continue

        try:
            vol = transforms({"image": str(nifti)})[
                "image"
            ]  # (1, 512, 512, 253) raw HU
            vol = vol.as_tensor() if hasattr(vol, "as_tensor") else vol
            vol = torch.nn.functional.pad(
                vol, (0, DEPTH_PAD - vol.shape[-1]), value=AIR_HU
            )  # (1, 512, 512, 256)

            first = backbones[teachers[0]]
            crops, crop_grid = first.window(vol)  # (64, 1, 128, 128, 64), (4, 4, 4)

            for teacher in teachers:
                dense, global_ = backbones[teacher].encode_crops(
                    crops,
                    crop_grid,
                    want_global=teacher in GLOBAL_TEACHERS,
                    max_crops_per_forward=args.max_crops_per_forward,
                )  # (32, 32, 32, 1080)
                for g in grids:
                    grid_feat = (
                        dense
                        if g == 32
                        else SpectreBackbone.pool_grid(dense, (g, g, g))
                    )
                    flat = grid_feat.reshape(-1, grid_feat.shape[-1])  # (g³, 1080)
                    np.save(targets[(teacher, g)], flat.to(torch.float16).cpu().numpy())
                if global_ is not None:
                    np.save(
                        out_dirs[(teacher, 32)] / f"{vol_id}_global.npy",
                        global_.to(torch.float16).cpu().numpy(),
                    )
                del dense
            n_done += 1
        except Exception as exc:  # noqa: BLE001 — one bad scan must not kill the shard
            print(f"  [fail] {vol_id}: {type(exc).__name__}: {exc}", flush=True)
            n_failed += 1

    print(f"[shard {args.shard}] done={n_done} skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
