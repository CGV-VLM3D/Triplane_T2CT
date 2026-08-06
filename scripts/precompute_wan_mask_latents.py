#!/usr/bin/env python3
"""Precompute per-volume Wan2.1 VAE **mask** latents for report2ct_wan_mask conditioning.

Twin of `scripts/precompute_wan_image_embeddings.py`, but the input is the ts_seg
TotalSegmentator label map (not the CT). Each mask is reduced to our 4 흉부 organ groups
(`src/data/organ_groups.py::apply_grouping` → labels {0=bg,1=lung,2=heart,3=aorta,4=esophagus}),
painted into a grayscale HU volume, resampled to the SAME fixed geometry as the image latents,
and encoded with the SAME frozen `WanVAE` — so the mask latent lands in the exact same normalized
Wan diffusion space as the image latent and can be channel-concatenated at the UNet input.

Painting (bg=-1, organs evenly spread over WAN's [-1,1] input; see the approved plan):
  hu = -1000 + 500 * group_id  →  {bg:-1000, lung:-500, heart:0, aorta:+500, esoph:+1000}
`WanVAE.encode` clamps to [-1000,1000] and divides by 1000 → {-1,-0.5,0,+0.5,+1} in VAE input space.

Geometry (mirrors the image script; see docs/wan_latent_runbook.md):
  ts_seg (X,Y,Z) --Orientation(RAS)--Resized(NEAREST)-> (512,512,253) --apply_grouping+paint->
     HU volume --permute(2,0,1)-> (T=253,H=512,W=512) --WanVAE.encode-> latent (16,64,64,64)
     --rearrange-> (64,64,64,16) HWDC, saved as <id>_mask_emb.nii.gz.
  NEAREST (not the image script's trilinear) because labels must never be blended.

Cache hygiene: a `meta.json` (geometry + VAE id + paint scheme + provenance) is written into
--out-dir on the first run and VERIFIED on later runs; a mismatch aborts before writing.

Runs in the dedicated `wan` conda env (diffusers ≥ 0.34). GPU 1:
  CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/wan/bin/python \\
      scripts/precompute_wan_mask_latents.py \\
      --ids-file /workspace/data/ctrate_toy_v2/train/ids.json \\
      --split train_fixed \\
      --out-dir /workspace/data/report2ct_wan/mask_latents_512x512x253 --device cuda:0 --limit 100

Run again with the valid_v2 ids-file + --split valid_fixed into the SAME --out-dir (train_*/valid_*
filenames distinguish them; the shared meta.json is re-verified). --ids-file accepts a .json
({"ids": [...]} or a bare list) or a .txt. Full run: omit --limit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
from einops import rearrange
from monai.transforms import Compose
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.wan_vae import DEFAULT_WAN_HF_ID, WanVAE  # noqa: E402
from src.data.organ_groups import GROUP_NAMES, apply_grouping  # noqa: E402

# ts_total lives under the CT-RATE ts_seg dir; masks mirror {train,valid}_fixed/<patient>/<study>/.
DEFAULT_TS_DIR = "/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/ts_total"
# Paint step: group_id 0..4 -> HU (WanVAE then clamp/1000 -> [-1,-0.5,0,0.5,1]).
HU_MIN, HU_STEP = -1000.0, 500.0


def ts_mask_path(vol_id: str, ts_dir: Path, split: str) -> Path:
    """Derive the ts_seg label path from a volume ID (mirrors precompute_mask_latentgrid.mask_path_for).

    ``train_10000_a_1`` → ``<ts_dir>/<split>/train_10000/train_10000_a/train_10000_a_1.nii.gz``.
    """
    parts = vol_id.split("_")
    patient = "_".join(parts[:2])
    series = "_".join(parts[:3])
    return ts_dir / split / patient / series / f"{vol_id}.nii.gz"


def build_resample_transforms(in_plane: int, depth: int) -> Compose:
    """Resample-only MONAI transforms to a fixed (X=in_plane, Y=in_plane, Z=depth) grid.

    NEAREST interpolation (labels must never be blended); grouping + HU painting happen after,
    on the resampled label array. No intensity scaling — WanVAE consumes raw HU.
    """
    return Compose(
        [
            monai.transforms.LoadImaged(keys="mask"),
            monai.transforms.EnsureChannelFirstd(keys="mask"),
            monai.transforms.Orientationd(keys="mask", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="mask", dtype=torch.float32),
            monai.transforms.Resized(
                keys="mask",
                spatial_size=(in_plane, in_plane, depth),  # (X, Y, Z)
                mode="nearest",
            ),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute Wan2.1 VAE mask latents for report2ct_wan_mask"
    )
    p.add_argument(
        "--ids-file", required=True, help="One volume ID per line (.json/.txt)"
    )
    p.add_argument(
        "--split",
        required=True,
        choices=["train_fixed", "valid_fixed"],
        help="ts_total sub-split for these ids",
    )
    p.add_argument("--ts-dir", default=DEFAULT_TS_DIR, help="ts_total root dir")
    p.add_argument(
        "--out-dir", required=True, help="Output dir for *_mask_emb.nii.gz files"
    )
    p.add_argument(
        "--hf-id", default=DEFAULT_WAN_HF_ID, help="Wan VAE HuggingFace repo id"
    )
    p.add_argument("--in-plane", type=int, default=512, help="Resampled X=Y size (÷8)")
    p.add_argument(
        "--depth", type=int, default=253, help="Resampled Z (≡1 mod 4 for lossless)"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N volumes (smoke)"
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split ids into N interleaved shards (run N processes on one GPU to hide CPU/IO)",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This process's shard index in [0, num_shards)",
    )
    p.add_argument(
        "--device", default="cuda:0", help="Torch device (cuda:0 under CVD=1)"
    )
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p


def read_ids(ids_file: str) -> list[str]:
    """Load volume IDs from a `.json` ({"ids": [...]} or a bare list) or a `.txt` (one per line)."""
    path = Path(ids_file)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        ids = data["ids"] if isinstance(data, dict) else data
        return [str(i).strip() for i in ids if str(i).strip()]
    return [i.strip() for i in path.read_text().splitlines() if i.strip()]


def _git_sha() -> str:
    """Current repo HEAD sha for provenance, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_or_verify_meta(out_dir: Path, args, latent_shape_hwdc: list[int]) -> None:
    """Write a geometry+paint manifest into ``out_dir/meta.json`` — or abort if an existing one mismatches.

    Cache-misuse guard: prevents silently mixing mask latents precomputed at different geometries /
    VAEs / paint schemes into one dir.

    Args:
        out_dir: destination dir for the ``_mask_emb.nii.gz`` latents (gets ``meta.json``).
        args: parsed CLI args (reads ``hf_id`` / ``in_plane`` / ``depth``).
        latent_shape_hwdc: the produced latent shape ``(H, W, D, C)`` = ``(64, 64, 64, 16)``.
    """
    meta_path = out_dir / "meta.json"
    # group_id -> painted HU, for provenance / auditability.
    paint = {GROUP_NAMES[g]: HU_MIN + HU_STEP * g for g in sorted(GROUP_NAMES)}
    meta = {
        "kind": "wan_mask_latent",
        "hf_id": args.hf_id,
        "subfolder": "vae",
        "in_plane": args.in_plane,
        "depth": args.depth,
        "latent_shape_hwdc": latent_shape_hwdc,  # (H, W, D, C)
        "interpolation": "nearest",
        "grouping": "src.data.organ_groups.apply_grouping (117 TS -> 5: bg+4 organ)",
        "paint_hu": paint,
        "hu_clip": 1000,
        "normalization": "per-channel latents_mean/std",
        "spatial_compression": 8,
        "temporal_compression": 4,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    critical = (
        "kind",
        "hf_id",
        "in_plane",
        "depth",
        "latent_shape_hwdc",
        "interpolation",
        "grouping",
        "paint_hu",
        "normalization",
        "spatial_compression",
        "temporal_compression",
    )
    if meta_path.is_file():
        prev = json.loads(meta_path.read_text())
        diffs = {k: (prev.get(k), meta[k]) for k in critical if prev.get(k) != meta[k]}
        if diffs:
            raise SystemExit(
                f"meta.json mismatch in {out_dir}:\n"
                + "\n".join(
                    f"  {k}: existing {a!r} != requested {b!r}"
                    for k, (a, b) in diffs.items()
                )
                + "\nRefusing to write into a cache built differently — "
                "use a fresh --out-dir or delete the old cache."
            )
        return
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path}")


def main() -> None:
    args = build_parser().parse_args()

    ids = read_ids(args.ids_file)
    if args.limit:
        ids = ids[: args.limit]
    if (
        args.num_shards > 1
    ):  # interleaved shard for this worker (skip-existing makes it resumable)
        ids = ids[args.shard :: args.num_shards]

    ts_dir = Path(args.ts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Latent geometry implied by the resample grid: in-plane //8, depth 1+(D-1)//4, channels 16.
    lat_hw = args.in_plane // 8
    lat_d = 1 + (args.depth - 1) // 4
    if args.shard == 0:
        write_or_verify_meta(out_dir, args, [lat_hw, lat_hw, lat_d, 16])

    transforms = build_resample_transforms(args.in_plane, args.depth)
    print(f"Loading Wan VAE {args.hf_id} on {args.device} ...")
    vae = WanVAE(hf_id=args.hf_id, device=args.device)
    print("  VAE ready")

    skipped, processed, missing = 0, 0, 0
    for vol_id in tqdm(ids, desc="wan mask latents"):
        out_path = out_dir / f"{vol_id}_mask_emb.nii.gz"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        mask_path = ts_mask_path(vol_id, ts_dir, args.split)
        if not mask_path.is_file():
            print(f"WARNING: ts mask not found: {mask_path}", file=sys.stderr)
            missing += 1
            continue

        try:
            out = transforms({"mask": str(mask_path)})
            nda = out["mask"]
            affine = nda.meta["affine"].numpy() if hasattr(nda, "meta") else np.eye(4)
            m = (
                nda.numpy().squeeze()
                if hasattr(nda, "numpy")
                else np.asarray(nda).squeeze()
            )  # (X, Y, Z) label map, nearest-resampled 117-label
            grouped = apply_grouping(np.rint(m).astype(np.int16))  # -> {0..4} (X, Y, Z)
            hu = HU_MIN + HU_STEP * grouped.astype(np.float32)  # paint -> HU volume
            # (X, Y, Z) HU -> (T=Z, H=X, W=Y) as Wan expects (depth is the temporal axis)
            mask_thw = (
                torch.from_numpy(hu).permute(2, 0, 1).contiguous()
            )  # (253, 512, 512)
            z = vae.encode(mask_thw)  # (16, 64, 64, 64) = (C, D_lat, H_lat, W_lat)
            hwdc = rearrange(z.numpy(), "c d h w -> h w d c")  # (64, 64, 64, 16)
            nib.save(nib.Nifti1Image(np.float32(hwdc), affine=affine), str(out_path))
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR processing {vol_id}: {exc}", file=sys.stderr)

    print(f"\nDone. processed={processed}, skipped={skipped}, ts_missing={missing}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
