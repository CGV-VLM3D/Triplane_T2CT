#!/usr/bin/env python3
"""Precompute per-volume Wan2.1 VAE latents for report2ct_wan training.

Twin of `scripts/precompute_report2ct_image_embeddings.py` (MAISI) but on the Wan2.1 VAE
substrate. Each CT is resampled to a FIXED geometry so every latent shares one shape (required
for diffusion training), encoded through `src.baselines.wan_vae.WanVAE`, and saved as
`<id>_emb.nii.gz` in the report2ct HWDC convention — shape (H, W, D, C) = (64, 64, 64, 16) for
the default 512×512×253 input.

Geometry (see docs/wan_latent_runbook.md):
  CT (X, Y, Z) --Resized-> (512, 512, 253) --permute(2,0,1)-> (T=253, H=512, W=512)
     --WanVAE.encode-> latent (16, T_lat=64, 64, 64) --rearrange-> (64, 64, 64, 16) HWDC.
  T=253 (≡1 mod 4) makes the Wan causal ×4 temporal compression lossless (253→64→253) and keeps
  every latent spatial dim divisible by 8 (MAISI-UNet 3-downsample requirement). In-plane must be
  ÷64 (512//8=64, 64÷8 ✓; 480 would give latent 60, NOT ÷8, breaking the UNet). depth 253 ≈
  CT-RATE native z (~260 median) ⇒ minimal z-resample ⇒ best reconstruction fidelity, and matches
  MAISI's 256-slice z-density for a fair MAISI-vs-Wan comparison.

Intensity: WanVAE does the HU clip[-1000,1000]/1000→[-1,1] itself, so these transforms only
resample (LoadImaged→EnsureChannelFirstd→Orientationd(RAS)→Resized), NO ScaleIntensityRanged —
raw HU is handed to WanVAE. CT-RATE `_fixed` inputs already have HU baked in (never re-apply
slope/intercept, [[ctrate-fixed-hu-no-rescale]]).

Cache hygiene: a `meta.json` (geometry + VAE id + normalization + provenance) is written into
--out-dir on the first run and VERIFIED on later runs; a geometry mismatch aborts BEFORE writing
so latents built at one geometry can never be silently mixed with another. Per-file `_emb.nii.gz`
headers additionally carry the (varied, per-scan) resampled spacing.

Runs in the dedicated `wan` conda env (diffusers ≥ 0.34). GPU 1:
  CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/wan/bin/python \\
      scripts/precompute_wan_image_embeddings.py \\
      --ids-file /workspace/data/ctrate_toy_v2/train/ids.json \\
      --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \\
      --out-dir /workspace/data/report2ct_wan/latents_512x512x253 --device cuda:0 --limit 100

Run again with the valid_v2 ids-file + valid_fixed into the SAME --out-dir (train_*/valid_*
filenames distinguish them; the shared meta.json is re-verified). build_report2ct_datalist.py then
reads that single dir. --ids-file accepts a .json ({"ids": [...]} or a bare list) or a .txt.
Full run: omit --limit.
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


def id_to_nifti_path(vol_id: str, ct_rate_dir: Path) -> Path:
    """Derive the CT NIfTI path from a volume ID (same convention as the MAISI precompute).

    ``train_10000_a_1`` → ``<ct_rate_dir>/train_10000/train_10000_a/train_10000_a_1.nii.gz``.
    """
    parts = vol_id.split("_")
    patient = "_".join(parts[:2])
    series = "_".join(parts[:3])
    return ct_rate_dir / patient / series / f"{vol_id}.nii.gz"


def build_resample_transforms(in_plane: int, depth: int) -> Compose:
    """Resample-only MONAI transforms to a fixed (X=in_plane, Y=in_plane, Z=depth) grid.

    No intensity scaling — WanVAE consumes raw HU and does the [-1000,1000]/1000 itself.
    """
    return Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
            monai.transforms.Resized(
                keys="image",
                spatial_size=(in_plane, in_plane, depth),  # (X, Y, Z)
                mode="trilinear",
            ),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute Wan2.1 VAE latents for report2ct_wan"
    )
    p.add_argument("--ids-file", required=True, help="One volume ID per line")
    p.add_argument(
        "--ct-rate-dir", required=True, help="CT-RATE {train,valid}_fixed/ directory"
    )
    p.add_argument("--out-dir", required=True, help="Output dir for *_emb.nii.gz files")
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
    """Write a geometry manifest into ``out_dir/meta.json`` — or abort if an existing one mismatches.

    Cache-misuse guard: prevents silently mixing Wan latents precomputed at different geometries /
    VAEs into one dir. Geometry-critical fields (VAE id, in-plane, depth, latent shape,
    normalization, compression) are compared; a mismatch raises before any latent is written.

    Args:
        out_dir: destination dir for the ``_emb.nii.gz`` latents (gets ``meta.json``).
        args: parsed CLI args (reads ``hf_id`` / ``in_plane`` / ``depth``).
        latent_shape_hwdc: the produced latent shape ``(H, W, D, C)`` = ``(64, 64, 64, 16)``.
    """
    meta_path = out_dir / "meta.json"
    meta = {
        "hf_id": args.hf_id,
        "subfolder": "vae",
        "in_plane": args.in_plane,
        "depth": args.depth,
        "latent_shape_hwdc": latent_shape_hwdc,  # (H, W, D, C)
        "hu_clip": 1000,
        "intensity": "[-1, 1]",
        "normalization": "per-channel latents_mean/std",
        "spatial_compression": 8,
        "temporal_compression": 4,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    # Fields whose mismatch would corrupt a reused cache (provenance fields excluded).
    critical = (
        "hf_id",
        "in_plane",
        "depth",
        "latent_shape_hwdc",
        "normalization",
        "spatial_compression",
        "temporal_compression",
    )
    if meta_path.is_file():
        prev = json.loads(meta_path.read_text())
        diffs = {k: (prev.get(k), meta[k]) for k in critical if prev.get(k) != meta[k]}
        if diffs:
            raise SystemExit(
                f"meta.json geometry mismatch in {out_dir}:\n"
                + "\n".join(
                    f"  {k}: existing {a!r} != requested {b!r}"
                    for k, (a, b) in diffs.items()
                )
                + "\nRefusing to write into a cache built at a different geometry — "
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

    ct_rate_dir = Path(args.ct_rate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Latent geometry implied by the resample grid: in-plane //8 (Wan 8× spatial),
    # depth 1+(D-1)//4 (Wan causal 4× temporal); channels 16.  (64, 64, 64, 16) for 512×512×253.
    # Only shard 0 writes/verifies meta.json — avoids a write race between concurrent workers.
    lat_hw = args.in_plane // 8
    lat_d = 1 + (args.depth - 1) // 4
    if args.shard == 0:
        write_or_verify_meta(out_dir, args, [lat_hw, lat_hw, lat_d, 16])

    transforms = build_resample_transforms(args.in_plane, args.depth)
    print(f"Loading Wan VAE {args.hf_id} on {args.device} ...")
    vae = WanVAE(hf_id=args.hf_id, device=args.device)
    print("  VAE ready")

    skipped, processed, missing = 0, 0, 0
    for vol_id in tqdm(ids, desc="wan latents"):
        out_path = out_dir / f"{vol_id}_emb.nii.gz"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        nifti_path = id_to_nifti_path(vol_id, ct_rate_dir)
        if not nifti_path.is_file():
            print(f"WARNING: NIfTI not found: {nifti_path}", file=sys.stderr)
            missing += 1
            continue

        try:
            out = transforms({"image": str(nifti_path)})
            nda = out["image"]
            affine = nda.meta["affine"].numpy() if hasattr(nda, "meta") else np.eye(4)
            x = (
                nda.numpy().squeeze()
                if hasattr(nda, "numpy")
                else np.asarray(nda).squeeze()
            )
            # (X, Y, Z) HU -> (T=Z, H=X, W=Y) as Wan expects (depth is the temporal axis)
            ct_thw = (
                torch.from_numpy(x).permute(2, 0, 1).contiguous()
            )  # (253, 512, 512)
            z = vae.encode(ct_thw)  # (16, 64, 64, 64) = (C, D_lat, H_lat, W_lat)
            # -> report2ct HWDC layout (H_lat, W_lat, D_lat, C) for EnsureChannelFirstd on load
            hwdc = rearrange(z.numpy(), "c d h w -> h w d c")  # (64, 64, 64, 16)
            nib.save(nib.Nifti1Image(np.float32(hwdc), affine=affine), str(out_path))
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR processing {vol_id}: {exc}", file=sys.stderr)

    print(f"\nDone. processed={processed}, skipped={skipped}, nifti_missing={missing}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
