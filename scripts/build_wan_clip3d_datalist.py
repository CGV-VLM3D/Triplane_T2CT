#!/usr/bin/env python3
"""Build the training datalist for report2ct_wan (Wan2.1 latent + CLIP3D-768 cond).

Twin of `scripts/build_fvlm_cond_datalist.py`, but pairs the **Wan2.1** image latent with the
**FrozenCLIP3D (1,768)** condition (the exact text2ct-twin conditioner), and stamps a **fixed**
voxel spacing `[0.75, 0.75, 3.0]` — text2ct's native output spacing — instead of reading a
per-scan header value. Rationale: report2ct_wan is the matched twin of report2ct_text2ct
(text2ct trains on a single fixed spacing), so the only variable vs that run is the VAE.

    {
      "image":   "<data/wan_clip3d/image_embeddings/<split>/<vol_id>_emb.nii.gz",  # (64,64,32,16)
      "spacing": [0.75, 0.75, 3.0],                       # fixed (text2ct native)
      "context": "<data/text2ct_clip3d/cond_embeddings/<split>/<scan_id>.pt"        # (1, 768)
    }

image keyed by *volume* id (train_10000_a_1); context keyed by *scan* id (train_10000_a) since
the report — hence the CLIP3D embedding — is shared across a scan's reconstructions.

Usage (toy_v2 5k train + valid_v2 1304):
  python scripts/build_wan_clip3d_datalist.py \\
      --ids-train /workspace/data/ctrate_toy_v2/train/ids.json \\
      --image-dir /workspace/data/wan_clip3d/image_embeddings/train \\
      --cond-dir  /workspace/data/text2ct_clip3d/cond_embeddings/train \\
      --ids-valid /workspace/data/ctrate_toy_v2/valid_v2/ids.json \\
      --image-dir-valid /workspace/data/wan_clip3d/image_embeddings/valid \\
      --cond-dir-valid  /workspace/data/text2ct_clip3d/cond_embeddings/valid \\
      --out /workspace/data/wan_clip3d/datalist_wan_clip3d_5k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.fvlm_organ_report import scan_id_from_volume_name  # noqa: E402

# text2ct native output spacing (config_diff_model.json diffusion_unet_inference.spacing).
FIXED_SPACING = [0.75, 0.75, 3.0]


def _read_vol_ids(ids_file: Path) -> list[str]:
    """Volume ids from a `.json` ({"ids":[...]} or bare list) or `.txt` (one per line)."""
    text = ids_file.read_text()
    if ids_file.suffix == ".json":
        raw = json.loads(text)
        return raw["ids"] if isinstance(raw, dict) else raw
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def build_split(
    vol_ids: list[str], image_dir: Path, cond_dir: Path
) -> tuple[list[dict], int, int]:
    """One split → (entries, missing_image, missing_context). Skips a scan missing either file."""
    entries: list[dict] = []
    missing_img = missing_ctx = 0
    for vid in tqdm(vol_ids, leave=False):
        img_path = image_dir / f"{vid}_emb.nii.gz"
        if not img_path.is_file():
            missing_img += 1
            continue
        scan_id = scan_id_from_volume_name(vid)  # train_10000_a_1 -> train_10000_a
        ctx_path = cond_dir / f"{scan_id}.pt"
        if not ctx_path.is_file():
            missing_ctx += 1
            continue
        entries.append(
            {
                "image": str(img_path.resolve()),
                "spacing": list(FIXED_SPACING),  # fixed (text2ct twin), not per-scan
                "context": str(ctx_path.resolve()),
            }
        )
    return entries, missing_img, missing_ctx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build report2ct_wan (Wan latent + CLIP3D) datalist"
    )
    p.add_argument("--ids-train", required=True, help="train ids (.json 'ids' or .txt)")
    p.add_argument(
        "--image-dir", required=True, help="dir of train <vol_id>_emb.nii.gz (Wan)"
    )
    p.add_argument("--cond-dir", required=True, help="dir of train CLIP3D <scan_id>.pt")
    p.add_argument("--out", required=True, help="output datalist.json path")
    p.add_argument("--ids-valid", default=None, help="optional valid ids (valid_v2)")
    p.add_argument(
        "--image-dir-valid", default=None, help="dir of valid Wan _emb.nii.gz"
    )
    p.add_argument(
        "--cond-dir-valid", default=None, help="dir of valid CLIP3D <scan_id>.pt"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    train_ids = _read_vol_ids(Path(args.ids_train))
    print(f"Building training split ({len(train_ids)} vol ids)...")
    train_entries, mi, mc = build_split(
        train_ids, Path(args.image_dir), Path(args.cond_dir)
    )
    print(
        f"  → {len(train_entries)} included, {mi} missing image, {mc} missing cond .pt"
    )

    val_entries: list[dict] = []
    if args.ids_valid and args.image_dir_valid and args.cond_dir_valid:
        val_ids = _read_vol_ids(Path(args.ids_valid))
        print(f"Building validation split ({len(val_ids)} vol ids)...")
        val_entries, vmi, vmc = build_split(
            val_ids, Path(args.image_dir_valid), Path(args.cond_dir_valid)
        )
        print(
            f"  → {len(val_entries)} included, {vmi} missing image, {vmc} missing cond .pt"
        )

    if not train_entries:
        print(
            "ERROR: no training entries — run precompute_wan_image_embeddings.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"training": train_entries, "validation": val_entries}, indent=2)
    )
    print(f"\nWrote {out_path} ({len(train_entries)} train, {len(val_entries)} val)")


if __name__ == "__main__":
    main()
