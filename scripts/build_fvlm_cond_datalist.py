#!/usr/bin/env python3
"""Build the training datalist for the fVLM-conditioned ctgen model.

Pairs, per scan, the **exact** image latent Report2CT trains on with the frozen
fVLM per-organ condition embedding:

    {
      "image":   "<report2ct_work_dir>/image_embeddings/<vol_id>_emb.nii.gz",
      "spacing": [sx, sy, sz],         # per-scan, read from the _emb.nii.gz header
      "context": "<data/fVLM/cond_embeddings/<split>>/<scan_id>.pt"   # (4, 256)
    }

Why the report2ct `_emb.nii.gz` (not the ctrate_toy_v2 `mu.pt`): the two are
different MAISI encodings (toy `mu.pt` symlinks a collaborator latent set built
with a *different* preprocessing pipeline). To keep this a clean "swap only the
conditioner" comparison against Report2CT, we reuse Report2CT's own image latents
**and** their per-scan spacing. Mirrors `scripts/build_report2ct_datalist.py`,
except the per-sample condition is the fVLM `.pt` instead of the 2560-d text JSON.

The image is keyed by *volume* id (e.g. 'train_10000_a_1'); the condition is keyed
by *scan-study* id (e.g. 'train_10000_a') since the report — and thus the fVLM
embedding — is shared across a scan's reconstructions.

Usage (toy-train 5000):
  python scripts/build_fvlm_cond_datalist.py \\
      --ids-train /workspace/data/ctrate_toy_v2/train/ids.json \\
      --cond-dir  /workspace/data/fVLM/cond_embeddings/train \\
      --out       /workspace/data/fVLM/datalist_fvlm_cond_5k.json

  # optional held-out validation (needs valid _emb.nii.gz + its cond embeddings):
  #   --ids-valid /workspace/data/ctrate_toy_v2/valid_v2/ids.json \\
  #   --cond-dir-valid /workspace/data/fVLM/cond_embeddings/valid_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
from tqdm import tqdm

from src.data.fvlm_organ_report import scan_id_from_volume_name

_DEFAULT_IMAGE_DIR = "/workspace/data/report2ct_work_dir/image_embeddings"


def _read_vol_ids(ids_file: Path) -> list[str]:
    """Volume ids (e.g. 'train_10000_a_1') from a `.json` ({"ids": [...]} or a bare
    list) or `.txt` (one per line). No scan-id collapse — image paths need vol ids."""
    text = ids_file.read_text()
    if ids_file.suffix == ".json":
        raw = json.loads(text)
        return raw["ids"] if isinstance(raw, dict) else raw
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def build_split(
    vol_ids: list[str], image_dir: Path, cond_dir: Path
) -> tuple[list[dict], int, int]:
    """One split → (entries, missing_image, missing_context).

    Skips a scan if either its image latent or its fVLM condition .pt is absent.
    """
    entries: list[dict] = []
    missing_img = 0
    missing_ctx = 0
    for vid in tqdm(vol_ids, leave=False):
        img_path = image_dir / f"{vid}_emb.nii.gz"
        if not img_path.is_file():
            missing_img += 1
            continue
        scan_id = scan_id_from_volume_name(vid)  # 'train_10000_a_1' -> 'train_10000_a'
        ctx_path = cond_dir / f"{scan_id}.pt"
        if not ctx_path.is_file():
            missing_ctx += 1
            continue
        # Per-scan spacing from the latent NIfTI header (build_report2ct_datalist.py:62-64).
        spacing = [float(s) for s in nib.load(str(img_path)).header.get_zooms()[:3]]
        entries.append(
            {
                "image": str(img_path.resolve()),
                "spacing": spacing, # NIfTI 헤더의 spacing 값인 것 같다. 
                "context": str(ctx_path.resolve()),
            }
        )
    return entries, missing_img, missing_ctx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build fVLM-cond ctgen datalist JSON")
    p.add_argument(
        "--ids-train", required=True, help="train ids (.json with 'ids' or .txt)"
    )
    p.add_argument(
        "--cond-dir", required=True, help="dir of train fVLM cond <scan_id>.pt"
    )
    p.add_argument("--out", required=True, help="output datalist.json path")
    p.add_argument(
        "--image-dir",
        default=_DEFAULT_IMAGE_DIR,
        help="dir of <vol_id>_emb.nii.gz (default: report2ct_work_dir/image_embeddings)",
    )
    p.add_argument("--ids-valid", default=None, help="optional held-out valid ids")
    p.add_argument(
        "--cond-dir-valid", default=None, help="dir of valid fVLM cond <scan_id>.pt"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir)
    cond_dir = Path(args.cond_dir)

    train_ids = _read_vol_ids(Path(args.ids_train))
    print(f"Building training split ({len(train_ids)} vol ids)...")
    train_entries, mi, mc = build_split(train_ids, image_dir, cond_dir)
    print(
        f"  → {len(train_entries)} included, {mi} missing image, {mc} missing cond .pt"
    )

    val_entries: list[dict] = []
    if args.ids_valid and args.cond_dir_valid:
        val_ids = _read_vol_ids(Path(args.ids_valid))
        print(f"Building validation split ({len(val_ids)} vol ids)...")
        val_entries, vmi, vmc = build_split(
            val_ids, image_dir, Path(args.cond_dir_valid)
        )
        print(
            f"  → {len(val_entries)} included, {vmi} missing image, {vmc} missing cond .pt"
        )

    if not train_entries:
        print(
            "ERROR: no training entries — run precompute_fvlm_cond_embeddings.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    datalist = {"training": train_entries, "validation": val_entries}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(datalist, indent=2))
    print(
        f"\nWrote {out_path} ({len(train_entries)} train, {len(val_entries)} val entries)"
    )


if __name__ == "__main__":
    main()
