#!/usr/bin/env python3
"""Merge image + text embeddings into per-sample JSON and build the training datalist.

Expected outputs:
  - <image_dir>/<id>_emb.nii.gzmulti_2560.json  — merged JSON with keys:
      dim, spacing, findings_embeddings, impression_embeddings
  - <out>  — datalist.json with training / validation lists

Usage:
  python scripts/build_report2ct_datalist.py \\
      --image-dir /workspace/data/report2ct_work_dir/image_embeddings \\
      --text-dir  /workspace/data/report2ct_work_dir/text_embeddings \\
      --ids-train /workspace/data/report2ct_work_dir/ids_train.txt \\
      --ids-valid /workspace/data/report2ct_work_dir/ids_valid.txt \\
      --out /workspace/data/report2ct_work_dir/datalist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Report2CT datalist JSON")
    p.add_argument(
        "--image-dir", required=True, help="Directory containing *_emb.nii.gz files"
    )
    p.add_argument(
        "--text-dir", required=True, help="Directory containing *multi_2560.json files"
    )
    p.add_argument(
        "--ids-train", required=True, help="ids_train.txt — one volume ID per line"
    )
    p.add_argument(
        "--ids-valid", required=True, help="ids_valid.txt — one volume ID per line"
    )
    p.add_argument("--out", required=True, help="Output datalist.json path")
    p.add_argument(
        "--mask-dir",
        default=None,
        help="Optional dir of <id>_mask_emb.nii.gz Wan mask latents; when set, each entry gains a "
        "'mask_latent' path and volumes missing the mask latent are skipped (report2ct_wan_mask).",
    )
    return p


def merge_sample(
    vol_id: str, image_dir: Path, text_dir: Path, mask_dir: Path | None = None
) -> dict | None:
    """Merge image embedding header (dim, spacing) with text embeddings into one JSON.

    Args:
        mask_dir: if given, also require + attach the ``<id>_mask_emb.nii.gz`` Wan mask latent as
            a ``"mask_latent"`` entry key (report2ct_wan_mask); a missing mask latent skips the vol.

    Returns the datalist entry dict, or None if any required file is missing.
    """
    img_path = image_dir / f"{vol_id}_emb.nii.gz"
    text_path = text_dir / f"{vol_id}_emb.nii.gzmulti_2560.json"
    merged_json_path = image_dir / f"{vol_id}_emb.nii.gzmulti_2560.json"

    if not img_path.is_file():
        return None
    if not text_path.is_file():
        return None
    mask_path = mask_dir / f"{vol_id}_mask_emb.nii.gz" if mask_dir else None
    if mask_path is not None and not mask_path.is_file():
        return None

    # Load spacing from image embedding NIfTI header
    img = nib.load(str(img_path))
    dim = list(img.shape[:3])
    spacing = [float(s) for s in img.header.get_zooms()[:3]]

    # Load text embeddings
    text_data = json.loads(text_path.read_text())

    merged = {
        "dim": dim,
        "spacing": spacing,
        "findings_embeddings": text_data["findings_embeddings"],
        "impression_embeddings": text_data["impression_embeddings"],
    }
    merged_json_path.write_text(json.dumps(merged))

    abs_img = str(img_path.resolve())
    abs_json = str(merged_json_path.resolve())
    entry = {
        "image": abs_img,
        "spacing": abs_json,
        "context_f": abs_json,
        "context_i": abs_json,
    }
    if mask_path is not None:
        entry["mask_latent"] = str(mask_path.resolve())
    return entry


def build_split(
    ids: list[str], image_dir: Path, text_dir: Path, mask_dir: Path | None = None
) -> tuple[list[dict], int]:
    entries = []
    missing = 0
    for vol_id in tqdm(ids, leave=False):
        entry = merge_sample(vol_id, image_dir, text_dir, mask_dir)
        if entry is None:
            missing += 1
        else:
            entries.append(entry)
    return entries, missing


def main() -> None:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir)
    text_dir = Path(args.text_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    def _read_ids(path: str) -> list[str]:
        p = Path(path)
        if p.suffix == ".json":  # toy_v2 ids are JSON ({"ids": [...]} or a bare list)
            data = json.loads(p.read_text())
            ids = data["ids"] if isinstance(data, dict) else data
            return [str(i).strip() for i in ids if str(i).strip()]
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

    ids_train = _read_ids(args.ids_train)
    ids_valid = _read_ids(args.ids_valid)

    if mask_dir is not None:
        print(f"Mask-latent dir set → each entry gains 'mask_latent' ({mask_dir})")

    print(f"Building training split ({len(ids_train)} IDs)...")
    train_entries, train_missing = build_split(ids_train, image_dir, text_dir, mask_dir)
    print(
        f"  → {len(train_entries)} included, {train_missing} skipped "
        "(missing image/text embedding" + (" or mask latent)" if mask_dir else ")")
    )

    print(f"Building validation split ({len(ids_valid)} IDs)...")
    val_entries, val_missing = build_split(ids_valid, image_dir, text_dir, mask_dir)
    print(f"  → {len(val_entries)} included, {val_missing} skipped")

    if not train_entries:
        print(
            "ERROR: no training entries — run precompute scripts first", file=sys.stderr
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
