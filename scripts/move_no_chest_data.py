#!/usr/bin/env python3
"""Physically move CT-RATE non-chest (brain) scans + their MAISI latents into quarantine dirs.

RUN ON THE HOST. Inside the dev container the CT-RATE tree is mounted read-only, so this
performs the *physical* move that `scripts/apply_no_chest_correction.py` could only emulate
via ids.json substitution. The two are complementary: run this on the host to relocate the
files, then the toy_v2 ids already point at chest substitutes.

The non-chest lists live at
    <root>/CT-RATE/dataset/metadata/no_chest_{train,valid}.txt
with lines like `train/train_10100/train_10100_a/train_10100_a_1.nii.gz` (751 train + 37 valid).
The leading `train/`|`valid/` is the *list* prefix; the on-disk folders are `train_fixed/`|
`valid_fixed/`.

Two independent moves:

1. Raw CT volumes (.nii.gz) — per-recon granularity, so only the listed reconstruction files
   move; other recons of the same patient stay put.
       <root>/CT-RATE/dataset/train_fixed/<sub>
         -> <root>/CT-RATE/dataset/no_chest_data/train_fixed/<sub>
       <root>/CT-RATE/dataset/valid_fixed/<sub>
         -> <root>/CT-RATE/dataset/no_chest_data/valid_fixed/<sub>

2. MAISI latents (one dir per scan: mu.pt/sigma.pt/src.txt). NOTE both <root>/latents/train
   and <root>/latents/valid hold `train_*_a_1` dirs — the latents "valid" split is a held-out
   subset of CT-RATE *train* scans, not CT-RATE valid. A latent dir is non-chest iff its name
   matches a non-chest scan id (the basename of any list line, minus .nii.gz). Both list files
   are unioned for matching; only `_a_1` ids can ever match a latent dir.
       <root>/latents/train/<id>  -> <root>/latents/no_chest_data/train/<id>
       <root>/latents/valid/<id>  -> <root>/latents/no_chest_data/valid/<id>

Default is a DRY RUN (prints the plan, moves nothing). Pass --apply to actually move.
Idempotent: a source already moved (missing source, present destination) is reported and skipped.

Usage (on the host):
    python move_no_chest_data.py --root /path/to/datasets/datasets            # preview
    python move_no_chest_data.py --root /path/to/datasets/datasets --apply    # move
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# list prefix -> on-disk `_fixed` dir, for the raw-CT move
FIXED = {"train": "train_fixed", "valid": "valid_fixed"}


def read_list(metadata_dir: Path, split: str) -> list[str]:
    """Read no_chest_<split>.txt -> list of POSIX subpaths (`train/.../*.nii.gz`)."""
    path = metadata_dir / f"no_chest_{split}.txt"
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def scan_id(rel_path: str) -> str:
    """`train/train_10100/.../train_10100_a_1.nii.gz` -> `train_10100_a_1`."""
    stem = os.path.basename(rel_path)
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    return stem


def move_one(src: Path, dst: Path, dry: bool, counts: dict[str, int]) -> None:
    """Move a single path (file or dir) src->dst, creating dst's parent; idempotent."""
    if dst.exists():
        # destination already there: treat as done (and drop a stale source if any)
        counts["already"] += 1
        if src.exists():
            print(f"  SKIP (dest exists): {src} -> {dst}")
        return
    if not src.exists():
        counts["missing"] += 1
        print(f"  MISSING (neither src nor dst): {src}")
        return
    counts["move"] += 1
    print(f"  MOVE: {src} -> {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def move_raw(root: Path, dry: bool) -> None:
    """Move per-recon .nii.gz volumes listed in no_chest_{train,valid}.txt."""
    dataset = root / "CT-RATE" / "dataset"
    metadata = dataset / "metadata"
    quarantine = dataset / "no_chest_data"
    print("\n=== Raw CT volumes (.nii.gz, per-recon) ===")
    for split, fixed in FIXED.items():
        rels = read_list(metadata, split)
        print(f"\n[{split}] {len(rels)} listed -> {quarantine / fixed}")
        counts = {"move": 0, "already": 0, "missing": 0}
        for rel in rels:
            # strip list prefix (`train/` | `valid/`), re-root under the `_fixed` dir
            sub = rel.split("/", 1)[1]  # drop leading `train/` or `valid/`
            src = dataset / fixed / sub
            dst = quarantine / fixed / sub
            move_one(src, dst, dry, counts)
        print(
            f"[{split}] moved={counts['move']} already={counts['already']} "
            f"missing={counts['missing']}"
        )


def move_latents(root: Path, dry: bool) -> None:
    """Move MAISI latent dirs whose name is a non-chest scan id (union of both lists)."""
    metadata = root / "CT-RATE" / "dataset" / "metadata"
    latents = root / "latents"
    quarantine = latents / "no_chest_data"
    nc_ids = {scan_id(r) for s in ("train", "valid") for r in read_list(metadata, s)}
    print(f"\n=== MAISI latents (dir per scan) ===\nnon-chest scan ids: {len(nc_ids)}")
    for split in ("train", "valid"):
        src_dir = latents / split
        dst_dir = quarantine / split
        if not src_dir.is_dir():
            print(f"\n[{split}] {src_dir} absent — skipping")
            continue
        present = sorted(d for d in os.listdir(src_dir) if d in nc_ids)
        print(f"\n[{split}] {len(present)} non-chest latent dir(s) -> {dst_dir}")
        counts = {"move": 0, "already": 0, "missing": 0}
        for name in present:
            move_one(src_dir / name, dst_dir / name, dry, counts)
        print(
            f"[{split}] moved={counts['move']} already={counts['already']} "
            f"missing={counts['missing']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/user/hyunjun/datasets/datasets"),
        help="datasets root holding CT-RATE/ and latents/ (default: %(default)s)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="actually move (default: dry run)"
    )
    ap.add_argument("--skip-raw", action="store_true", help="skip the .nii.gz move")
    ap.add_argument(
        "--skip-latents", action="store_true", help="skip the latent-dir move"
    )
    args = ap.parse_args()
    dry = not args.apply

    print(f"root={args.root}  mode={'DRY RUN' if dry else 'APPLY'}")
    if not (args.root / "CT-RATE" / "dataset" / "metadata").is_dir():
        raise SystemExit(
            f"metadata dir not found under {args.root} — pass the right --root"
        )

    if not args.skip_raw:
        move_raw(args.root, dry)
    if not args.skip_latents:
        move_latents(args.root, dry)

    if dry:
        print("\n(DRY RUN — nothing moved. Re-run with --apply to perform the moves.)")


if __name__ == "__main__":
    main()
