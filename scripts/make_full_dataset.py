#!/usr/bin/env python
"""Build the FULL CT-RATE dataset (train_fixed + valid_fixed) mirroring ctrate_toy_v2.

Companion to ``scripts/make_toy_dataset.py``. Where the toy build is a representative
*subsample* (5000 train) + a one-per-patient eval set, this build is the **full census**
of both official splits, with MAISI VAE latents symlinked from the precomputed embedding
store and a recorded backlog of scans whose latent does not exist yet.

Layout under ``--out-dir`` (default ``/workspace/data/ctrate_full``)::

    train/
      ids.json                # full train_fixed census (from train_metadata.csv)
      latents/<sid>_emb.nii.gz -> EMB_DIR symlink   (+ .gzmulti_2560.json sidecar)
      missing_latents.json    # census scans with no precomputed latent (backfill list)
      stats.json              # channel mean/std (copied; see provenance note)
    valid/
      ids.json                # full valid_fixed census (from validation_metadata.csv)
      latents/<sid>_emb.nii.gz -> EMB_DIR symlink
      missing_latents.json    # scans to encode later (GPU backfill)
    manifest.json
    README.md

Latent source is the fp32 ``*_emb.nii.gz`` store (``(120,120,64,4)``), NOT the small ``.pt``
mu/sigma set: only the emb store covers all ~47k train scans. valid_fixed latents are only
partially precomputed (image_embeddings has 1000 of 3038 valid scans) — the rest land in
``missing_latents.json`` for a later frozen-MAISI encode pass.

This script also backfills a ``latents/`` folder into the existing
``ctrate_toy_v2/valid_v2`` (it is used by reportgen / abnclass, not just ctgen) unless
``--no-valid-latents`` is passed.

Examples
--------
    python scripts/make_full_dataset.py                 # symlink, build full + valid_v2 latents
    python scripts/make_full_dataset.py --copy          # materialise instead of symlink
    python scripts/make_full_dataset.py --no-valid-latents
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_full_dataset")

# --- source paths (read-only) ---
DS = Path("/workspace/datasets/datasets/CT-RATE/dataset")
EMB_DIR = Path("/workspace/data/report2ct_work_dir/image_embeddings")
LAT_STATS = Path("/workspace/datasets/datasets/latents/stats.json")
TRAIN_META = DS / "metadata" / "train_metadata.csv"
VALID_META = DS / "metadata" / "validation_metadata.csv"
TOY_VALID_V2 = Path("/workspace/data/ctrate_toy_v2/valid_v2")

# Volumes that can never yield a clean latent — dropped from the census in addition to
# no_chest (brain) scans. Two reasons (both dataset-intrinsic and stable):
#   - NaN z-spacing (data_correction_note.md §3): needs a manual z-spacing before encode.
#   - missing raw NIfTI: no source volume on disk, so MAISI can never encode it.
UNENCODABLE: dict[str, set[str]] = {
    "train": {
        "train_1267_a_4",  # NaN z-spacing (note §3)
        "train_11755_a_3",  # NaN z-spacing (note §3)
        "train_11755_a_4",  # NaN z-spacing (note §3)
        "train_14384_a_2",  # missing raw NIfTI
    },
    "valid": {
        "valid_251_a_2",  # missing raw NIfTI
    },
}


# --------------------------------------------------------------------------- helpers
def _census_ids(meta_csv: Path) -> list[str]:
    """Full scan-id census for a split, from the metadata VolumeName column."""
    vn = pd.read_csv(meta_csv)["VolumeName"].astype(str)
    return sorted(v[:-7] if v.endswith(".nii.gz") else v for v in vn)


def _load_no_chest() -> dict[str, set[str]]:
    """Brain / non-chest volume stems per split, from metadata/no_chest_{train,valid}.txt.

    These are dropped from the census (data_correction_note.md §2). Mirrors the parser in
    apply_no_chest_correction.py:load_no_chest (basename + strip ``.nii.gz``); read directly
    from the read-only upstream txt so this builder needs no prior script run.
    """
    out: dict[str, set[str]] = {}
    for split in ("train", "valid"):
        stems: set[str] = set()
        for line in (
            (DS / "metadata" / f"no_chest_{split}.txt").read_text().splitlines()
        ):
            line = line.strip()
            if not line:
                continue
            stem = Path(line).name
            if stem.endswith(".nii.gz"):
                stem = stem[: -len(".nii.gz")]
            stems.add(stem)
        out[split] = stems
    return out


def _purge_excluded(latents_dir: Path, excluded: set[str]) -> int:
    """Unlink any existing latent symlink (+ sidecar) whose id is in ``excluded``.

    A prior build materialised brain symlinks; the filtered census no longer recreates them
    but won't remove them either. Mirrors apply_no_chest_correction.py:purge_brain_artifacts.
    """
    if not latents_dir.exists():
        return 0
    removed = 0
    for sid in excluded:
        for suffix in ("_emb.nii.gz", "_emb.nii.gzmulti_2560.json"):
            d = latents_dir / f"{sid}{suffix}"
            if d.is_symlink() or d.exists():
                d.unlink()
                removed += 1
    return removed


def _link_latents(ids: list[str], latents_dir: Path, copy: bool) -> list[str]:
    """Symlink/copy ``<sid>_emb.nii.gz`` (+ sidecar) for every id with a precomputed latent.

    Returns the sorted list of ids whose latent is MISSING (backfill backlog).
    """
    latents_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for sid in ids:
        src = EMB_DIR / f"{sid}_emb.nii.gz"
        if not src.exists():
            missing.append(sid)
            continue
        for suffix in ("_emb.nii.gz", "_emb.nii.gzmulti_2560.json"):
            s = EMB_DIR / f"{sid}{suffix}"
            if not s.exists():
                continue
            d = latents_dir / f"{sid}{suffix}"
            if d.is_symlink() or d.exists():
                d.unlink()
            if copy:
                shutil.copy2(s, d)
            else:
                d.symlink_to(s)
    return sorted(missing)


def _write_split(
    out: Path,
    name: str,
    census: list[str],
    excluded: set[str],
    n_excluded: int,
    n_unencodable: int,
    copy: bool,
) -> dict:
    sdir = out / name
    sdir.mkdir(parents=True, exist_ok=True)
    purged = _purge_excluded(sdir / "latents", excluded)
    missing = _link_latents(census, sdir / "latents", copy)
    have = len(census) - len(missing)
    (sdir / "ids.json").write_text(
        json.dumps(
            {
                "split": name,
                "source": "train_fixed" if name == "train" else "valid_fixed",
                "selection": "full census minus no_chest (brain) and unencodable "
                "(NaN z-spacing / missing raw NIfTI) scans",
                "n": len(census),
                "n_with_latent": have,
                "n_missing_latent": len(missing),
                "n_excluded_no_chest": n_excluded,
                "n_excluded_unencodable": n_unencodable,
                "ids": census,
            },
            indent=2,
        )
    )
    (sdir / "missing_latents.json").write_text(
        json.dumps(
            {
                "note": "scans in census with no precomputed *_emb.nii.gz; encode via "
                "frozen MAISI VAE later, then re-run this script to symlink them in.",
                "n": len(missing),
                "ids": missing,
            },
            indent=2,
        )
    )
    if LAT_STATS.exists():
        stats = json.loads(LAT_STATS.read_text())
        stats["_provenance"] = (
            "channel mean/std from datasets/latents/stats.json (computed on the 5000-scan "
            ".pt subset); same MAISI VAE latent space as the emb store, so applicable here."
        )
        (sdir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "%s: census=%d  with_latent=%d  missing=%d  excluded_no_chest=%d  "
        "excluded_unencodable=%d  purged=%d  latents %s -> %s",
        name,
        len(census),
        have,
        len(missing),
        n_excluded,
        n_unencodable,
        purged,
        "copied" if copy else "symlinked",
        sdir / "latents",
    )
    return {
        "n": len(census),
        "n_with_latent": have,
        "n_missing_latent": len(missing),
        "n_excluded_no_chest": n_excluded,
        "n_excluded_unencodable": n_unencodable,
    }


# --------------------------------------------------------------------------- valid_v2
def backfill_valid_v2_latents(copy: bool) -> dict:
    ids_json = TOY_VALID_V2 / "ids.json"
    if not ids_json.exists():
        log.warning("valid_v2 ids.json not found at %s; skipping", ids_json)
        return {}
    ids = json.loads(ids_json.read_text())["ids"]
    missing = _link_latents(ids, TOY_VALID_V2 / "latents", copy)
    have = len(ids) - len(missing)
    (TOY_VALID_V2 / "missing_latents.json").write_text(
        json.dumps(
            {
                "note": "valid_v2 scans (valid_fixed one-per-patient) lacking a precomputed "
                "latent; encode via frozen MAISI VAE later and re-run with --no-... to fill.",
                "n": len(missing),
                "ids": missing,
            },
            indent=2,
        )
    )
    log.info(
        "valid_v2: n=%d  with_latent=%d  missing=%d  latents -> %s",
        len(ids),
        have,
        len(missing),
        TOY_VALID_V2 / "latents",
    )
    return {"n": len(ids), "n_with_latent": have, "n_missing_latent": len(missing)}


# --------------------------------------------------------------------------- docs
def write_docs(out: Path, summary: dict, args) -> None:
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "latent_source": "emb (image_embeddings/*_emb.nii.gz, fp32 (120,120,64,4))",
                "materialise": "copy" if args.copy else "symlink",
                "splits": summary,
                "no_chest_exclusion": {
                    "source": "metadata/no_chest_{train,valid}.txt (data_correction_note.md §2)",
                    "rule": "drop brain / non-chest volumes from the census "
                    "(full multi-volume set, no per-patient substitution)",
                    "n_excluded": {
                        "train": summary["train"].get("n_excluded_no_chest", 0),
                        "valid": summary["valid"].get("n_excluded_no_chest", 0),
                    },
                },
                "unencodable_exclusion": {
                    "source": "UNENCODABLE in make_full_dataset.py",
                    "rule": "drop volumes that can never yield a clean latent: NaN "
                    "z-spacing (data_correction_note.md §3) or missing raw NIfTI",
                    "n_excluded": {
                        "train": summary["train"].get("n_excluded_unencodable", 0),
                        "valid": summary["valid"].get("n_excluded_unencodable", 0),
                    },
                },
                "sources": {
                    "train_census": str(TRAIN_META),
                    "valid_census": str(VALID_META),
                    "latents": str(EMB_DIR),
                },
                "note": "valid_fixed latents are partial (image_embeddings has 1000 of 3038 "
                "valid scans); missing ids recorded per split for a later GPU encode pass.",
            },
            indent=2,
        )
    )
    (out / "README.md").write_text(
        "# CT-RATE full dataset (train_fixed + valid_fixed)\n\n"
        "Built by `scripts/make_full_dataset.py`. Full **census** of both official splits "
        "(cf. `ctrate_toy_v2`, which is a 5000-scan representative subsample + eval set).\n\n"
        f"- **train/** — {summary['train']['n']} train_fixed scans; "
        f"{summary['train']['n_with_latent']} with latent symlinked, "
        f"{summary['train']['n_missing_latent']} missing.\n"
        f"- **valid/** — {summary['valid']['n']} valid_fixed scans; "
        f"{summary['valid']['n_with_latent']} with latent symlinked, "
        f"{summary['valid']['n_missing_latent']} missing (backfill via frozen MAISI encode).\n\n"
        "Latents are fp32 `*_emb.nii.gz` `(120,120,64,4)` symlinks (NOT the `.pt` mu/sigma set, "
        "which only covers the 5000-scan split). Each split's `missing_latents.json` lists the "
        "scans still to encode; re-run this script after encoding to symlink them in.\n\n"
        "`valid_v2` (under `ctrate_toy_v2/valid_v2`, the one-per-patient eval set used by "
        "ctgen / reportgen / abnclass) gets its own `latents/` folder from the same store.\n"
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/workspace/data/ctrate_full", type=Path)
    ap.add_argument(
        "--copy", action="store_true", help="materialise instead of symlink"
    )
    ap.add_argument(
        "--no-valid-latents",
        action="store_true",
        help="skip backfilling latents into ctrate_toy_v2/valid_v2",
    )
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    nc = _load_no_chest()  # brain / non-chest volumes to drop, per split

    def _split(name: str, meta: Path, key: str) -> dict:
        full = _census_ids(meta)
        full_set = set(full)
        drop = nc[key] | UNENCODABLE[key]  # brain + unencodable, both dropped + purged
        census = [i for i in full if i not in drop]
        n_nc = len(nc[key] & full_set)
        n_un = len(UNENCODABLE[key] & full_set)
        return _write_split(out, name, census, drop, n_nc, n_un, args.copy)

    summary = {
        "train": _split("train", TRAIN_META, "train"),
        "valid": _split("valid", VALID_META, "valid"),
    }
    if not args.no_valid_latents:
        summary["valid_v2"] = backfill_valid_v2_latents(args.copy)
    write_docs(out, summary, args)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
