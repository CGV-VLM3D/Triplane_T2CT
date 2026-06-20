#!/usr/bin/env python3
"""Apply the CT-RATE non-chest (brain) scan correction to ctrate_toy_v2.

Background. `data_correction_note.md` §2 lists brain series that are not chest CTs
(`metadata/no_chest_{train,valid}.txt`: 752 train + 37 valid volumes). The CT-RATE source
tree under `/workspace/datasets` is mounted **read-only** (and is a collaborator's area), so
the volumes cannot be physically moved. Instead we:

  1. Write a canonical exclusion manifest `data/no_chest_quarantine/no_chest_ids.json`
     (the durable record + reusable filter list).
  2. Substitute every brain id inside `ctrate_toy_v2/{train,valid_v2}/ids.json` with the
     same patient's lowest-index chest volume (keeps train=5000, valid_v2=1304
     one-per-patient; verified all affected patients have a chest alternative).
  3. Sync the `latents/` symlink dirs to the corrected id lists (drop brain symlinks; add
     substitute symlinks where the source latent already exists).
  4. Record provenance in `ctrate_toy_v2/manifest.json`.

`ids.json` is the single source of truth consumed by `load_eval_cases`, the fvlm precompute
and the datalist builders, so step 2 is the authoritative fix; the `latents/` dirs are
secondary materializations (partial backfill is the pre-existing state).

Idempotent: re-running after correction is a no-op (no brain ids remain to substitute).

Usage:
  python scripts/apply_no_chest_correction.py            # apply
  python scripts/apply_no_chest_correction.py --dry-run  # preview, no writes
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

DATASET = Path("/workspace/datasets/datasets/CT-RATE/dataset")
TOY = Path("/workspace/data/ctrate_toy_v2")
EMB_DIR = Path("/workspace/data/report2ct_work_dir/image_embeddings")
PT_TRAIN = Path("/workspace/datasets/datasets/latents/train")
MANIFEST_DIR = Path("/workspace/data/no_chest_quarantine")

# (toy split dir, no_chest list stem, {split}_fixed dir, latents naming)
SPLITS = {
    "train": {"toy": "train", "fixed": "train_fixed", "list": "train"},
    "valid_v2": {"toy": "valid_v2", "fixed": "valid_fixed", "list": "valid"},
}


def load_no_chest(list_split: str) -> list[str]:
    """Read a no_chest list → volume stems (`train_109_a_1`), order preserved."""
    out: list[str] = []
    p = DATASET / "metadata" / f"no_chest_{list_split}.txt"
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stem = os.path.basename(line)
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        out.append(stem)
    return out


def chest_substitute(stem: str, fixed: str, brain: set[str]) -> str | None:
    """Lowest-index chest volume for `stem`'s patient (series letter, then volume idx)."""
    patient = "_".join(stem.split("_")[:2])
    vols = glob.glob(str(DATASET / fixed / patient / "*" / "*.nii.gz"))
    stems = [os.path.basename(v)[: -len(".nii.gz")] for v in vols]
    chest = [s for s in stems if s not in brain]

    def key(s: str) -> tuple[str, int]:
        parts = s.split("_")
        return (parts[2], int(parts[3]))

    return sorted(chest, key=key)[0] if chest else None


def write_exclusion_manifest(brain: dict[str, list[str]], dry_run: bool) -> None:
    payload = {
        "source": "CT-RATE data_correction_note.md §2 (brain / non-chest scans)",
        "note": "CT source tree is read-only; this manifest is the quarantine record.",
        "n_train": len(brain["train"]),
        "n_valid": len(brain["valid"]),
        "train": brain["train"],
        "valid": brain["valid"],
    }
    print(
        f"exclusion manifest: train={len(brain['train'])} valid={len(brain['valid'])}"
    )
    if not dry_run:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        (MANIFEST_DIR / "no_chest_ids.json").write_text(json.dumps(payload, indent=2))


def emb_exists(sub: str) -> bool:
    """Does the substitute's `_emb.nii.gz` latent exist — the latent both report2ct training
    and ctgen eval actually consume (resolved via ids.json, not the toy `latents/` dir)."""
    return (EMB_DIR / f"{sub}_emb.nii.gz").exists()


def sync_symlink(toy_split: str, old: str, sub: str, dry_run: bool) -> None:
    """Mechanically sync the toy `latents/` dir to its native format: drop the brain id's
    symlink, add the substitute's where its source materialisation exists.

    valid_v2/latents is `_emb.nii.gz`-style (→ image_embeddings); train/latents is the
    vestigial `mu.pt` dir-style (→ read-only datasets/latents/train, no substitute present).
    Coverage for the real consumer is reported separately via `emb_exists`.
    """
    if dry_run:
        return
    lat = TOY / toy_split / "latents"
    if toy_split == "train":
        # vestigial mu.pt dir-style; substitute has no pt materialisation → just drop brain
        old_link = lat / old
        if old_link.is_symlink() or old_link.exists():
            old_link.unlink()
        return
    # valid_v2: `_emb.nii.gz` symlink + its `_emb.nii.gzmulti_2560.json` context twin
    for suffix in ("_emb.nii.gz", "_emb.nii.gzmulti_2560.json"):
        old_link = lat / f"{old}{suffix}"
        new_link = lat / f"{sub}{suffix}"
        target = EMB_DIR / f"{sub}{suffix}"
        if old_link.is_symlink() or old_link.exists():
            old_link.unlink()
        if target.exists() and not (new_link.is_symlink() or new_link.exists()):
            new_link.symlink_to(target)


def purge_brain_artifacts(toy_split: str, brain: set[str], dry_run: bool) -> int:
    """Remove any latents-dir entry whose id-prefix is a brain id, regardless of ids.json.

    Catches leftovers from earlier backfills (e.g. brain `_a_1` context-JSON symlinks) that
    the id-driven substitution loop no longer iterates over once ids.json is clean.
    """
    lat = TOY / toy_split / "latents"
    removed = 0
    for entry in os.listdir(lat):
        if entry == ".omc":
            continue
        idtok = entry
        for suffix in ("_emb.nii.gzmulti_2560.json", "_emb.nii.gz"):
            if idtok.endswith(suffix):
                idtok = idtok[: -len(suffix)]
                break
        if idtok in brain:
            removed += 1
            if not dry_run:
                (lat / entry).unlink()
    if removed:
        print(
            f"{toy_split}: purged {removed} leftover brain artifact(s)"
            + ("  (DRY-RUN)" if dry_run else "")
        )
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    brain = {"train": load_no_chest("train"), "valid": load_no_chest("valid")}
    brain_set = {k: set(v) for k, v in brain.items()}
    write_exclusion_manifest(brain, dry)
    purge_brain_artifacts("train", brain_set["train"], dry)
    purge_brain_artifacts("valid_v2", brain_set["valid"], dry)

    n_sub: dict[str, int] = {}
    missing_latent: dict[str, list[str]] = {}
    for toy_split, cfg in SPLITS.items():
        ids_path = TOY / toy_split / "ids.json"
        doc = json.loads(ids_path.read_text())
        ids = doc["ids"]
        brn = brain_set[cfg["list"]]
        new_ids: list[str] = []
        for sid in ids:
            if sid in brn:
                sub = chest_substitute(sid, cfg["fixed"], brn)
                if sub is None:
                    raise SystemExit(f"no chest substitute for {sid}")
                new_ids.append(sub)
                sync_symlink(toy_split, sid, sub, dry)
            else:
                new_ids.append(sid)
        assert len(new_ids) == len(ids), "count changed"
        assert len(set(new_ids)) == len(new_ids), "duplicate id introduced"
        assert not (set(new_ids) & brn), "brain id still present after substitution"

        # Deterministic (run-order-independent) stats: a patient was substituted iff its
        # would-be `_a_1` pick is a brain id; pending = substitutes lacking an _emb latent.
        def patient(s: str) -> str:
            return "_".join(s.split("_")[:2])

        substitutes = [s for s in new_ids if f"{patient(s)}_a_1" in brn]
        missing = sorted(s for s in substitutes if not emb_exists(s))
        n_sub[toy_split] = len(substitutes)
        missing_latent[toy_split] = missing

        doc["ids"] = new_ids
        doc["no_chest_correction"] = {
            "n_substituted": len(substitutes),
            "rule": "brain _a_1 -> same-patient lowest-index chest volume",
            "source": "data_correction_note.md §2",
        }
        print(
            f"{toy_split}: ids={len(new_ids)} substituted={len(substitutes)} "
            f"latents_missing={len(missing)}" + ("  (DRY-RUN)" if dry else "")
        )
        if not dry:
            ids_path.write_text(json.dumps(doc, indent=2))

    # provenance in manifest.json
    man_path = TOY / "manifest.json"
    if man_path.exists() and not dry:
        man = json.loads(man_path.read_text())
        man["no_chest_correction"] = {
            "source": "data_correction_note.md §2",
            "applied": "substitute brain _a_1 -> same-patient lowest-index chest volume",
            "n_substituted": n_sub,
            "exclusion_manifest": str(MANIFEST_DIR / "no_chest_ids.json"),
            "latents_pending_backfill": missing_latent,
        }
        man_path.write_text(json.dumps(man, indent=2))

    if any(missing_latent.values()):
        print("\nLatents still to backfill (substitutes lacking a source latent):")
        for k, v in missing_latent.items():
            if v:
                print(f"  {k}: {len(v)} -> {v}")


if __name__ == "__main__":
    main()
