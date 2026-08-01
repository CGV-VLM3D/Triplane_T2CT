#!/usr/bin/env python3
"""Build a mask-intervention JSONL manifest (requirement F, generation side).

One row per ``(target, condition)`` pair, in the schema of
[docs/mask_intervention_manifest.md](../docs/mask_intervention_manifest.md). Targets come from
``load_eval_cases(n)`` (the frozen valid_v2 set); donors come from the clean CT-RATE censuses —
valid 3001 AND train 46,393 (``--donor-ids``).

Donor rules, enforced here rather than trusted:
  * ``gt``                    — ``cond_mask_source_id = target_id`` (no donor).
  * ``label_matched_swap``    — a cross-patient donor with EXACTLY the same 18-label vector (same
                                abnormalities, including "none at all" for a normal target).
                                No near-match fallback: a target with no exact twin gets NO
                                matched row at all, and is listed in the
                                ``.uncovered_targets.json`` sidecar. Among twins, one whose mask
                                latent already exists is preferred so the arm needs as little
                                extra precompute as possible.
  * ``label_mismatched_swap`` — a uniformly random cross-patient donor drawn from the scans that
                                ALREADY have a mask latent (this arm has no scarcity problem, so
                                it costs no precompute), excluding same-vector donors (~3 % of
                                uniform draws are same-vector, measured on this census — they
                                would otherwise turn part of the "mismatched" arm into a second
                                matched arm). No overlap cap: the realized overlap distribution
                                is reported instead.
  * ``null``                  — no mask at all; allowed ONLY for a checkpoint that actually
                                carries the learned ``no_mask_embed`` (report2ct_wan_mask_v2).

Why train is in the donor pool: only 770 distinct label combinations occur among the 3001 valid
scans, and a high-burden combination is usually unique — searching valid alone finds an exact twin
for just 193 of the 300 evaluated targets (measured 2026-08-02); adding train raises it to 249.
The remaining 51 have no twin anywhere in CT-RATE's 49,394 scans.

Self-pairing and patient leakage are impossible by construction: donor candidates are filtered by
``patient_id`` (NOT ``scan_id``) — this census holds 3001 scans from only 1304 patients
(2.30 scans/patient), so a scan-level check would let a different scan of the SAME patient
through. Both are re-verified over the written rows before the script exits.

Determinism: each row's donor is drawn from ``random.Random(f"{seed}|{condition}|{target_id}")``,
so the same ``--seed`` reproduces a byte-identical JSONL, and adding a condition never perturbs
another condition's donors. Build provenance (timestamp, git sha, argv) goes into the sidecar
``<out>.meta.json`` — keeping it OUT of the JSONL is what makes byte-identity checkable.

⚠ SCOPE — a manifest run is for DIAGNOSTIC metrics only (CLIPScore-T2I, ``dice_to_input_mask``,
``dice_to_gt_mask``). Do NOT score it for FID/FVD as leaderboard metrics: one target appears in
several rows, which breaks the "1 volume = 1 independent sample" premise those set-level metrics
rest on (the GT reference would also be counted once per repeat of the same target).

Example (headline model, 50 targets × 4 conditions, s_m=1.0):
  python scripts/build_mask_intervention_manifest.py \\
      --n 50 --seed 0 --conditions gt label_matched_swap label_mismatched_swap null \\
      --out /workspace/data/mask_intervention/manifest_n50_sm1.0_seed0.jsonl \\
      --ckpt /workspace/outputs/report2ct_wan_mask_v2/2026-07-26_2/checkpoints/epoch_299.ckpt \\
      --run-id wanmaskv2_ep299 --cfg-text 5.0 --cfg-mask 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.analysis import persample as persample_mod  # noqa: E402
from src.eval.analysis.labels import (  # noqa: E402
    LABEL_NAMES,
    load_label_csv,
    patient_id_of,
)
from src.eval.ct_rate_cases import load_eval_cases  # noqa: E402

CONDITIONS: tuple[str, ...] = (
    "gt",
    "label_matched_swap",
    "label_mismatched_swap",
    "null",
)
SWAP_CONDITIONS: tuple[str, ...] = ("label_matched_swap", "label_mismatched_swap")

# Clean censuses (quarantined no_chest/unencodable scans already excluded): valid 3001 and train
# 46,393. Both are donor sources for the matched arm — see build_donor_pool for why train is
# needed at all.
DEFAULT_DONOR_IDS = (
    "/workspace/data/ctrate_full/valid/ids.json",
    "/workspace/data/ctrate_full/train/ids.json",
)
# Wan mask latents (scripts/precompute_wan_mask_latents.py --out-dir); a donor without one
# cannot be conditioned on, so the pool is filtered by what exists here.
DEFAULT_MASK_LATENT_DIR = "/workspace/data/report2ct_wan/mask_latents_512x512x253"

_NOTE_MATCHED_EXACT = (
    "label vector identical to target, but the source patient's report findings are not — "
    "matched label vector is not the same as matched findings."
)
_NOTE_MISMATCHED = (
    "uniformly random cross-patient donor; same-vector donors excluded. label_overlap records "
    "the realized (uncapped) overlap."
)
_NOTE_NULL = (
    "no input mask — the model's learned no_mask_embed is used, so dice_to_input_mask is "
    "undefined for this row (only dice_to_gt_mask is meaningful)."
)

# One scan's 18 labels as a plain tuple of 0/1, e.g. (0, 1, 0, ...). Tuples (not arrays) so that
# `==` means "same abnormalities" and the vector can key a dict.
LabelVec = tuple[int, ...]


def read_ids(path: str | Path) -> list[str]:
    """Load scan ids from a ``.json`` (``{"ids": [...]}`` or a bare list) or a ``.txt``."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        ids = data["ids"] if isinstance(data, dict) else data
        return [str(i).strip() for i in ids if str(i).strip()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _git_sha() -> str:
    """Current repo HEAD sha for provenance, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def n_different_labels(a: LabelVec, b: LabelVec) -> int:
    """How many of the 18 labels differ between two scans (Hamming distance).

    ``0`` means the two scans carry exactly the same set of abnormalities (a normal scan matches
    another normal scan this way, since both vectors are all-zero).
    """
    return sum(x != y for x, y in zip(a, b))


def n_shared_positives(a: LabelVec, b: LabelVec) -> int:
    """How many abnormalities the two scans BOTH have (the schema's ``label_overlap``)."""
    return sum(x and y for x, y in zip(a, b))


def build_donor_pool(
    donor_ids_files: list[str | Path],
    mask_latent_dir: str | Path,
    label_map: dict[str, np.ndarray],
) -> tuple[list[str], set[str], dict]:
    """Donor pool = the given censuses ∩ has-labels, plus the subset that already has a mask latent.

    Two pools, because the two swap arms have opposite problems:

    * ``label_matched_swap`` needs an EXACT label twin, which is scarce (only 770 distinct label
      combinations exist among the 3001 valid scans, and a high-burden combination is often
      unique), so it searches the WHOLE census — train included — and accepts that a few donors
      still need their mask latent computed.
    * ``label_mismatched_swap`` needs any differently-labelled donor, which is abundant, so it
      draws only from scans whose mask latent ALREADY exists and costs no extra precompute.

    Returns:
        ``(sorted full pool, subset with a mask latent, counts dict)``. Sorting is load-bearing:
        donors are drawn by index into lists derived from these, so the order is part of the
        determinism guarantee.
    """
    census: set[str] = set()
    for path in donor_ids_files:
        census |= set(read_ids(path))
    suffix = "_mask_emb.nii.gz"
    with_latent = {
        p.name[: -len(suffix)] for p in Path(mask_latent_dir).glob(f"*{suffix}")
    }
    pool = sorted(census & set(label_map))
    cached = census & with_latent & set(label_map)
    if not cached:
        raise SystemExit(
            f"No id in {donor_ids_files} has a mask latent in {mask_latent_dir}. Run "
            "scripts/precompute_wan_mask_latents.py (wan env) first."
        )
    counts = {
        "census": len(census),
        "pool": len(pool),
        "pool_patients": len({patient_id_of(i) for i in pool}),
        "pool_with_mask_latent": len(cached),
        "dropped_no_labels": len(census - set(label_map)),
    }
    return pool, cached, counts


def require_learned_null_mask(ckpt_path: str | Path) -> None:
    """Abort unless the checkpoint carries the learned null mask ``no_mask_embed``.

    Gate on the checkpoint's own ``state_dict``, not on a model NAME: the model that supports
    ``condition=null`` is precisely the one trained with ``mask_cfg=true``. Older mask models
    (``report2ct_wan_mask``, ``report2ct_text2ct_mask``) have no learned null, and substituting
    an empty/zero mask feeds them an input they were never trained to see.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if "no_mask_embed" not in state_dict:
        raise SystemExit(
            f"condition=null refused: {ckpt_path} has no `no_mask_embed` in its state_dict, so "
            "this model has NO LEARNED NULL MASK. Only report2ct_wan_mask_v2 (trained with "
            "mask_cfg=true) supports condition=null; for report2ct_wan_mask / "
            "report2ct_text2ct_mask an empty or zero mask is an input the model never saw in "
            "training, so the resulting volumes would not measure 'no mask' — they would "
            "measure an out-of-distribution mask. Drop `null` from --conditions."
        )


def pick_matched_donor(
    target_id: str,
    vecs: dict[str, LabelVec],
    by_vector: dict[LabelVec, list[str]],
    cached: set[str],
    rng: random.Random,
) -> str | None:
    """A donor with EXACTLY the same 18 abnormalities, from a different patient.

    No near-match fallback: if no scan in the census carries this exact combination, the row is
    skipped (the caller records the target as uncovered). Among the exact twins, one whose mask
    latent already exists is preferred, so the arm costs as little extra precompute as possible.

    Args:
        target_id: the scan this row generates for.
        vecs: ``{scan_id: 18 labels as a 0/1 tuple}``.
        by_vector: census index, label vector -> the scans carrying it (sorted).
        cached: the subset of the census whose Wan mask latent already exists.
        rng: this row's private generator.

    Returns:
        The donor's scan id, or ``None`` when this target has no exact twin anywhere.
    """
    target_patient = patient_id_of(target_id)
    twins = [
        d
        for d in by_vector.get(vecs[target_id], [])
        if patient_id_of(d) != target_patient
    ]
    if not twins:
        return None
    return rng.choice([d for d in twins if d in cached] or twins)


def pick_mismatched_donor(
    target_id: str,
    vecs: dict[str, LabelVec],
    donors: list[str],
    rng: random.Random,
) -> str:
    """A uniformly random donor from another patient with a DIFFERENT label vector.

    Drawn only from scans whose mask latent already exists: unlike the matched arm this one has
    no scarcity problem, so it never needs extra precompute. Same-vector donors are excluded
    because ~3 % of uniform draws land on one (mostly normal↔normal), which would quietly turn
    part of the "mismatched" arm into a second matched arm.

    Rejection sampling keeps the draw uniform over the eligible donors without materialising the
    filtered list per target (the pool is ~7k ids and well over 90 % of it is eligible, so this
    almost always accepts on the first try).
    """
    target_patient, target_vec = patient_id_of(target_id), vecs[target_id]
    for _ in range(1000):
        donor = donors[rng.randrange(len(donors))]
        if patient_id_of(donor) != target_patient and vecs[donor] != target_vec:
            return donor
    raise SystemExit(
        f"No cross-patient donor with a DIFFERENT label vector found for {target_id} in "
        f"{len(donors)} draws — cannot build label_mismatched_swap."
    )


def build_rows(
    target_ids: list[str],
    pool: list[str],
    cached: set[str],
    label_map: dict[str, np.ndarray],
    conditions: list[str],
    seed: int,
    cfg_text: float,
    cfg_mask: float,
    run_id: str,
    ckpt: str,
) -> tuple[list[dict], list[str]]:
    """Build every manifest row (targets outer, conditions inner).

    Returns:
        ``(rows, uncovered_targets)`` — ``uncovered_targets`` are the targets with no exact label
        twin in the census, which therefore have NO ``label_matched_swap`` row (their other
        conditions are still generated). They are returned rather than logged away so the caller
        can record them: a matched arm smaller than the target set is a fact the analysis must
        see, not a silent gap.
    """
    missing = [t for t in target_ids if t not in label_map]
    if missing:
        raise SystemExit(f"Targets absent from the 18-label CSV: {missing[:5]}")
    vecs: dict[str, LabelVec] = {
        i: tuple(int(v) for v in label_map[i]) for i in set(pool) | set(target_ids)
    }
    by_vector: dict[LabelVec, list[str]] = {}
    for donor in pool:
        by_vector.setdefault(vecs[donor], []).append(donor)
    mismatch_donors = sorted(cached)

    rows: list[dict] = []
    uncovered: list[str] = []
    for target_id in target_ids:
        target_vec = vecs[target_id]

        for condition in conditions:
            # Per-(condition, target) stream: adding a condition or reordering --conditions
            # leaves every other row's donor untouched. Python seeds a str via sha512, so this
            # is stable across processes and machines (unlike hash()).
            rng = random.Random(f"{seed}|{condition}|{target_id}")
            note: str | None = None
            if condition == "gt":
                source_id = target_id
            elif condition == "null":
                source_id, note = None, _NOTE_NULL
            elif condition == "label_matched_swap":
                source_id = pick_matched_donor(target_id, vecs, by_vector, cached, rng)
                if source_id is None:  # no exact twin exists — omit the row entirely
                    uncovered.append(target_id)
                    continue
                note = _NOTE_MATCHED_EXACT
            else:
                source_id = pick_mismatched_donor(target_id, vecs, mismatch_donors, rng)
                note = _NOTE_MISMATCHED

            is_swap = condition in SWAP_CONDITIONS
            exact_match = is_swap and vecs[source_id] == target_vec
            hamming = (
                n_different_labels(vecs[source_id], target_vec) if is_swap else None
            )
            rows.append(
                {
                    "sample_id": (
                        f"{target_id}__{condition}__src-{source_id or 'none'}"
                        f"__sm-{cfg_mask}__seed-{seed}"
                    ),
                    "target_id": target_id,
                    "condition": condition,
                    "cond_mask_source_id": source_id,
                    "seed": seed,
                    "cfg_scale_text": cfg_text,
                    "cfg_scale_mask": cfg_mask,
                    "run_id": run_id,
                    "ckpt": ckpt,
                    "target_labels": dict(zip(LABEL_NAMES, target_vec)),
                    "source_labels": dict(zip(LABEL_NAMES, vecs[source_id]))
                    if is_swap
                    else None,
                    "label_overlap": n_shared_positives(target_vec, vecs[source_id])
                    if is_swap
                    else None,
                    "label_exact_match": exact_match,
                    "label_hamming": hamming,
                    "report_note": note,
                    # Derived by the consumer as `pred_dir / f"{sample_id}.mha"`
                    # (persample.build_per_sample) — left null so the manifest never carries a
                    # path that can go stale relative to the run that used it.
                    "gen_path": None,
                }
            )
    return rows, uncovered


def write_manifest(rows: list[dict], out_path: str | Path) -> Path:
    """Write the JSONL. No timestamps/paths inside rows -> same seed ⇒ byte-identical file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    return out_path


def _reread_manifest(out_path: Path) -> tuple[int, int]:
    """Round-trip through the CONSUMER's reader: ``(records kept, warnings emitted)``.

    ``persample._read_manifest`` warns-and-skips a bad line instead of raising, so a silent loss
    only shows up as a lower record count or a warning — both are returned and checked.
    """
    warnings: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    logger = logging.getLogger(persample_mod.__name__)
    logger.addHandler(handler)
    try:
        records = persample_mod._read_manifest(out_path)
    finally:
        logger.removeHandler(handler)
    return len(records), len(warnings)


def verify(
    rows: list[dict],
    out_path: Path,
    label_map: dict[str, np.ndarray],
    n_targets: int,
    conditions: list[str],
    uncovered: list[str],
) -> dict:
    """Re-check the written manifest against the generation-side rules; print and return numbers.

    Covers acceptance checks 1/2/4/5/6 (3 = determinism is a two-build comparison, fixed by
    ``tests/eval_analysis/test_manifest_builder.py``). Every label field is recomputed from the
    CSV so a wrong value cannot verify itself.

    The expected row count is ``len(conditions) * n_targets`` MINUS the targets with no exact
    label twin, which get no ``label_matched_swap`` row (``uncovered``).
    """
    vecs = {i: tuple(int(v) for v in vec) for i, vec in label_map.items()}
    swaps = [r for r in rows if r["condition"] in SWAP_CONDITIONS]
    matched = [r for r in rows if r["condition"] == "label_matched_swap"]
    mismatched = [r for r in rows if r["condition"] == "label_mismatched_swap"]
    sample_ids = [r["sample_id"] for r in rows]
    n_records, n_warnings = _reread_manifest(out_path)

    expected_rows = len(conditions) * n_targets - len(uncovered)
    non_exact_matched = [r for r in matched if not r["label_exact_match"]]
    overlaps = [r["label_overlap"] for r in mismatched]

    report = {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "targets": n_targets,
        "matched_covered": len(matched),
        "matched_uncovered": len(uncovered),
        "matched_not_exact": len(non_exact_matched),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "self_pairs": sum(
            1 for r in swaps if r["cond_mask_source_id"] == r["target_id"]
        ),
        "patient_leaks": sum(
            1
            for r in swaps
            if patient_id_of(r["cond_mask_source_id"]) == patient_id_of(r["target_id"])
        ),
        "roundtrip_records": n_records,
        "roundtrip_warnings": n_warnings,
        "label_overlap_recompute_mismatches": sum(
            1
            for r in swaps
            if r["label_overlap"]
            != n_shared_positives(vecs[r["target_id"]], vecs[r["cond_mask_source_id"]])
        ),
        "mismatched_same_vector_draws": sum(
            1
            for r in mismatched
            if vecs[r["target_id"]] == vecs[r["cond_mask_source_id"]]
        ),
        "donors_from_train": len(
            {
                r["cond_mask_source_id"]
                for r in swaps
                if r["cond_mask_source_id"].startswith("train_")
            }
        ),
    }
    if overlaps:
        report["mismatched_overlap_mean"] = float(np.mean(overlaps))
        report["mismatched_overlap_median"] = float(np.median(overlaps))
        report["mismatched_overlap_p90"] = float(np.percentile(overlaps, 90))
        report["mismatched_overlap_max"] = max(overlaps)

    problems = [
        name
        for name, bad in (
            ("row count", report["rows"] != expected_rows),
            ("duplicate sample_id", report["duplicate_sample_ids"]),
            ("self-pair", report["self_pairs"]),
            ("patient leak", report["patient_leaks"]),
            ("round-trip loss", report["roundtrip_records"] != report["rows"]),
            ("round-trip warning", report["roundtrip_warnings"]),
            ("label_overlap", report["label_overlap_recompute_mismatches"]),
            ("mismatched same-vector donor", report["mismatched_same_vector_draws"]),
            ("matched donor is not an exact twin", report["matched_not_exact"]),
        )
        if bad
    ]
    report["ok"] = not problems

    print("\n--- manifest verification ------------------------------------------")
    print(
        f"  rows                          {report['rows']} (expected {expected_rows})"
    )
    print(f"  duplicate sample_id           {report['duplicate_sample_ids']}")
    print(f"  self-pairs (swap src==target) {report['self_pairs']}")
    print(f"  patient leaks (patient_id)    {report['patient_leaks']}")
    print(
        f"  _read_manifest round-trip     {report['roundtrip_records']} records, "
        f"{report['roundtrip_warnings']} warnings"
    )
    print(
        f"  label_overlap recompute       "
        f"{report['label_overlap_recompute_mismatches']} mismatches"
    )
    if matched or uncovered:
        print(
            f"  matched (exact twin only)     {len(matched)}/{n_targets} targets covered, "
            f"{len(uncovered)} have no twin anywhere -> no matched row"
        )
        print(f"    donors taken from train     {report['donors_from_train']}")
    if mismatched:
        print(
            f"  mismatched overlap: mean {report['mismatched_overlap_mean']:.2f}, "
            f"median {report['mismatched_overlap_median']:.0f}, "
            f"p90 {report['mismatched_overlap_p90']:.0f}, "
            f"max {report['mismatched_overlap_max']}; "
            f"same-vector draws {report['mismatched_same_vector_draws']}"
        )
    print(f"  VERDICT                       {'PASS' if report['ok'] else 'FAIL'}")
    print("--------------------------------------------------------------------\n")
    if problems:
        raise SystemExit(f"manifest verification FAILED: {', '.join(problems)}")
    return report


def _write_todo_sidecars(
    out_path: Path, rows: list[dict], uncovered: list[str], cached: set[str]
) -> dict[str, Path]:
    """Write the two "before you generate" lists next to the manifest.

    * ``<out>.needs_mask_latent.{valid,train}.json`` — donors this manifest names whose Wan mask
      latent does not exist yet. Generation would fail on them, so they are listed in the exact
      shape ``precompute_wan_mask_latents.py --ids-file`` expects, split by ``--split`` since
      that script takes one split per invocation.
    * ``<out>.uncovered_targets.json`` — targets with no exact label twin, hence no
      ``label_matched_swap`` row. Recorded as data so the analysis can state how many targets
      the matched arm actually covers instead of quietly comparing unequal sets.

    Returns:
        ``{name: path}`` for the files actually written.
    """
    donors = {
        r["cond_mask_source_id"] for r in rows if r["condition"] in SWAP_CONDITIONS
    }
    written: dict[str, Path] = {}
    for split in ("valid", "train"):
        todo = sorted(d for d in donors - cached if d.startswith(f"{split}_"))
        if not todo:
            continue
        path = out_path.with_suffix(f"{out_path.suffix}.needs_mask_latent.{split}.json")
        path.write_text(json.dumps({"ids": todo}, indent=2))
        written[f"needs_mask_latent_{split}"] = path
        print(
            f"⚠ {len(todo)} {split} donor(s) have no Wan mask latent yet. Precompute them first:\n"
            f"    CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/wan/bin/python "
            f"scripts/precompute_wan_mask_latents.py \\\n"
            f"        --ids-file {path} --split {split}_fixed \\\n"
            f"        --out-dir {DEFAULT_MASK_LATENT_DIR} --device cuda:0"
        )
    if uncovered:
        path = out_path.with_suffix(f"{out_path.suffix}.uncovered_targets.json")
        path.write_text(json.dumps({"ids": sorted(uncovered)}, indent=2))
        written["uncovered_targets"] = path
        print(f"  {len(uncovered)} target(s) with no exact label twin -> {path}")
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a mask-intervention JSONL manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, required=True, help="number of target scans (valid_v2)"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="donor-pairing seed (recorded per row)"
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="intervention conditions to emit (one row per target per condition)",
    )
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument(
        "--ckpt",
        required=True,
        help="generation checkpoint (provenance; also gates condition=null via no_mask_embed)",
    )
    p.add_argument(
        "--run-id", required=True, help="provenance tag, e.g. wanmaskv2_ep299"
    )
    p.add_argument(
        "--cfg-text",
        type=float,
        required=True,
        help="s_t recorded in every row (no default)",
    )
    p.add_argument(
        "--cfg-mask",
        type=float,
        required=True,
        help="s_m recorded in every row and in sample_id (no default). condition and s_m are "
        "ORTHOGONAL axes: each s_m value is its own generation pass, so build one manifest per "
        "s_m — not one manifest reinterpreted afterwards.",
    )
    p.add_argument(
        "--donor-ids",
        nargs="+",
        default=list(DEFAULT_DONOR_IDS),
        help="donor census id list(s). Default = the clean valid (3001) AND train (46,393) "
        "censuses: an exact label twin is scarce enough that valid alone leaves ~36%% of "
        "targets without one.",
    )
    p.add_argument(
        "--mask-latent-dir",
        default=DEFAULT_MASK_LATENT_DIR,
        help="donors are restricted to scans with a precomputed Wan mask latent here",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not Path(args.ckpt).is_file():
        raise SystemExit(
            f"--ckpt not found: {args.ckpt} (it is recorded as provenance)"
        )
    if "null" in args.conditions:
        require_learned_null_mask(args.ckpt)

    # Donors may come from either split, so both label CSVs are needed (ids are disjoint:
    # "valid_*" vs "train_*").
    label_map = {**load_label_csv("valid"), **load_label_csv("train")}
    targets = [c.scan_id for c in load_eval_cases(n_samples=args.n)]
    if len(targets) != args.n:
        raise SystemExit(
            f"load_eval_cases({args.n}) returned {len(targets)} cases — asked for {args.n}."
        )
    pool, cached, pool_counts = build_donor_pool(
        args.donor_ids, args.mask_latent_dir, label_map
    )
    print(
        f"Donor pool: {pool_counts['pool']} scans / {pool_counts['pool_patients']} patients; "
        f"{pool_counts['pool_with_mask_latent']} already have a Wan mask latent in "
        f"{args.mask_latent_dir}"
    )

    rows, uncovered = build_rows(
        target_ids=targets,
        pool=pool,
        cached=cached,
        label_map=label_map,
        conditions=list(args.conditions),
        seed=args.seed,
        cfg_text=args.cfg_text,
        cfg_mask=args.cfg_mask,
        run_id=args.run_id,
        ckpt=str(args.ckpt),
    )
    out_path = write_manifest(rows, args.out)
    report = verify(
        rows, out_path, label_map, len(targets), list(args.conditions), uncovered
    )
    _write_todo_sidecars(out_path, rows, uncovered, cached)

    # Provenance sidecar — deliberately NOT inside the JSONL, whose byte-identity across two
    # same-seed builds is one of the acceptance checks.
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "kind": "mask_intervention_manifest",
                "created": datetime.now().isoformat(timespec="seconds"),
                "git_sha": _git_sha(),
                "args": vars(args),
                "donor_pool": pool_counts,
                "verification": report,
                "uncovered_targets": sorted(uncovered),
                "scope": (
                    "DIAGNOSTIC ONLY (CLIPScore-T2I, dice_to_input_mask, dice_to_gt_mask). "
                    "FID/FVD are invalid on a manifest run: one target has several generated "
                    "volumes, breaking the '1 volume = 1 independent sample' premise."
                ),
            },
            indent=2,
        )
    )
    print(f"Manifest: {out_path}  ({len(rows)} rows)\nProvenance: {meta_path}")


if __name__ == "__main__":
    main()
