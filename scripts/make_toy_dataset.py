#!/usr/bin/env python
"""Build a representative, self-contained CT-RATE toy dataset (v2).

Plan: /root/.claude/plans/inherited-seeking-pike.md

Two cleanly-separated, documented pieces under ``--out-dir`` (default
``/workspace/data/ctrate_toy_v2``):

  train/       5000 ``train_fixed`` scans (reused from ``split.json``), latents symlinked
               (no 614 GB copy). Already representative of the full train pool.
  valid_v2/  ``valid_fixed`` one scan per patient = 1304 (frozen id list + prompts). This
               is the evaluation stand-in for the (hidden) challenge test set. Replaces the old
               ``df.head(1000)`` selection that covered only ~425 of 1304 valid patients.

The lab's train-drawn dev-valid (``datasets/latents/valid``, all ``train_*``) is DEPRECATED
and intentionally NOT carried over — it is left untouched in the read-only source area.

Heavy GT ``.mha`` (~40-90 GB for 1304 volumes) is NOT built here by default: ``run_eval.py``
builds it idempotently at eval time via ``prepare_valid_gt``. Pass ``--build-gt`` to do it now.

Examples
--------
    python scripts/make_toy_dataset.py                       # symlink pt latents, no GT
    python scripts/make_toy_dataset.py --latent-source emb   # symlink fp32 _emb.nii.gz
    python scripts/make_toy_dataset.py --copy --build-gt     # fully self-contained
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ct_rate_cases import (  # noqa: E402
    _volume_name_to_id,
    _volume_name_to_nifti,
    load_eval_cases,
    prepare_valid_gt,
    write_prompt_xlsx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_toy_dataset")

# --- source paths (read-only) ---
DS = Path("/workspace/datasets/datasets/CT-RATE/dataset")
SPLIT_JSON = Path("/workspace/datasets/split.json")
LAT_PT_TRAIN = Path("/workspace/datasets/datasets/latents/train")
LAT_STATS = Path("/workspace/datasets/datasets/latents/stats.json")
EMB_DIR = Path("/workspace/data/report2ct_work_dir/image_embeddings")
TRAIN_META = DS / "metadata" / "train_metadata.csv"
VALID_META = DS / "metadata" / "validation_metadata.csv"
TRAIN_LABELS = DS / "multi_abnormality_labels" / "train_predicted_labels.csv"
VALID_LABELS = DS / "multi_abnormality_labels" / "valid_predicted_labels.csv"
DEFAULT_GT_DIR = Path("/workspace/data/vlm3d_eval/_valid_full_3001")

CAT_AXES = ["Manufacturer", "ConvolutionKernel", "PatientSex"]
CONT_AXES = ["ZSpacing", "NumberofSlices", "PatientAge"]
TVD_GATE = 0.05
KS_GATE = 0.05


# --------------------------------------------------------------------------- helpers
def _pid(name: str) -> str | None:
    m = re.match(r"((?:train|valid)_\d+)", str(name))
    return m.group(1) if m else None


def _tvd(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return float(
        0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()
    )


def _num(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        m = re.search(r"-?\d+\.?\d*", str(v))
        out.append(float(m.group()) if m else np.nan)
    arr = np.asarray(out, dtype=float)
    return arr[~np.isnan(arr)]


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


# --------------------------------------------------------------------------- train
def build_train(out: Path, latent_source: str, copy: bool) -> list[str]:
    sp = json.loads(SPLIT_JSON.read_text())
    train_ids = sorted(
        {Path(e["path"]).name.replace(".nii.gz", "") for e in sp["train"]}
    )

    tdir = out / "train"
    (tdir / "latents").mkdir(parents=True, exist_ok=True)
    (tdir / "ids.json").write_text(
        json.dumps(
            {
                "split": "train",
                "source": "train_fixed (split.json)",
                "n": len(train_ids),
                "ids": train_ids,
            },
            indent=2,
        )
    )

    missing: list[str] = []
    for sid in train_ids:
        if latent_source == "pt":
            src, dst = LAT_PT_TRAIN / sid, tdir / "latents" / sid
        else:  # emb
            src, dst = (
                EMB_DIR / f"{sid}_emb.nii.gz",
                tdir / "latents" / f"{sid}_emb.nii.gz",
            )
        if not src.exists():
            missing.append(sid)
            continue
        if dst.is_symlink() or dst.exists():
            (shutil.rmtree if dst.is_dir() and not dst.is_symlink() else Path.unlink)(
                dst
            )
        if copy:
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        else:
            dst.symlink_to(src)
    if missing:
        raise SystemExit(
            f"{len(missing)} train ids missing '{latent_source}' latents, e.g. {missing[:5]}"
        )

    stats = json.loads(LAT_STATS.read_text())
    if stats.get("num_volumes") != len(train_ids):
        raise SystemExit(
            f"stats.json num_volumes={stats.get('num_volumes')} != {len(train_ids)} train ids; recompute needed."
        )
    (tdir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info(
        "train: %d ids, latents %s under %s",
        len(train_ids),
        "copied" if copy else "symlinked",
        tdir / "latents",
    )
    return train_ids


# --------------------------------------------------------------------------- valid_v2
def select_valid_v2_ids(seed: int = 42) -> list[str]:
    """One scan-``a`` reconstruction per valid_fixed patient, seed-fixed uniform random.

    CT-RATE reconstructs each scan into a sharp/soft kernel pair (``_a_1``/``_a_2``) that
    *share the same report*; the legacy ``_a_1``-preferred rule always took the sharp recon and
    skewed the reconstruction-kernel marginal (B 28%->5%, YA 26%->51%, χ² p≈0). Here we keep one
    scan per patient (scan ``a`` — the primary acquisition; every valid patient has one) and draw
    one of its clean recons uniformly at random with a fixed ``seed``, de-biasing the kernel
    distribution while leaving the report/prompt unchanged (recons share the report). Longitudinal
    extra scans (``_b``/``_c``…) are intentionally excluded (same-patient separate studies).
    Excludes no_chest/error-quarantined scans and any volume whose NIfTI is missing.

    Args:
        seed: RNG seed for the per-patient uniform recon draw (reproducible id list).

    Returns:
        Sorted list of 1304 volume ids (one scan-``a`` recon per patient).
    """
    from tests.mask_vae.dataset import excluded_scan_ids  # no_chest/error quarantine

    excluded = excluded_scan_ids()
    meta = pd.read_csv(VALID_META)
    by_patient: dict[str, list[str]] = {}
    for vn in sorted(str(v) for v in meta["VolumeName"]):  # deterministic input order
        vid = _volume_name_to_id(vn)  # "valid_4_a_1.nii.gz" -> "valid_4_a_1"
        parts = vid.split("_")  # ["valid", "4", "a", "1"]
        if parts[2] != "a":  # scan-a only (drop _b/_c longitudinal studies)
            continue
        if vid in excluded or not _volume_name_to_nifti(vn).is_file():
            continue
        pid = "_".join(parts[:2])  # "valid_4"
        by_patient.setdefault(pid, []).append(vid)
    rng = np.random.default_rng(seed)
    chosen = [
        str(rng.choice(by_patient[p])) for p in sorted(by_patient)
    ]  # 1 recon / patient
    return sorted(chosen)


def build_valid_v2(out: Path, valid_v2_ids: list[str]) -> list:
    pdir = out / "valid_v2"
    pdir.mkdir(parents=True, exist_ok=True)
    ids_path = pdir / "ids.json"
    ids_path.write_text(
        json.dumps(
            {
                "split": "valid_v2",
                "source": "valid_fixed",
                "selection": "one scan-a recon per patient (seed=42 uniform-random over clean recons)",
                "seed": 42,
                "n": len(valid_v2_ids),
                "ids": valid_v2_ids,
            },
            indent=2,
        )
    )
    cases = load_eval_cases(
        ids_json=ids_path
    )  # single source of truth for case construction
    write_prompt_xlsx(cases, pdir / "prompts.xlsx")
    (pdir / "gt_dir.txt").write_text(str(DEFAULT_GT_DIR) + "\n")
    log.info(
        "valid_v2: %d ids, prompts.xlsx written; GT dir -> %s",
        len(valid_v2_ids),
        DEFAULT_GT_DIR,
    )
    return cases


# --------------------------------------------------------------------------- report
def _cat_row(full: pd.DataFrame, sub: pd.DataFrame, col: str) -> tuple[float, dict]:
    base = full[col].astype(str).value_counts(normalize=True)
    cur = sub[col].astype(str).value_counts(normalize=True)
    return _tvd(base, cur), {k: round(v, 3) for k, v in cur.head(4).items()}


def make_report(out: Path, train_ids: list[str], valid_v2_ids: list[str]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = out / "report" / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(TRAIN_META)
    tr["vn"] = tr["VolumeName"].astype(str)
    va = pd.read_csv(VALID_META)
    va["vn"] = va["VolumeName"].astype(str)

    full_tr_a1 = tr[tr["vn"].str.endswith("_a_1.nii.gz")]
    toy_tr = tr[tr["vn"].map(lambda v: v.replace(".nii.gz", "")).isin(set(train_ids))]
    valid_v2_set = set(valid_v2_ids)
    valid_v2_df = va[
        va["vn"].map(lambda v: v.replace(".nii.gz", "")).isin(valid_v2_set)
    ]

    full_va_a1 = va[
        va["vn"].str.endswith("_a_1.nii.gz")
    ]  # legacy _a_1-only reference (pre-de-bias)

    dist: dict = {}
    lines = ["# CT-RATE toy dataset v2 — representativeness report", ""]

    # Only the toy-train block is GATED: it is a 5000/20000 subsample, so representativeness
    # is a real property to enforce. valid_v2 is a full one-per-patient set (every valid_fixed
    # patient, one scan-a recon each, kernel-de-biased by a random recon draw) — no sampling to
    # gate. The two valid_v2 blocks are reported for transparency: block 2 shows how far the
    # random-recon draw moved off the legacy _a_1-only selection (the kernel de-bias delta);
    # block 3 shows kernel/geometry representativeness vs all valid scans (now close, since the
    # recon is no longer _a_1-fixed).
    comparisons = [
        (
            "toy-train (5000) vs train_fixed one-per-patient (20000) — GATED (subsample)",
            full_tr_a1,
            toy_tr,
            "train_vs_train_fixed_a1",
            True,
            "A 5000/20000 subsample; gated axes must stay within TVD<0.05 / KS<0.05.",
        ),
        (
            "valid_v2 (1304) vs valid_fixed _a_1-only (1304) — info: kernel de-bias delta",
            full_va_a1,
            valid_v2_df,
            "valid_v2_vs_valid_fixed_a1",
            False,
            "valid_v2 now draws a random scan-a recon per patient, so this quantifies the shift "
            "away from the legacy _a_1-only selection (mainly ConvolutionKernel).",
        ),
        (
            "valid_v2 (1304, random scan-a recon) vs valid_fixed ALL scans (3039) — info, NOT gated",
            va,
            valid_v2_df,
            "valid_v2_vs_valid_fixed_all",
            False,
            "Kernel/geometry representativeness: the random-recon selection matches the all-scans "
            "ConvolutionKernel marginal (the _a_1-preferred bias is fixed).",
        ),
    ]
    for label, full, sub, key, gated, note in comparisons:
        dist[key] = {}
        lines += [
            f"## {label}",
            "",
            f"_{note}_",
            "",
            "| axis | distance | top / median |",
            "|---|---|---|",
        ]
        for col in CAT_AXES:
            tvd, top = _cat_row(full, sub, col)
            dist[key][col] = {"metric": "TVD", "value": round(tvd, 4), "gated": gated}
            flag = " ⚠" if (gated and tvd >= TVD_GATE) else ""
            lines.append(f"| {col} | TVD={tvd:.4f}{flag} | {top} |")
        for col in CONT_AXES:
            ks = _ks(_num(full[col]), _num(sub[col]))
            dist[key][col] = {"metric": "KS", "value": round(ks, 4), "gated": gated}
            flag = " ⚠" if (gated and ks >= KS_GATE) else ""
            lines.append(
                f"| {col} | KS={ks:.4f}{flag} | median {np.median(_num(sub[col])):.1f} |"
            )
        lines.append("")

    # patient coverage statement for valid_v2
    cov = valid_v2_df["vn"].map(_pid).nunique()
    lines += [
        "## valid_v2 patient coverage",
        "",
        f"- unique patients: **{cov} / 1304** (one scan each, redundancy {len(valid_v2_df) / max(cov, 1):.2f}x)",
        f"- replaces legacy head-1000 (425 patients, 2.4x redundancy, patients 426..1304 dropped).",
        "",
    ]

    # figures: continuous overlays + categorical bars
    for col in CONT_AXES:
        plt.figure(figsize=(5, 3))
        plt.hist(
            _num(full_tr_a1[col]),
            bins=40,
            density=True,
            alpha=0.5,
            label="train_fixed a1",
        )
        plt.hist(_num(toy_tr[col]), bins=40, density=True, alpha=0.5, label="toy-train")
        plt.hist(
            _num(valid_v2_df[col]), bins=40, density=True, alpha=0.5, label="valid_v2"
        )
        plt.title(col)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figs / f"{col}.png", dpi=90)
        plt.close()

    (out / "report" / "REPORT.md").write_text("\n".join(lines))
    log.info("report written -> %s", out / "report" / "REPORT.md")
    return dist


# --------------------------------------------------------------------------- assertions
def assert_invariants(
    train_ids: list[str], valid_v2_ids: list[str], cases: list, dist: dict
) -> None:
    train_pat = {_pid(s) for s in train_ids}
    valid_v2_pat = {_pid(s) for s in valid_v2_ids}
    assert not (train_pat & valid_v2_pat), (
        "train/valid_v2 patient overlap (unexpected: should be train_ vs valid_)"
    )
    assert len(valid_v2_pat) == len(valid_v2_ids), (
        "valid_v2 is not one-scan-per-patient"
    )
    empty = [
        c.scan_id for c in cases if not (c.findings.strip() or c.impression.strip())
    ]
    assert not empty, (
        f"{len(empty)} valid_v2 cases have an empty report, e.g. {empty[:5]}"
    )
    bad = [
        f"{key}.{ax}={v['value']}"
        for key, axes in dist.items()
        for ax, v in axes.items()
        if v.get("gated")
        and (
            (v["metric"] == "TVD" and v["value"] >= TVD_GATE)
            or (v["metric"] == "KS" and v["value"] >= KS_GATE)
        )
    ]
    if bad:
        raise AssertionError(f"distribution gate exceeded: {bad}")
    log.info(
        "assertions passed: patient-disjoint, one-per-patient, reports non-empty, distance gates OK"
    )


# --------------------------------------------------------------------------- README / manifest
def write_docs(
    out: Path, train_ids: list[str], valid_v2_ids: list[str], dist: dict, args
) -> None:
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "latent_source": args.latent_source,
                "materialise": "copy" if args.copy else "symlink",
                "counts": {"train": len(train_ids), "valid_v2": len(valid_v2_ids)},
                "gt_dir": str(DEFAULT_GT_DIR),
                "sources": {
                    "train_ids": str(SPLIT_JSON),
                    "train_latents": str(
                        LAT_PT_TRAIN if args.latent_source == "pt" else EMB_DIR
                    ),
                    "valid_v2": str(VALID_META),
                },
                "distances": dist,
                "deprecated": "datasets/latents/valid (lab train-drawn dev-valid) — not used",
            },
            indent=2,
        )
    )
    (out / "README.md").write_text(
        "# CT-RATE toy dataset v2\n\n"
        "Built by `scripts/make_toy_dataset.py` (plan: inherited-seeking-pike.md).\n\n"
        "- **train/** — 5000 `train_fixed` scans (`split.json`), latents "
        f"{'copied' if args.copy else 'symlinked'} ({args.latent_source}). Already representative.\n"
        "- **valid_v2/** — `valid_fixed` one scan per patient = "
        f"{len(valid_v2_ids)}. Frozen `ids.json` (the evaluation stand-in for the hidden challenge test).\n"
        "  GT `.mha` lives under `valid_v2/gt_dir.txt` and is built idempotently by `run_eval.py`.\n\n"
        "## Deprecated\n"
        "- `datasets/latents/valid` + `split.json:valid` (lab train-drawn dev-valid, all `train_*`) "
        "are **deprecated** and unused by the ctgen train/eval path. Left untouched (read-only source).\n"
        "- Follow-up: `src/data/fvlm_organ_report.py` still reads `split.json:valid`; align to official valid later.\n"
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="/workspace/data/ctrate_toy_v2", type=Path)
    ap.add_argument("--latent-source", choices=["pt", "emb"], default="pt")
    ap.add_argument(
        "--copy", action="store_true", help="copy latents instead of symlink"
    )
    ap.add_argument(
        "--build-gt",
        action="store_true",
        help="materialise valid_v2 GT .mha now (~40-90 GB)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log.info("building toy dataset v2 -> %s", out)

    train_ids = build_train(out, args.latent_source, args.copy)
    valid_v2_ids = select_valid_v2_ids(args.seed)
    cases = build_valid_v2(out, valid_v2_ids)
    dist = make_report(out, train_ids, valid_v2_ids)
    assert_invariants(train_ids, valid_v2_ids, cases, dist)
    write_docs(out, train_ids, valid_v2_ids, dist, args)

    if args.build_gt:
        log.info("building valid_v2 GT .mha (idempotent) -> %s", DEFAULT_GT_DIR)
        prepare_valid_gt(cases, DEFAULT_GT_DIR)

    log.info(
        "DONE: train=%d valid_v2=%d  out=%s", len(train_ids), len(valid_v2_ids), out
    )


if __name__ == "__main__":
    main()
