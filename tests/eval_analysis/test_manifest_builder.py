"""Acceptance checks for the mask-intervention manifest builder
(``scripts/build_mask_intervention_manifest.py``): self-pairing, patient leakage, determinism,
coverage/uniqueness, the ``persample._read_manifest`` round-trip (which warns-and-skips rather
than raising, so its warning log is inspected too), and the label bookkeeping that distinguishes
an exactly-matched donor from a nearest-Hamming fallback.

Runs against the real CT-RATE label CSV / valid_v2 id list / Wan mask-latent cache — the same
inputs the builder uses in production. Only the checkpoint is synthetic (a 1-tensor stand-in for
report2ct_wan_mask_v2's learned ``no_mask_embed``), so no 2.8 GB load is needed to exercise the
``condition=null`` gate.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, "/workspace/scripts")

import build_mask_intervention_manifest as builder  # noqa: E402

from src.eval.analysis.labels import load_label_csv, patient_id_of  # noqa: E402
from src.eval.analysis.persample import _read_manifest  # noqa: E402

N_TARGETS = 12
CONDITIONS = ["gt", "label_matched_swap", "label_mismatched_swap", "null"]


def _build(out_path: Path, ckpt: Path, seed: int = 0) -> None:
    """Invoke the builder's CLI entrypoint (exercises argparse + verify(), like production)."""
    argv = [
        "build_mask_intervention_manifest.py",
        "--n",
        str(N_TARGETS),
        "--seed",
        str(seed),
        "--conditions",
        *CONDITIONS,
        "--out",
        str(out_path),
        "--ckpt",
        str(ckpt),
        "--run-id",
        "test_run",
        "--cfg-text",
        "5.0",
        "--cfg-mask",
        "1.0",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        builder.main()
    finally:
        sys.argv = old


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> SimpleNamespace:
    """Two same-seed builds (+ one different-seed build) and the parsed rows of the first."""
    d = tmp_path_factory.mktemp("mask_intervention")
    ckpt = d / "fake_wan_mask_v2.ckpt"
    torch.save({"state_dict": {"no_mask_embed": torch.zeros(16)}}, ckpt)

    out_a, out_b, out_seed1 = d / "a.jsonl", d / "b.jsonl", d / "seed1.jsonl"
    _build(out_a, ckpt)
    _build(out_b, ckpt)
    _build(out_seed1, ckpt, seed=1)

    rows = [json.loads(line) for line in out_a.read_text().splitlines()]
    return SimpleNamespace(
        dir=d,
        ckpt=ckpt,
        out_a=out_a,
        out_b=out_b,
        out_seed1=out_seed1,
        rows=rows,
        labels=load_label_csv("valid"),
    )


def _swaps(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["condition"].endswith("_swap")]


def test_no_self_pairs(built):
    """(1) A swap row never conditions on the target's own mask."""
    offenders = [
        r for r in _swaps(built.rows) if r["cond_mask_source_id"] == r["target_id"]
    ]
    assert offenders == [], offenders
    assert len(_swaps(built.rows)) == 2 * N_TARGETS  # the check actually saw rows


def test_no_patient_leakage(built):
    """(2) Donor and target are different PATIENTS, not merely different scans.

    The census averages 2.30 scans/patient, so a scan-id-only check would let a different
    reconstruction/series of the same patient through.
    """
    offenders = [
        (r["target_id"], r["cond_mask_source_id"])
        for r in _swaps(built.rows)
        if patient_id_of(r["cond_mask_source_id"]) == patient_id_of(r["target_id"])
    ]
    assert offenders == [], offenders


def test_deterministic_bytes(built):
    """(3) Same seed -> byte-identical JSONL; a different seed actually moves the donors."""
    assert built.out_a.read_bytes() == built.out_b.read_bytes()

    other = [json.loads(line) for line in built.out_seed1.read_text().splitlines()]
    changed = sum(
        1
        for r, o in zip(_swaps(built.rows), _swaps(other))
        if r["cond_mask_source_id"] != o["cond_mask_source_id"]
    )
    assert changed > 0, "seed had no effect on donor selection"


def test_coverage_and_uniqueness(built):
    """(4) rows == len(conditions) x n, and every sample_id is unique."""
    assert len(built.rows) == len(CONDITIONS) * N_TARGETS
    sample_ids = [r["sample_id"] for r in built.rows]
    assert len(set(sample_ids)) == len(sample_ids)
    assert {r["target_id"] for r in built.rows} == set(
        r["target_id"] for r in built.rows if r["condition"] == "gt"
    )
    for condition in CONDITIONS:
        assert sum(1 for r in built.rows if r["condition"] == condition) == N_TARGETS


def test_roundtrip_through_read_manifest(built, caplog):
    """(5) The consumer reads every line: 0 skipped AND 0 warnings.

    ``_read_manifest`` skips a malformed line with a ``log.warning`` instead of raising, so the
    record count alone can hide a silent loss only if two lines shared a sample_id — the warning
    log is the direct evidence and is asserted here.
    """
    caplog.set_level(logging.WARNING, logger="src.eval.analysis.persample")
    records = _read_manifest(built.out_a)
    assert len(records) == len(built.rows)
    assert caplog.records == []


def test_matched_and_mismatched_label_bookkeeping(built):
    """(6) Recompute every swap row's label relationship from the CSV.

    ``label_exact_match`` must agree with an independent vector comparison, ``label_overlap``
    with an independent intersection count, and a "mismatched" donor must never be an exact
    label-vector match.
    """
    labels = built.labels
    matched = [r for r in built.rows if r["condition"] == "label_matched_swap"]
    mismatched = [r for r in built.rows if r["condition"] == "label_mismatched_swap"]

    for r in matched + mismatched:
        t_vec = labels[r["target_id"]]
        s_vec = labels[r["cond_mask_source_id"]]
        assert r["label_overlap"] == int((t_vec * s_vec).sum())
        assert r["label_hamming"] == int((t_vec != s_vec).sum())
        assert r["source_labels"] is not None

    for r in matched:
        t_vec, s_vec = labels[r["target_id"]], labels[r["cond_mask_source_id"]]
        assert r["label_exact_match"] == bool((t_vec == s_vec).all())
        if not r["label_exact_match"]:
            assert r["label_hamming"] > 0
            assert "NO donor with the same label vector" in r["report_note"]

    for r in mismatched:
        t_vec, s_vec = labels[r["target_id"]], labels[r["cond_mask_source_id"]]
        assert not bool((t_vec == s_vec).all()), r["sample_id"]

    # gt / null carry no donor labels (schema: source_labels null for both).
    for r in built.rows:
        if r["condition"] == "gt":
            assert r["cond_mask_source_id"] == r["target_id"]
            assert r["source_labels"] is None and r["label_overlap"] is None
        if r["condition"] == "null":
            assert r["cond_mask_source_id"] is None
            assert r["source_labels"] is None


def test_null_condition_refused_without_learned_null_mask(built, tmp_path):
    """A checkpoint with no ``no_mask_embed`` (report2ct_wan_mask / text2ct_mask) cannot be used
    for ``condition=null`` — an untrained empty mask would measure OOD input, not "no mask"."""
    plain_ckpt = tmp_path / "wan_mask_no_null.ckpt"
    torch.save({"state_dict": {"unet.conv.weight": torch.zeros(2)}}, plain_ckpt)

    with pytest.raises(SystemExit) as exc:
        _build(tmp_path / "should_not_exist.jsonl", plain_ckpt)
    assert "no_mask_embed" in str(exc.value)
    assert not (tmp_path / "should_not_exist.jsonl").exists()

    # …and the same checkpoint is fine when `null` is not requested.
    out = tmp_path / "no_null.jsonl"
    argv = [
        "build_mask_intervention_manifest.py",
        "--n",
        "3",
        "--seed",
        "0",
        "--conditions",
        "gt",
        "label_mismatched_swap",
        "--out",
        str(out),
        "--ckpt",
        str(plain_ckpt),
        "--run-id",
        "test_run",
        "--cfg-text",
        "5.0",
        "--cfg-mask",
        "0.0",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        builder.main()
    finally:
        sys.argv = old
    assert len(out.read_text().splitlines()) == 6


def test_sample_id_is_path_safe_and_encodes_provenance(built):
    """sample_id is the generated file's stem, so it must survive ``_is_safe_id`` and carry the
    (condition, source, s_m, seed) tuple that distinguishes two rows of the same target."""
    for r in built.rows:
        sid = r["sample_id"]
        assert "/" not in sid and "\\" not in sid and ".." not in sid
        assert sid.startswith(f"{r['target_id']}__{r['condition']}__src-")
        assert sid.endswith("__sm-1.0__seed-0")


def test_donor_pool_is_restricted_to_available_mask_latents(built):
    """Every donor a row names must have a precomputed Wan mask latent — the generation side
    cannot condition on a mask that does not exist."""
    latent_dir = Path(builder.DEFAULT_MASK_LATENT_DIR)
    for r in _swaps(built.rows):
        path = latent_dir / f"{r['cond_mask_source_id']}_mask_emb.nii.gz"
        assert path.is_file(), path


def test_report_numbers(built, capsys):
    """Print the acceptance table's per-condition numbers (visible with ``pytest -s``)."""
    labels = built.labels
    matched = [r for r in built.rows if r["condition"] == "label_matched_swap"]
    mismatched = [r for r in built.rows if r["condition"] == "label_mismatched_swap"]
    n_exact = sum(1 for r in matched if r["label_exact_match"])
    overlaps = [r["label_overlap"] for r in mismatched]
    with capsys.disabled():
        print(
            f"\n  matched: {n_exact}/{len(matched)} exact, {len(matched) - n_exact} fallback"
            f" | mismatched overlap mean {np.mean(overlaps):.2f} max {max(overlaps)}"
        )
    assert len(labels) > 0
