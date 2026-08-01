"""``metrics.json`` is merged, not overwritten (2026-07-31).

``CTGenEvaluator.evaluate`` used to ``json.dump`` only the metrics enabled in THAT call, so
scoring one metric into a directory that already held a complete result silently dropped the
rest. That is the whole reason an FVD-only re-score had to be parked in its own ``fvd/``
subdirectory instead of landing beside the FID/CLIP it belongs with — see
``docs/ctgen_local_eval.md``.

The merge keeps every metric key **flat at the top level** (every aggregator reads them that
way) and records what each scoring pass added or replaced under ``_history``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.tasks.ctgen import _merge_metrics


def _read(out_dir: Path) -> dict:
    return json.loads((out_dir / "metrics.json").read_text())


def test_merge_preserves_keys_from_an_earlier_pass(tmp_path: Path) -> None:
    """A later FVD-only pass must not drop the CLIP/FID an earlier pass recorded."""
    _merge_metrics(tmp_path, {"CLIPScore": 45.4, "FID_2p5D_Avg": 1.39}, ["clip_score"])
    _merge_metrics(tmp_path, {"FVD_CTCLIP": 0.43}, ["fvd_ctclip"])

    got = _read(tmp_path)
    assert got["CLIPScore"] == 45.4
    assert got["FID_2p5D_Avg"] == 1.39
    assert got["FVD_CTCLIP"] == 0.43


def test_metric_keys_stay_flat_at_top_level(tmp_path: Path) -> None:
    """Aggregators do ``metrics.get("FID_2p5D_Avg")`` — nesting them would break every table."""
    _merge_metrics(
        tmp_path, {"FID_2p5D_Avg": 1.39, "fid_profile": "research"}, ["fid_2p5d"]
    )

    got = _read(tmp_path)
    assert got["FID_2p5D_Avg"] == 1.39
    assert got["fid_profile"] == "research"
    # Only the history is allowed to be non-scalar bookkeeping.
    assert [k for k in got if k.startswith("_")] == ["_history"]


def test_rescoring_replaces_the_value_and_records_the_old_one(tmp_path: Path) -> None:
    """Re-running a metric overwrites its value, but the previous number stays auditable."""
    _merge_metrics(tmp_path, {"FID_2p5D_Avg": 1.39}, ["fid_2p5d"])
    _merge_metrics(tmp_path, {"FID_2p5D_Avg": 1.42}, ["fid_2p5d"])

    got = _read(tmp_path)
    assert got["FID_2p5D_Avg"] == 1.42
    assert got["_history"][-1]["replaced"] == {"FID_2p5D_Avg": 1.39}
    assert got["_history"][-1]["added"] == []


def test_history_grows_one_entry_per_pass_with_a_timestamp(tmp_path: Path) -> None:
    """Each pass is one dated row naming the metric set it ran."""
    _merge_metrics(tmp_path, {"CLIPScore": 45.4}, ["clip_score"])
    _merge_metrics(tmp_path, {"FVD_CTCLIP": 0.43}, ["fvd_ctclip"])

    hist = _read(tmp_path)["_history"]
    assert len(hist) == 2
    assert hist[0]["metrics"] == ["clip_score"]
    assert hist[0]["added"] == ["CLIPScore"]
    assert hist[1]["metrics"] == ["fvd_ctclip"]
    # ISO-8601 with an offset, e.g. 2026-07-31T04:20:11+09:00
    assert hist[0]["at"][:4].isdigit() and hist[0]["at"][4] == "-"


def test_first_pass_records_no_replacements(tmp_path: Path) -> None:
    """Nothing pre-exists, so everything is 'added' and nothing is 'replaced'."""
    _merge_metrics(
        tmp_path, {"CLIPScore": 45.4, "FID_2p5D_Avg": 1.39}, ["clip_score", "fid_2p5d"]
    )

    hist = _read(tmp_path)["_history"]
    assert len(hist) == 1
    assert hist[0]["replaced"] == {}
    assert sorted(hist[0]["added"]) == ["CLIPScore", "FID_2p5D_Avg"]


def test_returns_the_full_merged_view_not_just_this_pass(tmp_path: Path) -> None:
    """The return value is what the caller reports/summarises, so it must be the whole picture."""
    _merge_metrics(tmp_path, {"CLIPScore": 45.4}, ["clip_score"])
    merged = _merge_metrics(tmp_path, {"FVD_CTCLIP": 0.43}, ["fvd_ctclip"])

    assert merged == {"CLIPScore": 45.4, "FVD_CTCLIP": 0.43}
    assert (
        "_history" not in merged
    )  # bookkeeping stays on disk, out of the returned metrics


def test_an_unreadable_existing_file_is_not_silently_dropped(tmp_path: Path) -> None:
    """A truncated metrics.json must fail loudly rather than be replaced by this pass alone."""
    (tmp_path / "metrics.json").write_text("{not json")
    try:
        _merge_metrics(tmp_path, {"CLIPScore": 45.4}, ["clip_score"])
    except json.JSONDecodeError:
        return
    raise AssertionError(
        "expected the corrupt metrics.json to raise, not be overwritten"
    )
