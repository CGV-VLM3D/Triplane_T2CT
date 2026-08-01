"""run_eval.py's multi-profile scoring pass: what gets recomputed, copied, and reported where.

``task.fid_profile`` defaults to BOTH families, which introduces two failure modes that a
green run would hide: paying for CLIP/FVD twice (identical numbers, ~24 min at n=300), and
letting one profile's FID leak into the other's folder or into the combined summary unlabelled
— the same ambiguity that cost two research FIDs on 2026-07-29.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, "/workspace/scripts")

from run_eval import _is_fid_key, _score_profiles, _write_summaries  # noqa: E402

from src.eval.tasks.ctgen import _merge_metrics  # noqa: E402

# Two incomparable FID scales, as on the real prediction set (docker 57.87 vs research 1.61).
_FID_VALUE = {"docker": 57.87, "research": 1.61, "docker_n300": 44.4}
_FVD = 0.269
_CLIP = 59.42


class _FakeEvaluator:
    """Stands in for CTGenEvaluator: records the metric flags it was handed, writes real files.

    Only ``evaluate`` is faked — the merge into metrics.json is the production
    ``_merge_metrics``, so the on-disk result is what a real run would produce.
    """

    calls: list[dict] = []

    def __init__(self, *, gt_dir, metrics, ctclip_ckpt, prompt_xlsx, fid_profile):
        self.metrics = metrics
        self.fid_profile = fid_profile
        _FakeEvaluator.calls.append({"profile": fid_profile, "metrics": dict(metrics)})

    def evaluate(self, pred_dir: Path, out_dir: Path) -> dict:
        results: dict = {}
        if self.metrics.get("fvd_ctclip"):
            results["FVD_CTCLIP"] = _FVD
        if self.metrics.get("clip_score"):
            results["CLIPScore_T2I"] = _CLIP
        if self.metrics.get("fid_2p5d"):
            results["FID_2p5D_Avg"] = _FID_VALUE[self.fid_profile]
            results["fid_profile"] = self.fid_profile
            results["fid_num_images"] = 100
        return _merge_metrics(Path(out_dir), results, ["fake"])


@pytest.fixture
def scored(tmp_path: Path, monkeypatch):
    """Score tmp_path with [docker, research] through the fake evaluator."""
    _FakeEvaluator.calls = []
    monkeypatch.setattr("src.eval.tasks.ctgen.CTGenEvaluator", _FakeEvaluator)
    profiles = ["docker", "research"]
    dirs = [tmp_path / f"fid_{p}" for p in profiles]
    for d in dirs:
        d.mkdir()
    result = _score_profiles(
        fid_profiles=profiles,
        score_dirs=dirs,
        pred_dir=tmp_path / "predictions",
        gt_dir=tmp_path / "gt",
        metrics_cfg={
            "fvd": False,
            "fvd_ctclip": True,
            "clip_score": True,
            "fid_2p5d": True,
        },
        ctclip_ckpt=None,
        prompt_xlsx=None,
        clip_agg=None,
    )
    return result, tmp_path, dirs


def test_profile_independent_metrics_run_only_once(scored):
    """CLIP/FVD are ~24 min at n=300 and do not depend on the FID feature network, so only the
    first profile's pass may compute them."""
    _, _, _ = scored
    first, second = _FakeEvaluator.calls
    assert first["profile"] == "docker" and second["profile"] == "research"
    for key in ("fvd", "fvd_ctclip", "clip_score"):
        assert second["metrics"][key] is False, key
    assert first["metrics"]["fvd_ctclip"] is True
    assert first["metrics"]["clip_score"] is True
    # ...but every profile still scores FID — that is the whole point of the second pass.
    assert first["metrics"]["fid_2p5d"] and second["metrics"]["fid_2p5d"]


def test_every_profile_folder_stays_self_contained(scored):
    """Aggregators read one fid_<profile>/metrics.json in isolation, so the copied CLIP/FVD must
    be physically present in each — identical values, not a cross-reference."""
    _, _, dirs = scored
    for d in dirs:
        recorded = json.loads((d / "metrics.json").read_text())
        assert recorded["FVD_CTCLIP"] == _FVD
        assert recorded["CLIPScore_T2I"] == _CLIP


def test_no_profile_receives_another_profiles_fid(scored):
    """The reason the folders are separate in the first place."""
    _, _, dirs = scored
    for profile, d in zip(("docker", "research"), dirs):
        recorded = json.loads((d / "metrics.json").read_text())
        assert recorded["FID_2p5D_Avg"] == _FID_VALUE[profile]
        assert recorded["fid_profile"] == profile


def test_shared_and_fid_keys_are_partitioned(scored):
    result, _, _ = scored
    assert not any(_is_fid_key(k) for k in result.shared)
    for profile, block in result.fid_by_profile.items():
        assert all(_is_fid_key(k) for k in block)
        assert block["FID_2p5D_Avg"] == _FID_VALUE[profile]


def test_persample_clip_is_copied_onward_like_any_shared_metric(
    tmp_path: Path, monkeypatch
):
    """When per-sample CLIP supplies the aggregate, the SECOND profile must get it too —
    otherwise a run with task.metrics.per_sample=true silently yields a CLIP-less folder."""
    _FakeEvaluator.calls = []
    monkeypatch.setattr("src.eval.tasks.ctgen.CTGenEvaluator", _FakeEvaluator)
    profiles = ["docker", "research"]
    dirs = [tmp_path / f"fid_{p}" for p in profiles]
    for d in dirs:
        d.mkdir()

    _score_profiles(
        fid_profiles=profiles,
        score_dirs=dirs,
        pred_dir=tmp_path / "predictions",
        gt_dir=tmp_path / "gt",
        # per-sample routing turns the evaluator's own CLIP off and passes clip_agg instead
        metrics_cfg={
            "fvd": False,
            "fvd_ctclip": True,
            "clip_score": False,
            "fid_2p5d": True,
        },
        ctclip_ckpt=None,
        prompt_xlsx=None,
        clip_agg={"CLIPScore_T2I_persample": 58.0},
    )
    for d in dirs:
        recorded = json.loads((d / "metrics.json").read_text())
        assert recorded["CLIPScore_T2I_persample"] == 58.0


def test_summaries_keep_each_fid_labelled(scored):
    """Per-profile summary.json keeps its original flat schema; the combined one nests FID by
    profile so no number in it is unlabelled."""
    result, out_dir, dirs = scored
    _write_summaries(out_dir, dirs, "fake_model", result)

    for profile, d in zip(("docker", "research"), dirs):
        per_profile = json.loads((d / "summary.json").read_text())
        assert per_profile["metrics"]["FID_2p5D_Avg"] == _FID_VALUE[profile]
        assert "fid" not in per_profile  # the flat schema every older reader expects

    combined = json.loads((out_dir / "summary.json").read_text())
    assert combined["fid_profiles"] == ["docker", "research"]
    assert combined["fid"]["docker"]["FID_2p5D_Avg"] == _FID_VALUE["docker"]
    assert combined["fid"]["research"]["FID_2p5D_Avg"] == _FID_VALUE["research"]
    # The shared block carries the once-computed metrics and NO bare FID key.
    assert combined["metrics"]["CLIPScore_T2I"] == _CLIP
    assert not any(_is_fid_key(k) for k in combined["metrics"])


def test_single_profile_run_is_unchanged(tmp_path: Path, monkeypatch):
    """`task.fid_profile=research` must still behave exactly as it did before the loop existed:
    one folder, its own metrics, no copy pass."""
    _FakeEvaluator.calls = []
    monkeypatch.setattr("src.eval.tasks.ctgen.CTGenEvaluator", _FakeEvaluator)
    d = tmp_path / "fid_research"
    d.mkdir()

    result = _score_profiles(
        fid_profiles=["research"],
        score_dirs=[d],
        pred_dir=tmp_path / "predictions",
        gt_dir=tmp_path / "gt",
        metrics_cfg={
            "fvd": False,
            "fvd_ctclip": True,
            "clip_score": True,
            "fid_2p5d": True,
        },
        ctclip_ckpt=None,
        prompt_xlsx=None,
        clip_agg=None,
    )
    assert len(_FakeEvaluator.calls) == 1
    recorded = json.loads((d / "metrics.json").read_text())
    assert recorded["FID_2p5D_Avg"] == _FID_VALUE["research"]
    assert [h["metrics"] for h in recorded["_history"]] == [["fake"]]  # no copy row
    assert list(result.fid_by_profile) == ["research"]
