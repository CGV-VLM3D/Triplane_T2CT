"""summary.py: SUMMARY.md renders the surface-distance family (2026-08-01) alongside Dice, and
the new organ-generation-rate section is available for non-mask models — unlike ``_mask_section``,
which only renders when ``dice_to_input_mask`` is populated (mask-conditioned runs).

No prior test file existed for this module; these pin the sections most likely to silently drop
a metric family again (matches the ``METRIC_COLS``/``compare_runs.py`` hand-list incident this
same day, ``subgroup.py``'s module docstring).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.analysis.summary import write_summary


def _write_subgroup_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "label": "Cardiomegaly",
                "n": 10,
                "clip_t2i_mean": 60.0,
                "dice_to_gt_mask_mean": 0.85,
                "dice_to_input_mask_mean": 0.85,
                "hd95_mm_to_gt_mask_mean": 4.2,
                "hd95_mm_to_input_mask_mean": 4.2,
            }
        ]
    ).to_csv(path, index=False)


def test_subgroup_section_includes_hd95_columns(tmp_path: Path):
    analysis_dir = tmp_path / "analysis"
    _write_subgroup_csv(analysis_dir / "subgroup" / "per_abnormality.csv")

    summary_path = write_summary(tmp_path)
    text = summary_path.read_text()
    # headers drop the `_mean` suffix (the section note says the values are means)
    assert "hd95_mm_to_gt_mask" in text
    assert "hd95_mm_to_input_mask" in text
    assert "Values are means" in text
    assert "4.2000" in text


def test_mask_section_includes_hd95(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "dice_to_input_mask": 0.9,
                "dice_to_input_mask_lung": 0.95,
                "dice_to_gt_mask_lung": 0.95,
                "hd95_mm_to_input_mask_lung": 3.1,
                "hd95_mm_to_gt_mask_lung": 3.1,
            }
        ]
    )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    df.to_csv(analysis_dir / "per_sample.csv", index=False)

    summary_path = write_summary(tmp_path)
    text = summary_path.read_text()
    assert "Mask-following Dice + HD95" in text
    assert "hd95_mm_to_input_mask" in text
    assert "3.1000" in text


def test_organ_generation_section_renders_for_non_mask_model(tmp_path: Path):
    """A non-mask model's per_sample.csv has NO ``dice_to_input_mask`` (all-NaN, per
    ``persample.build_per_sample``'s ``is_mask_model=False`` path) — ``_mask_section`` must skip,
    but organ generation rate has nothing to do with input-mask conditioning and must still show."""
    df = pd.DataFrame(
        [
            {
                "dice_to_input_mask": float("nan"),
                "organ_generated_lung": 1,
                "organ_generated_heart": 1,
                "organ_generated_aorta": 0,
                "organ_generated_esophagus": 0,
            },
            {
                "dice_to_input_mask": float("nan"),
                "organ_generated_lung": 1,
                "organ_generated_heart": 0,
                "organ_generated_aorta": 0,
                "organ_generated_esophagus": 1,
            },
        ]
    )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    df.to_csv(analysis_dir / "per_sample.csv", index=False)

    summary_path = write_summary(tmp_path)
    text = summary_path.read_text()
    assert "Mask-following Dice" not in text  # gated section correctly absent
    assert "Organ generation rate" in text
    assert "| lung | 1.0000 |" in text
    assert "| aorta | 0.0000 |" in text


def test_per_abnormality_table_shows_every_label(tmp_path: Path):
    """All rows, not a top-6 cut (user request, 2026-08-02) — for every subgroup table, since
    the CSV they summarise is too wide to read directly."""
    analysis_dir = tmp_path / "analysis"
    rows = [
        {"label": f"label_{i:02d}", "n": 30 - i, "clip_t2i_mean": 60.0 + i}
        for i in range(18)
    ]
    (analysis_dir / "subgroup").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(
        analysis_dir / "subgroup" / "per_abnormality.csv", index=False
    )
    pd.DataFrame(
        [{"band": f"band_{i}", "n": 10 - i, "clip_t2i_mean": 55.0} for i in range(8)]
    ).to_csv(analysis_dir / "subgroup" / "label_burden.csv", index=False)

    text = write_summary(tmp_path).read_text()
    for i in range(18):
        assert f"| label_{i:02d} |" in text
    # no table is truncated any more — the burden table shows all 8 rows too
    assert "rows by n — see the CSV for all" not in text
    for i in range(8):
        assert f"| band_{i} |" in text


def test_setlevel_appendix_lists_every_axis(tmp_path: Path):
    """The 3-row Set-level headline stays, and every axis is rendered once, at the end."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "setlevel").mkdir(parents=True)
    rows = [
        {
            "axis": axis,
            "FID_2p5D_Avg": 40.0 + i,
            "real_n": 100 + i,
            "gen_n": 100 + i,
            "real_patients": 100 + i,
            "below_threshold": False,
        }
        for i, axis in enumerate(
            ["overall", "normal", "disease", "label:Cardiomegaly", "cluster:other"]
        )
    ]
    pd.DataFrame(rows).to_csv(
        analysis_dir / "setlevel" / "setlevel_fid_fvd.csv", index=False
    )

    text = write_summary(tmp_path).read_text()
    head, appendix = text.split("## Appendix — set-level")
    assert "label:Cardiomegaly" not in head  # headline stays overall/normal/disease
    assert "full table in the appendix" in head
    for axis in ("overall", "label:Cardiomegaly", "cluster:other"):
        assert f"| {axis} |" in appendix
    assert "| 100 |" in appendix  # counts stay integers, not 100.0000
    assert text.rstrip().endswith("|")  # appendix is last


def test_condition_fid_section_renders_when_scored(tmp_path: Path):
    """FID/FVD by condition only appears after scripts/score_condition_fid.py has run (writes
    condition_fid/condition_fid_fvd.csv); a run without it gets no such section."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    (tmp_path / "condition_fid").mkdir()
    pd.DataFrame(
        [
            {"condition": "gt", "n": 300, "FID_2p5D_Avg": 12.3, "FVD_CTCLIP": 0.2},
            {
                "condition": "label_mismatched_swap",
                "n": 300,
                "FID_2p5D_Avg": 18.7,
                "FVD_CTCLIP": 0.4,
            },
        ]
    ).to_csv(tmp_path / "condition_fid" / "condition_fid_fvd.csv", index=False)

    text = write_summary(tmp_path).read_text()
    assert "FID/FVD by condition" in text
    assert "| gt | 300 | 12.3000 | 0.2000 |" in text
    assert "| label_mismatched_swap | 300 | 18.7000 | 0.4000 |" in text

    plain = tmp_path / "plain"
    (plain / "analysis").mkdir(parents=True)
    assert "FID/FVD by condition" not in write_summary(plain).read_text()


def test_organ_generation_section_absent_without_the_column(tmp_path: Path):
    df = pd.DataFrame([{"clip_t2i": 55.0}])
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    df.to_csv(analysis_dir / "per_sample.csv", index=False)

    summary_path = write_summary(tmp_path)
    assert "Organ generation rate" not in summary_path.read_text()


def test_condition_section_renders_the_intervention_arms(tmp_path: Path):
    """A mask-intervention run gets a per-condition table; a plain run (one condition) does not."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    rows = [
        {
            "sample_id": f"t{t}__{cond}",
            "target_id": f"t{t}",
            "condition": cond,
            "clip_t2i": 60.0 + i,
            "dice_to_input_mask": 0.9 - 0.1 * i,
            "dice_to_gt_mask": 0.9,
            "hd95_mm_to_input_mask": 4.0 + i,
        }
        for t in range(2)
        for i, cond in enumerate(["gt", "label_matched_swap", "label_mismatched_swap"])
    ]
    pd.DataFrame(rows).to_csv(analysis_dir / "per_sample.csv", index=False)

    text = write_summary(tmp_path).read_text()
    assert "Mask intervention by condition" in text
    assert "6 generated volumes over 2 target scans" in text
    # canonical arm order, not alphabetical
    order = [
        text.index(f"| {c} |")
        for c in ("gt", "label_matched_swap", "label_mismatched_swap")
    ]
    assert order == sorted(order)
    assert "| gt | 2 |" in text

    plain = tmp_path / "plain"
    (plain / "analysis").mkdir(parents=True)
    pd.DataFrame(
        [{"sample_id": "s", "target_id": "s", "condition": "gt", "clip_t2i": 60.0}]
    ).to_csv(plain / "analysis" / "per_sample.csv", index=False)
    assert "Mask intervention by condition" not in write_summary(plain).read_text()
