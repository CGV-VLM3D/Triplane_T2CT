"""qc_figures.py: diverse-abnormality case selection, no-flip (is_lps=True) orientation
treatment, and end-to-end figure generation on tiny synthetic volumes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.eval.analysis.labels import LABEL_NAMES
from src.eval.analysis.qc_figures import _load_lps, _select_cases, run
from tests.eval_analysis._helpers import write_mha


def test_load_lps_applies_no_flip(tmp_path: Path):
    """is_lps=True means NO in-plane flip — the loaded array must equal sitk's raw array."""
    arr = np.zeros((3, 4, 5), dtype=np.int16)
    arr[0, 0, 0] = 500  # asymmetric marker voxel
    path = write_mha(tmp_path / "vol.mha", arr)
    loaded, spacing = _load_lps(path)
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert np.array_equal(loaded, raw), (
        "qc_figures must not flip — GT/pred use is_lps=True"
    )


def test_select_cases_spans_distinct_abnormalities():
    label_map = {}
    # scan i is positive ONLY for LABEL_NAMES[i] (i < 5), plus one all-zero normal scan
    for i in range(5):
        vec = np.zeros(18, dtype=np.int64)
        vec[i] = 1
        label_map[f"scan{i}"] = vec
    label_map["scan_normal"] = np.zeros(18, dtype=np.int64)

    scan_ids = list(label_map.keys())
    picked = _select_cases(scan_ids, label_map, n=4)
    assert len(picked) == 4
    labels_picked = [label for _, label in picked]
    assert len(set(labels_picked)) == len(labels_picked), "no duplicate showcase labels"


def test_select_cases_fewer_available_than_n_does_not_duplicate():
    label_map = {"scan0": np.zeros(18, dtype=np.int64)}  # only 1 scan available
    picked = _select_cases(["scan0"], label_map, n=5)
    assert len(picked) == 1  # not padded with duplicates


def test_run_end_to_end_synthetic(tmp_path: Path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    label_map = {}
    scan_ids = []
    for i, name in enumerate(["Cardiomegaly", "Lung nodule", "Pleural effusion"]):
        sid = f"scan{i}"
        scan_ids.append(sid)
        vec = np.zeros(18, dtype=np.int64)
        vec[LABEL_NAMES.index(name)] = 1
        label_map[sid] = vec
        arr = np.full((6, 6, 6), -1000, dtype=np.int16)
        arr[2:4, 2:4, 2:4] = (
            40  # a soft-tissue-ish blob so body_centroid isn't degenerate
        )
        write_mha(pred_dir / f"{sid}.mha", arr)
        write_mha(gt_dir / f"{sid}.mha", arr)

    out_dir = tmp_path / "figures"
    written = run(pred_dir, gt_dir, label_map, out_dir, model_name="test_model", n=3)
    assert len(written) == 3
    for p in written:
        assert p.is_file() and p.stat().st_size > 0
    cases = json.loads((out_dir / "cases.json").read_text())
    assert len(cases) == 3
