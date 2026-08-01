"""subgroup_config.py: typo detection + valid load."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.eval.analysis import subgroup_config


def test_load_default_config():
    cfg = subgroup_config.load("/workspace/configs/eval/subgroup/default.yaml")
    assert cfg.subgroup_fid_small_n == 100
    assert "all_zero" in cfg.label_burden_bands
    assert cfg.label_burden_bands["all_zero"] == (0, 0)
    assert "lung_parenchyma_airway" in cfg.organ_clusters


def test_typo_label_raises(tmp_path: Path):
    bad_yaml = {
        "label_burden_bands": {"all_zero": [0, 0]},
        "organ_clusters": {
            "bad_cluster": ["Cardiomegally"]
        },  # typo: should be "Cardiomegaly"
        "subgroup_fid_small_n": 100,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(bad_yaml))
    with pytest.raises(ValueError, match="Unknown label"):
        subgroup_config.load(path)


def test_correct_labels_load_fine(tmp_path: Path):
    good_yaml = {
        "label_burden_bands": {"all_zero": [0, 0], "rest": [1, 18]},
        "organ_clusters": {"cluster_a": ["Cardiomegaly", "Emphysema"]},
        "subgroup_fid_small_n": 50,
    }
    path = tmp_path / "good.yaml"
    path.write_text(yaml.dump(good_yaml))
    cfg = subgroup_config.load(path)
    assert cfg.organ_clusters["cluster_a"] == ["Cardiomegaly", "Emphysema"]
    assert cfg.subgroup_fid_small_n == 50
