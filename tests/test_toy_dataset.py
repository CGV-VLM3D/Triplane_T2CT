"""대표성 있는 CT-RATE toy 데이터셋 v2(scripts/make_toy_dataset.py) 테스트.

GPU·가중치 불필요. valid_v2가 valid_fixed 환자당 1스캔 census(1304)인지,
train 5000과 환자 단위로 분리되는지, latent 심볼릭 링크가 풀리는지, stats가 맞는지,
load_eval_cases가 frozen valid_v2를 쓰고 subsample이 결정적·중첩인지 확인한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.eval.ct_rate_cases import _DEFAULT_VALID_V2_IDS, load_eval_cases

_TOY = Path("/workspace/data/ctrate_toy_v2")
_TRAIN_IDS = _TOY / "train" / "ids.json"
_VALID_V2_IDS = _TOY / "valid_v2" / "ids.json"
_STATS = _TOY / "train" / "stats.json"
_CT_RATE_VALID = Path("/workspace/datasets/datasets/CT-RATE/dataset/valid_fixed")

_have_toy = _VALID_V2_IDS.is_file() and _TRAIN_IDS.is_file()
_needs_toy = pytest.mark.skipif(not _have_toy, reason="ctrate_toy_v2 not built")
_needs_ctrate = pytest.mark.skipif(
    not (_have_toy and _CT_RATE_VALID.is_dir()),
    reason="CT-RATE valid_fixed not mounted",
)


def _patient(scan_id: str) -> str:
    return re.match(r"((?:train|valid)_\d+)", scan_id).group(1)


@_needs_toy
def test_train_ids_count_and_disjoint_from_proxy() -> None:
    train = json.loads(_TRAIN_IDS.read_text())["ids"]
    proxy = json.loads(_VALID_V2_IDS.read_text())["ids"]
    assert len(train) == 5000
    assert {_patient(s) for s in train}.isdisjoint({_patient(s) for s in proxy})


@_needs_toy
def test_train_stats_matches_5000() -> None:
    stats = json.loads(_STATS.read_text())
    assert stats.get("num_volumes") == 5000
    assert len(stats.get("channel_mean", [])) == 4


@_needs_ctrate
def test_load_eval_cases_uses_frozen_valid_v2() -> None:
    cases = load_eval_cases()
    frozen = set(json.loads(_DEFAULT_VALID_V2_IDS.read_text())["ids"])
    assert len(cases) == 1304
    assert {c.scan_id for c in cases} == frozen
    assert all(c.findings.strip() or c.impression.strip() for c in cases)


@_needs_ctrate
def test_load_eval_cases_subsample_is_nested_and_deterministic() -> None:
    full = {c.scan_id for c in load_eval_cases()}
    s1 = {c.scan_id for c in load_eval_cases(n_samples=100)}
    s2 = {c.scan_id for c in load_eval_cases(n_samples=100)}
    assert len(s1) == 100
    assert s1 == s2, "subsample must be deterministic (seeded)"
    assert s1 <= full, "n_samples subset must be nested in the full frozen set"
