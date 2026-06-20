"""src/data/fvlm_organ_report.py(fVLM 리포트 장기별 분해 라이브러리) 테스트.

GPU·가중치 불필요. 장기별 텍스트 조립 규칙(마침표·conc+desc 순서·따옴표 제거),
비정상 플래그, 누락 장기 템플릿 채움이 upstream 루프와 동일한지, 그리고 Qwen 분해기
순수 로직(모의 백엔드)을 검증한다. author 분해 JSON이 없으면 해당 테스트는 skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.fvlm_organ_report import (
    DEFAULT_REPORT_ROOT,
    ORGANS,
    build_organ_text,
    build_split_organ_texts,
    normal_organ_template,
    fill_missing_with_template,
    load_decomposed_report,
    organ_abnormal_flags,
    scan_id_from_volume_name,
    scan_ids_from_split,
)

_DESC = DEFAULT_REPORT_ROOT / "desc_info.json"
_CONC = DEFAULT_REPORT_ROOT / "conc_info.json"
_HAVE_AUTHOR_JSON = _DESC.is_file() and _CONC.is_file()
_needs_author = pytest.mark.skipif(
    not _HAVE_AUTHOR_JSON, reason="author decomposed_report JSON not present"
)


# --- upstream reference (INLINED, not imported) ----------------------------
# Verbatim copy of the dict-build loop from
# third_party/fvlm/lavis/datasets/datasets/caption_datasets.py:68-91. Importing
# the upstream module would drag in monai/lavis; copying it makes test 4 a
# drift guard against our byte-for-byte mirror in build_organ_text.
def _upstream_build_organ_text(desc_info, conc_info, scan_id, organs):
    all_info = {}
    for organ in organs:
        desc = ""
        if organ in desc_info.get(scan_id, {}):
            desc += desc_info[scan_id][organ]
            if not desc.endswith("."):
                desc += "."
        conc = ""
        if organ in conc_info.get(scan_id, {}):
            conc += conc_info[scan_id][organ]
            if not conc.endswith("."):
                conc += "."
        if not len(conc):
            conc = f"{organ} shows no significant abnormalities."
        input_text = conc + desc
        input_text = input_text.replace('"', "")
        input_text = input_text.replace("'", "")
        input_text = input_text.replace("(", "")
        input_text = input_text.replace(")", "")
        all_info[organ] = input_text
    return all_info


# --- Test 1: author JSON loads ---------------------------------------------
@_needs_author
def test_author_json_loads():
    desc, conc = load_decomposed_report()
    assert len(desc) > 24000 and len(conc) > 24000
    assert set(desc["train_1_a"]) >= {"Findings", *ORGANS}
    # conc_info is sparse — train_1_a impression mentions only the lung.
    assert "Conclusion" in conc["train_1_a"]


# --- Test 2: build_organ_text semantics ------------------------------------
@_needs_author
def test_build_organ_text_author_scan():
    desc, conc = load_decomposed_report()
    t = build_organ_text(desc, conc, "train_1_a")
    assert set(t) == set(ORGANS)
    # heart absent from impression -> starts with the default "normal" template.
    assert t["heart"].startswith(normal_organ_template("heart"))
    # lung present in impression -> NOT the template, and impression text comes
    # before the findings text (conc + desc order, L84).
    assert not t["lung"].startswith(normal_organ_template("lung"))
    # all quotes/parens stripped (L86-89).
    for organ in ORGANS:
        assert all(ch not in t[organ] for ch in "\"'()")


def test_build_organ_text_trailing_period_and_order():
    # Synthetic dicts exercise the trailing-period rule (L72-73 / L78-79) and
    # the conc+desc concat order (L84) without needing the author files.
    desc = {"s": {"lung": "findings text no period"}}
    conc = {"s": {"lung": "impression text"}}
    t = build_organ_text(desc, conc, "s", organs=("lung",))
    # impression first (gets a period), then findings (gets a period).
    assert t["lung"] == "impression text.findings text no period."


def test_build_organ_text_impression_missing_uses_template_but_keeps_findings():
    desc = {"s": {"heart": "cardiomegaly present"}}
    conc: dict = {"s": {}}  # heart not in impression
    t = build_organ_text(desc, conc, "s", organs=("heart",))
    assert (
        t["heart"] == "heart shows no significant abnormalities.cardiomegaly present."
    )
    # impression-absent -> normal, even though findings describes it (FNR rule).
    assert organ_abnormal_flags(t, organs=("heart",))["heart"] is False


# --- Test 3: abnormal flags + fill_missing ---------------------------------
def test_organ_abnormal_flags():
    organ_text = {
        "lung": "lung shows no significant abnormalities.",
        "heart": "heart is enlarged.",
        "esophagus": "esophagus shows no significant abnormalities.",
        "aorta": "aortic calcification.",
    }
    flags = organ_abnormal_flags(organ_text)
    assert flags == {
        "lung": False,
        "heart": True,
        "esophagus": False,
        "aorta": True,
    }


def test_fill_missing_with_template():
    filled = fill_missing_with_template({"lung": "lung opacity."})
    assert filled["lung"] == "lung opacity."
    for organ in ("heart", "esophagus", "aorta"):
        assert filled[organ] == normal_organ_template(organ)


# --- Test 4: golden equivalence vs inlined upstream loop -------------------
@_needs_author
def test_golden_equivalence_against_upstream():
    desc, conc = load_decomposed_report()
    sids = scan_ids_from_split("valid")[:20]
    for sid in sids:
        assert build_organ_text(desc, conc, sid) == _upstream_build_organ_text(
            desc, conc, sid, ORGANS
        )


# --- Test 5: Qwen-schema drop-in (author-path == Qwen-path) ----------------
def test_qwen_schema_dropin(tmp_path: Path):
    # A hand-written valid_fixed-style output with only a subset of organs per
    # section must load through the SAME builder and yield all 4 organs with
    # the template synthesized for the missing ones.
    desc_info = {"valid_1_a": {"Findings": "...", "lung": "linear atelectasis"}}
    conc_info = {"valid_1_a": {"Conclusion": "...", "lung": "atelectasis"}}
    (tmp_path / "desc_info.json").write_text(json.dumps(desc_info))
    (tmp_path / "conc_info.json").write_text(json.dumps(conc_info))

    desc, conc = load_decomposed_report(
        tmp_path / "desc_info.json", tmp_path / "conc_info.json"
    )
    t = build_organ_text(desc, conc, "valid_1_a")
    assert set(t) == set(ORGANS)
    assert not t["lung"].startswith(normal_organ_template("lung"))
    for organ in ("heart", "esophagus", "aorta"):
        assert t[organ] == normal_organ_template(organ)
    # whole-split helper round-trips on a 1-scan custom split file too.
    split_json = tmp_path / "split.json"
    split_json.write_text(
        json.dumps(
            {
                "valid": [
                    {
                        "patient_id": "valid_1",
                        "path": "valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
                    }
                ]
            }
        )
    )
    m = build_split_organ_texts("valid", desc, conc, split_json=split_json)
    assert list(m) == ["valid_1_a"] and set(m["valid_1_a"]) == set(ORGANS)


# --- scan-id helpers --------------------------------------------------------
def test_scan_id_from_volume_name():
    assert scan_id_from_volume_name("valid_1_a_1.nii.gz") == "valid_1_a"
    assert scan_id_from_volume_name("train_13572_a_2.nii.gz") == "train_13572_a"
    with pytest.raises(ValueError):
        scan_id_from_volume_name("weird.nii.gz")


@_needs_author
def test_split_valid_fully_covered_by_author():
    # The decisive scoping fact: split.json valid (1000) is entirely inside the
    # author decomposition (it is sampled from CT-RATE train_fixed).
    desc, conc = load_decomposed_report()
    sids = scan_ids_from_split("valid")
    assert len(sids) == 1000
    assert all(sid in desc and sid in conc for sid in sids)


# ---------------------------------------------------------------------------
# Tests 6-7: the Qwen decomposer (scripts/decompose_reports_fvlm.py) pure logic.
# No torch/network — exercised through a mock chat backend.
# ---------------------------------------------------------------------------
import re  # noqa: E402

from scripts.decompose_reports_fvlm import (  # noqa: E402
    decompose_scan,
    mentioned,
    substring_mentioned,
)


class _FakeBackend:
    """QwenChat mock: 'Yes' for organs in ``yes`` on mention prompts; canned
    '{anatomy}: ...' on extraction prompts. Parses the anatomy from the prompt
    (Fig 6/7 embed it as '(anatomy)')."""

    def __init__(self, yes):
        self.yes = set(yes)

    def generate(self, prompts, max_new_tokens=256):
        out = []
        for p in prompts:
            anat = re.search(r"\(([a-z]+)\)", p).group(1)
            if 'Yes" or "No"' in p:  # Fig 6 mention prompt
                out.append("Yes" if anat in self.yes else "No")
            else:  # Fig 7 extraction prompt
                out.append(f"{anat}: fake finding for {anat}")
        return out


class _NeverYes:
    def generate(self, prompts, max_new_tokens=256):
        return ["No"] * len(prompts)


def test_decompose_scan_mock_backend_schema():
    # Section deliberately has NO organ substring keyword, so the LLM decision
    # (the mock) fully controls which organs are "mentioned".
    section = "A focal mass lesion is noted."
    findings_organs, impression_organs = decompose_scan(
        section, section, _FakeBackend({"lung", "heart"})
    )
    assert set(findings_organs) == {"lung", "heart"}
    # '{anatomy}:' prefix stripped by normalize_extraction.
    assert findings_organs["lung"] == "fake finding for lung"
    assert "esophagus" not in findings_organs and "aorta" not in findings_organs

    # Drop-in proof: assemble into the author schema, load through build_organ_text.
    desc_info = {"valid_9_a": {"Findings": section, **findings_organs}}
    conc_info = {"valid_9_a": {"Conclusion": section, **impression_organs}}
    t = build_organ_text(desc_info, conc_info, "valid_9_a")
    assert set(t) == set(ORGANS)
    # esophagus/aorta absent from both -> default template.
    assert t["esophagus"] == normal_organ_template("esophagus")


def test_mention_substring_fallback():
    # substring short-circuits even when the LLM always says "No".
    assert mentioned("aortic arch calcification", "aorta", _NeverYes()) is True
    assert mentioned("liver is normal", "lung", _NeverYes()) is False
    assert substring_mentioned("peribronchial thickening", "lung") is True
    assert substring_mentioned("normal study", "heart") is False
