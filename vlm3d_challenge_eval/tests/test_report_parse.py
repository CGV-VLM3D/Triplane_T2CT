"""Round-trip test for the challenge prompt parser.

Rebuilds the challenge ``report`` string from the CT-RATE columns the model was trained on, parses
it back, and asserts we recover the originals. Runs over every row of ``validation_reports.csv``
(3039 — a superset of the 1304 valid_v2 cases), so a regression in the split rule cannot hide in a
sampled subset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_parse import report_markers, split_report  # noqa: E402

_REPORTS_CSV = Path(
    "/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv"
)

# The preamble the challenge puts before "Findings:" varies (age/sex, and it contains a colon).
# The parser must ignore it entirely, so exercise several shapes including "no preamble at all".
_PREAMBLES = ("58-year-old male: ", "7-year-old female: ", "")


@pytest.fixture(scope="module")
def reports() -> pd.DataFrame:
    if not _REPORTS_CSV.is_file():
        pytest.skip(f"CT-RATE reports not available at {_REPORTS_CSV}")
    return pd.read_csv(_REPORTS_CSV)


def test_roundtrip_all_validation_reports(reports: pd.DataFrame) -> None:
    """Every validation report survives challenge-format wrap → split_report unchanged."""
    findings = reports["Findings_EN"].fillna("").astype(str)
    impressions = reports["Impressions_EN"].fillna("").astype(str)

    for idx, (f, i) in enumerate(zip(findings, impressions)):
        preamble = _PREAMBLES[idx % len(_PREAMBLES)]
        prompt = f"{preamble}Findings: {f} Impression: {i}"
        got_f, got_i = split_report(prompt)
        assert got_f == f.strip(), f"row {idx}: findings mismatch"
        assert got_i == i.strip(), f"row {idx}: impression mismatch"

    assert len(findings) > 3000, "expected the full validation split"


def test_markers_absent_from_report_bodies(reports: pd.DataFrame) -> None:
    """The split is unambiguous only because the markers never occur inside a report body."""
    for col in ("Findings_EN", "Impressions_EN"):
        body = reports[col].fillna("").astype(str)
        assert not body.str.contains("Findings:", case=False).any()
        assert not body.str.contains("Impression:", case=False).any()


def test_report_markers_classification() -> None:
    """The parse diagnostic must separate 'both' from the silently-degrading 'findings_only'."""
    assert report_markers("58-year-old male: Findings: a. Impression: b.") == "both"
    assert report_markers("Findings: a.") == "findings_only"
    assert report_markers("Impression: b.") == "impression_only"
    assert report_markers("sample_report") == "none"


def test_report_markers_agrees_with_split(reports: pd.DataFrame) -> None:
    """Every challenge-formatted validation report classifies as 'both' and splits non-empty."""
    for f, i in zip(
        reports["Findings_EN"].fillna("").astype(str).head(200),
        reports["Impressions_EN"].fillna("").astype(str).head(200),
    ):
        prompt = f"58-year-old male: Findings: {f} Impression: {i}"
        assert report_markers(prompt) == "both"
        got_f, got_i = split_report(prompt)
        assert got_f and got_i


def test_placeholder_prompt_does_not_raise() -> None:
    """`forithmus generate` writes report="sample_report"; the dry-run must not crash."""
    assert split_report("sample_report") == ("sample_report", "")


def test_missing_impression_marker() -> None:
    """A findings-only prompt yields an empty impression rather than swallowing the text."""
    assert split_report("Findings: lungs are clear.") == ("lungs are clear.", "")


def test_case_insensitive_markers() -> None:
    """Upper-case marker variants in the hidden test set still split."""
    assert split_report("FINDINGS: a. IMPRESSION: b.") == ("a.", "b.")


def test_leading_whitespace_is_tokenizer_invariant() -> None:
    """We hand the encoder stripped text while training saw the raw (often space-led) column.

    Harmless only if the tokenizer ignores leading/trailing whitespace — assert that directly
    instead of assuming it.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )
    raw = "  Trachea, both main bronchi are open.  "
    assert tok(raw)["input_ids"] == tok(raw.strip())["input_ids"]
