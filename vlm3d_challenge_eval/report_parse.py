"""Split a VLM3D ctgen challenge prompt into the (findings, impression) pair our model expects.

The challenge hands us one flat ``report`` string per case
(``.forithmus/config.json`` → ``data_schema.input.sample``)::

    "58-year-old male: Findings: <findings text> Impression: <impression text>"

but ``report2ct_wan`` conditions on findings and impression *separately* — the sampler builds a
``(1, 2, 2560)`` context from ``EvalCase.findings`` / ``EvalCase.impression``
(`src/eval/samplers/report2ct.py::_encode_text`), and those two fields are fed the raw CT-RATE
``Findings_EN`` / ``Impressions_EN`` columns, which carry **no** ``"Findings:"`` prefix and no
age/sex preamble. So we strip the markers and drop the preamble to land back in the training
distribution.

Splitting is unambiguous: neither marker occurs inside any report body (verified over all 3039
rows of ``validation_reports.csv`` — 0 hits for both ``"Findings:"`` and ``"Impression:"``).
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Case-insensitive so a "FINDINGS:" / "IMPRESSION:" variant in the hidden test set still splits.
_FINDINGS = re.compile(r"\bFindings\s*:", re.IGNORECASE)
_IMPRESSION = re.compile(r"\bImpression[s]?\s*:", re.IGNORECASE)


def report_markers(report: str) -> str:
    """Classify which section markers a prompt carries, without exposing its text.

    Returns one of ``"both"``, ``"findings_only"``, ``"impression_only"``, ``"none"``, for the
    aggregate parse diagnostic in ``process.py``. Prompt text is never logged.
    """
    text = report.strip()
    m_f = _FINDINGS.search(text)
    m_i = _IMPRESSION.search(text, m_f.end() if m_f else 0)
    if m_f and m_i:
        return "both"
    if m_f:
        return "findings_only"
    return "impression_only" if m_i else "none"


def split_report(report: str) -> tuple[str, str]:
    """Split a challenge prompt into ``(findings, impression)``.

    Args:
        report: the challenge ``report`` string for one case.

    Returns:
        ``(findings, impression)``, both stripped. A missing ``Impression:`` marker yields an
        empty impression; a report with neither marker (e.g. the ``"sample_report"`` placeholder
        that ``forithmus generate`` writes) falls back to ``(whole_text, "")`` with a warning
        rather than raising — the schema dry-run must not crash.
    """
    text = report.strip()

    m_f = _FINDINGS.search(text)
    m_i = _IMPRESSION.search(text, m_f.end() if m_f else 0)

    if m_f is None and m_i is None:
        log.warning(
            "report has no Findings:/Impression: marker; using whole text as findings"
        )
        return text, ""

    # findings = between the Findings: marker (or the very start) and the Impression: marker.
    start = m_f.end() if m_f else 0
    end = m_i.start() if m_i else len(text)
    findings = text[start:end].strip()
    impression = text[m_i.end() :].strip() if m_i else ""
    return findings, impression
