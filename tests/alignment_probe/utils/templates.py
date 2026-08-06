"""
각 인코더는 학습되었던 텍스트 형식 대로 입력 받아야함('배포 상태의 alignment'를
재기 위함). 텍스트 조립은 여기서만 하고, 임베딩 추출(Unit 1b)은 이 함수들만 호출한다.

출처 인용(각 인코더 upstream):
  T5/GenerateCT  : videotextdataset.py:71-84
                   age = PatientAge[:-1].zfill(3)("036Y"→"036"), sex 그대로("M"/"F"),
                   f'{age} years old {sex}: {impression}' 후 " ' ( ) 제거.
  CT-CLIP        : 전체 report(findings + impression)
  Text2CT CLIP3D : Finding + Impression (ckpt CLIP3D_Finding_Impression_30ep).
  Report2CT      : findings / impression **분리**(3-인코더 concat은 encoder가 처리).
  fVLM           : per-organ build_organ_text(fvlm_report.py) — 분해 JSON 필요.
"""

from __future__ import annotations

from src.data.fvlm_organ_report import build_organ_text, scan_id_from_volume_name
from tests.alignment_probe.utils.cases import AlignmentCase

_SUFFIX = ".nii.gz"


def _strip_punct(text: str) -> str:
    """videotextdataset.py:130-133 — 따옴표/괄호 제거(upstream과 동일 순서)."""
    for ch in ('"', "'", "(", ")"):
        text = text.replace(ch, "")
    return text


def t5_text(case: AlignmentCase) -> str:
    """GenerateCT T5 입력(impression + 인구통계). videotextdataset.py:71-84 충실."""
    age = case.age[:-1].zfill(3) if case.age else ""  # "036Y" → "036"
    return _strip_punct(f"{age} years old {case.sex}: {case.impression}")


def ctclip_text(case: AlignmentCase) -> str:
    """CT-CLIP 입력 = 전체 report(findings + impression)."""
    return f"{case.findings} {case.impression}".strip()


def text2ct_text(case: AlignmentCase) -> str:
    """Text2CT CLIP3D 입력 = Finding + Impression."""
    return f"{case.findings} {case.impression}".strip()


def report2ct_texts(case: AlignmentCase) -> tuple[str, str]:
    """Report2CT 입력 = (findings, impression) 분리. 각각 encoder가 2560-d로 인코딩."""
    return case.findings, case.impression


def fvlm_organ_texts(
    case: AlignmentCase, desc_info: dict, conc_info: dict
) -> dict[str, str]:
    """fVLM 입력 = per-organ 텍스트 {organ: text}. 분해 JSON(desc/conc) 필요.

    valid_v2/train 분해 산출물의 키는 재구성 인덱스를 뗀 scan-study id(예: 'valid_1000_a').
    """
    scan_key = scan_id_from_volume_name(f"{case.scan_id}{_SUFFIX}")
    return build_organ_text(desc_info, conc_info, scan_key)
