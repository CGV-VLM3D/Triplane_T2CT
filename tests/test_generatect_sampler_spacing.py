"""GenerateCTSampler 저장 spacing 테스트.

eval 샘플러가 GenerateCT의 truthful native spacing (0.75, 0.75, 1.5) mm를 찍는지,
docker-parity 오버라이드(1,1,1)도 동작하는지 확인한다. 모델은 lazy build라
더미 볼륨에 _save_mha만 직접 호출(GPU·가중치 불필요).
"""

from __future__ import annotations

import SimpleITK as sitk
import torch

from src.eval.samplers.base import EvalCase
from src.eval.samplers.generatect import GenerateCTSampler


def test_prompt_matches_training_template() -> None:
    """Default prompt reproduces videotextdataset.py:84 — '{age} years old {sex}: {impr}'.

    GenerateCT was trained on impression only with an age/sex prefix; findings must NOT
    appear, and the upstream quote/paren cleaning (videotextdataset.py:130-133) is applied.
    """
    sampler = GenerateCTSampler()
    case = EvalCase(
        scan_id="valid_1_a_1",
        findings="Trachea patent. (incidental)",
        impression='Mild cardiomegaly ("borderline").',
        age="036Y",
        sex="M",
    )
    prompt = sampler._build_prompt(case)
    assert prompt == "36 years old male: Mild cardiomegaly borderline."
    assert "Trachea" not in prompt  # findings excluded
    assert '"' not in prompt and "(" not in prompt  # cleaned


def test_prompt_female_and_missing_metadata_uses_none_placeholder() -> None:
    """F->female; missing metadata reproduces training's 'None years old None: ...' format.

    Never falls back to findings under template mode (that's the off-distribution format
    we're fixing). videotextdataset.py:74,78 set age/sex to 'None' on a parse failure.
    """
    sampler = GenerateCTSampler()
    f = EvalCase("s", "F.", "Effusion.", age="042Y", sex="F")
    assert sampler._build_prompt(f) == "42 years old female: Effusion."

    nometa = EvalCase("s", "Findings text.", "Impression text.", age="", sex="")
    assert sampler._build_prompt(nometa) == "None years old None: Impression text."
    assert "Findings" not in sampler._build_prompt(nometa)


def test_prompt_docker_parity_uses_raw_report() -> None:
    """use_metadata_template=False reproduces the docker's raw findings+impression."""
    sampler = GenerateCTSampler(use_metadata_template=False)
    case = EvalCase("s", "Findings text.", "Impression text.", age="036Y", sex="M")
    assert sampler._build_prompt(case) == "Findings text. Impression text."


def test_save_mha_stamps_truthful_native_spacing(tmp_path) -> None:
    """Default: truthful native (0.75, 0.75, 1.5) in ITK (x, y, z) — 1.5 on the slice axis."""
    sampler = GenerateCTSampler()
    vol = torch.zeros(1, 4, 8, 8)  # (1, D, H, W)
    out = tmp_path / "g.mha"
    sampler._save_mha(vol, out)
    sp = sitk.ReadImage(str(out)).GetSpacing()
    assert tuple(round(s, 4) for s in sp) == (0.75, 0.75, 1.5)


def test_docker_parity_spacing_override(tmp_path) -> None:
    """`final_spacing_mm=[1,1,1]` reproduces the published docker number."""
    sampler = GenerateCTSampler(final_spacing_mm=[1.0, 1.0, 1.0])
    vol = torch.zeros(1, 4, 8, 8)
    out = tmp_path / "g.mha"
    sampler._save_mha(vol, out)
    sp = sitk.ReadImage(str(out)).GetSpacing()
    assert tuple(round(s, 4) for s in sp) == (1.0, 1.0, 1.0)
