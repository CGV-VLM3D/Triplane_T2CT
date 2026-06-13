"""fVLM anatomy decomposed report 전처리 파이프라인

fVLM (alibaba-damo-academy/fvlm, ICLR 2025)은 CT 스캔을 영상의학과 보고서에
**장기별**로 정렬. 각 보고서는 4개 흉부 장기 ``lung / heart / esophagus / aorta``
에 대한 장기별 텍스트로 분해됨. 저자는 CT-RATE **train** 분할에 대해서만
분해 결과를 두 JSON 파일로 릴리스:

    /workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report/desc_info.json   # 장기별 *Findings*
    /workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report/conc_info.json   # 장기별 *Impression*

(스캔 dict에 장기 키가 있으면 그 섹션에서 "언급된" 것으로 판단 — Appendix Fig 6).
공식 ``valid_fixed`` 세트는 릴리스되지 않음;
``scripts/decompose_reports_fvlm.py``가 Qwen으로 동일한 두 JSON 파일을 재현하므로
이 로더는 두 경로에서 동일하게 동작.

이 모듈이 **단일 소스**로 관리하는 것:
  * 정규 장기 목록 + 기본 "정상" 템플릿,
  * ``build_organ_text`` — 장기별 텍스트 구성. upstream 학습 데이터셋
    ``third_party/fvlm/lavis/datasets/datasets/caption_datasets.py:69-91``의
    **동작 동일(behavior-exact) 미러** — 순수 함수로 재작성했을 뿐 출력은
    동일하여 저자-JSON(train/split)과 Qwen-JSON(valid_fixed)이 호환됨,
  * 장기별 비정상 플래그 규칙 (upstream L130-133),
  * ``src/data/ct_rate_datamodule.py``와 공유하는 scan-id 헬퍼.

torch / lavis / monai 의존성 없이 유지: 단위 테스트와 Qwen 사전계산 스크립트가
무거운 의존성 없이 import 가능.
"""

from __future__ import annotations

import json
from pathlib import Path

# 정규 장기 목록 + 순서. 다음에서 그대로 가져옴:
# third_party/fvlm/lavis/datasets/datasets/caption_datasets.py:50-52
# (blip_pretrain.py:102-104, eval.py:140-142 와도 동일).
ORGANS: tuple[str, ...] = ("lung", "heart", "esophagus", "aorta")

DEFAULT_REPORT_ROOT = Path(
    "/workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report"
)
DEFAULT_SPLIT_JSON = Path("/workspace/datasets/split.json")


def default_template(organ: str) -> str:
    """impression에 장기가 없을 때 주입하는 "정상" 채움 문구.

    upstream의 정확한 리터럴 (caption_datasets.py:82, 비교는 L132/L136). 표현을
    바꾸면 안 됨 — 비정상 플래그 검사와 학습 시 템플릿 제거
    (blip_pretrain.py:212-214)가 이 문자열로 ``startswith`` / ``!=`` 비교함.
    """
    return f"{organ} shows no significant abnormalities."


def load_decomposed_report(
    desc_path: str | Path = DEFAULT_REPORT_ROOT / "desc_info.json",
    conc_path: str | Path = DEFAULT_REPORT_ROOT / "conc_info.json",
) -> tuple[dict, dict]:
    """두 분해 JSON 로드 (caption_datasets.py:60-61 미러).

    스캔 id (예: ``"train_1_a"``)를 키로 하는 ``(desc_info, conc_info)`` 원시 dict 반환.
    각 값은 ``{"Findings"/"Conclusion": <전체 텍스트>, <장기>: <장기별 텍스트>, ...}``
    형식이며 장기 키는 그 섹션에서 언급된 경우에만 존재.
    Qwen 경로가 같은 두 아티팩트를 다시 쓸 수 있도록 2-튜플(하나로 합친 dict 아님)로 반환.
    """
    with open(desc_path, encoding="utf-8") as f:
        desc_info = json.load(f)
    with open(conc_path, encoding="utf-8") as f:
        conc_info = json.load(f)
    return desc_info, conc_info


def build_organ_text(
    desc_info: dict,
    conc_info: dict,
    scan_id: str,
    organs: tuple[str, ...] = ORGANS,
) -> dict[str, str]:
    """스캔 하나에 대한 장기별 ``input_text`` 구성.

    upstream 학습 데이터셋
    ``third_party/fvlm/lavis/datasets/datasets/caption_datasets.py:69-91``의
    동작 동일 미러. 논문 분해의 post-processing "step 3" (§3.1):
    impression 파트와 findings 파트를 연결하고, impression에 장기가 없으면
    기본 "정상" 템플릿을 합성. JSON에는 *언급된* 장기만 저장되어 있으므로
    템플릿/연결은 파일에 구워지지 않고 여기서 모든 분할에 대해 동일하게 재구성.

    ``organs``의 **모든** 키가 항상 있는 dict 반환.
    """
    out: dict[str, str] = {}
    desc_scan = desc_info.get(scan_id, {})
    conc_scan = conc_info.get(scan_id, {})
    for organ in organs:
        # L69-73: findings 파트, 마침표 없으면 추가.
        desc = ""
        if organ in desc_scan:
            desc += desc_scan[organ]
            if not desc.endswith("."):
                desc += "."

        # L75-79: impression 파트, 마침표 없으면 추가.
        conc = ""
        if organ in conc_scan:
            conc += conc_scan[organ]
            if not conc.endswith("."):
                conc += "."

        # L81-82: 이 장기에 impression이 없으면 → 기본 "정상" 템플릿.
        if not len(conc):
            conc = default_template(organ)

        # L84: impression 먼저, 그 다음 findings.
        input_text = conc + desc

        # L85-88: 따옴표와 괄호 제거 (upstream과 동일한 순서).
        input_text = input_text.replace('"', "")
        input_text = input_text.replace("'", "")
        input_text = input_text.replace("(", "")
        input_text = input_text.replace(")", "")

        out[organ] = input_text
    return out


def organ_abnormal_flags(
    organ_text: dict[str, str], organs: tuple[str, ...] = ORGANS
) -> dict[str, bool]:
    """장기별 비정상 플래그 — caption_datasets.py:130-133 미러.

    텍스트가 존재하고 기본 "정상" 템플릿으로 시작하지 않으면 비정상.
    즉, impression에 장기가 언급된 경우 (논문 FNR 규칙 §3.3:
    impression에 없는 장기는 정상으로 처리).
    ``dict[str, bool]`` 반환 (torch 미사용 — 호출자가 필요하면 텐서화).
    """
    return {
        organ: (
            organ in organ_text
            and not organ_text[organ].startswith(default_template(organ))
        )
        for organ in organs
    }


def fill_missing_with_template(
    organ_text: dict[str, str], organs: tuple[str, ...] = ORGANS
) -> dict[str, str]:
    """없는 장기에 기본 템플릿 주입 — caption_datasets.py:135-136.

    ``build_organ_text``는 이미 모든 장기를 반환하므로 거기서는 방어적 no-op;
    부분 dict를 전달하는 호출자(예: Qwen 원시 출력)를 위해 유지.
    """
    out = dict(organ_text)
    for organ in organs:
        if organ not in out:
            out[organ] = default_template(organ)
    return out


# ---------------------------------------------------------------------------
# Scan-id 헬퍼 (src/data/ct_rate_datamodule.py:66-74 와 공유하는 규약)
# ---------------------------------------------------------------------------


def scan_id_from_volume_name(volume_name: str) -> str:
    """``'valid_1_a_1.nii.gz'`` -> ``'valid_1_a'`` (스캔-스터디 id).

    ``ct_rate_datamodule._resolve_nifti_path``의 스터디 id 유도
    (``"_".join(parts[:3])``)와 동일하며, 저자의 분해 키(``'train_1_a'``)와 일치.
    재구성 인덱스가 다른 경우 보고서는 동일하므로 끝의 ``_<recon-idx>``를 제거.
    """
    parts = Path(volume_name).name.replace(".nii.gz", "").split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected VolumeName format: {volume_name}")
    return "_".join(parts[:3])


def scan_ids_from_split(
    split: str, split_json: str | Path = DEFAULT_SPLIT_JSON
) -> list[str]:
    """``split.json`` 그룹('train'/'valid')의 스캔-스터디 id 목록, 중복 제거.

    ``split.json`` train/valid 항목은 ``{"patient_id", "path"}`` dict이며
    ``path``가 ``<scan>/<volume>.nii.gz``으로 끝남; 볼륨 basename에서 scan id 유도.
    순서 보존 중복 제거 (여러 재구성이 하나의 스캔으로 합쳐짐).
    """
    with open(split_json, encoding="utf-8") as f:
        data = json.load(f)
    entries = data[split]
    seen: set[str] = set()
    ids: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            sid = scan_id_from_volume_name(Path(entry["path"]).name)
        else:  # 평문 문자열 (예: 'unused'/'missing_a_1' patient id)
            sid = str(entry)
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def build_split_organ_texts(
    split: str,
    desc_info: dict,
    conc_info: dict,
    split_json: str | Path = DEFAULT_SPLIT_JSON,
    organs: tuple[str, ...] = ORGANS,
) -> dict[str, dict[str, str]]:
    """split 그룹의 모든 스캔에 대한 ``{scan_id: build_organ_text(...)}``."""
    return {
        sid: build_organ_text(desc_info, conc_info, sid, organs)
        for sid in scan_ids_from_split(split, split_json)
    }
