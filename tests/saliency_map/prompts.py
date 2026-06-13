"""Saliency 실험의 contrast 프롬프트 (두 스킴, config 전환).

두 모델에 **동일한** pos/neg 프롬프트를 주어 공정 비교한다. saliency score 는
`S = cos(I, t_pos) − cos(I, t_neg)` — 배경/공통 해부가 상쇄되고 abnormality-specific
신호만 남으며, zero-shot pos-vs-neg 결정과 시각화 근거가 일치한다.

스킴:
  - "fVLM"   : third_party/fvlm/eval.py 의 test_items 16종 (anatomy-tagged).
               DEFAULT_FVLM_TEST_ITEMS 재사용.
  - "ctclip" : third_party/ct_clip/scripts/zero_shot.py 의 18종 pathology,
               "{p} is present." / "{p} is not present." organ 태그는 fVLM 맵에서
               가져오고 없으면 None (Medical material, Lymphadenopathy).

PromptItem.organ 이 None 이면 fVLM(anatomy-routed)은 라우팅할 장기가 없어 건너뛴다
(runner 책임). CT-CLIP(global)은 organ 을 무시하므로 항상 사용 가능.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.baselines.fvlm_preprocess import DEFAULT_FVLM_TEST_ITEMS

from tests.saliency_map.data import ABNORMALITY_LABELS


@dataclass(frozen=True)
class PromptItem:
    """한 abnormality 의 contrast 프롬프트."""

    organ: str | None  # "lung"/"heart"/"esophagus"/"aorta" 또는 None(global-only)
    name: str  # abnormality 이름 (파일명/조회 키)
    neg: str  # 음성 프롬프트 (t_neg)
    pos: str  # 양성 프롬프트 (t_pos)


# abnormality 이름 → 장기 (fVLM test_items 에서 유도; 단일 소스).
_ORGAN_BY_ABNORMALITY: dict[str, str] = {
    name: organ for organ, name, _neg, _pos in DEFAULT_FVLM_TEST_ITEMS
}


def get_prompt_items(
    scheme: str = "fvlm",
    names: list[str] | None = None,
) -> list[PromptItem]:
    """스킴별 PromptItem 목록 반환.

    Args:
        scheme: "fvlm" (16, anatomy-tagged) | "ctclip" (18, present/not-present).
        names: 지정 시 해당 abnormality 이름만 필터 (config `abnormalities`).
    """
    if scheme == "fvlm":
        items = [
            PromptItem(organ=organ, name=name, neg=neg, pos=pos)
            for organ, name, neg, pos in DEFAULT_FVLM_TEST_ITEMS
        ]
    elif scheme == "ctclip":
        items = [
            PromptItem(
                organ=_ORGAN_BY_ABNORMALITY.get(name),  # 없으면 None(global-only), 2개빼고는 모두 가능하다고? 16/18
                name=name,
                neg=f"{name} is not present.",
                pos=f"{name} is present.",
            )
            for name in ABNORMALITY_LABELS
        ]
    else:
        raise ValueError(f"scheme must be 'fvlm' | 'ctclip', got {scheme!r}")

    if names is not None:
        keep = set(names)
        if bad := keep - {it.name for it in items}: # :=는 할당과 동시에 값을 반환하는 walrus operator
            raise ValueError(f"abnormality(s) not in scheme {scheme!r}: {sorted(bad)}")
        items = [it for it in items if it.name in keep]
    return items
