"""Saliency runner 통일 인터페이스 (arm 비종속).

run_saliency.py 가 CT-CLIP / fVLM 어떤 arm이든 동일하게 구동하도록 하는 계약:

    runner = SomeRunner(cfg)
    runner.build()                              # 백본 빌드 + weight 로드 (1회)
    for case in cases:
        mi = runner.preprocess(case)            # 케이스당 1회 (비싼 볼륨 로드/리샘플)
        for item in prompt_items:
            res = runner.saliency(mi, item)     # 프롬프트당 saliency
            if res is None:                     # fVLM: organ 없음 → skip
                continue
            # viz: mi.display_volume (배경) + res.cam (오버레이)

전처리·인코더 호출은 각 runner가 캡슐화 (모델별로 근본적으로 다름). 노출되는
타입(ModelInput / SaliencyResult)만 통일.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tests.saliency_map.data import SaliencyCase
from tests.saliency_map.prompts import PromptItem


@dataclass
class ModelInput:
    """케이스 1건의 전처리 결과 (프롬프트들 사이에서 재사용).

    runner 별 텐서(ctclip: 볼륨; fvlm: image+mask+grid)는 서브클래스가 필드 추가.
    base 는 viz 배경에 쓸 display_volume 만 요구.
    """

    scan_id: str
    display_volume: np.ndarray  # (D, H, W) viz 배경 (전처리 그리드, 표시용 스칼라)


@dataclass
class SaliencyResult:
    """프롬프트 1건의 saliency 산출."""

    item_name: str  # abnormality 이름 (파일명 키)
    organ: str | None  # 라우팅된 장기 (fVLM) 또는 None (CT-CLIP global)
    cam: np.ndarray  # (D, H, W) Grad-CAM, display 와 동일 그리드, [0,1]
    display: (
        np.ndarray
    )  # (D, H, W) 이 cam 의 배경 (CT-CLIP: 전처리 볼륨, fVLM: window 프레임)
    score: float  # S = cos(I,t_pos) − cos(I,t_neg) (raw contrast)
    pred_prob: float  # zero-shot 양성 확률 softmax([cos_neg, cos_pos])[1]
    native_map: np.ndarray | None  # (D, H, W) native token-text sim 맵, 없으면 None
    grid_shape: tuple[int, int, int]  # 토큰 그리드 (d, h, w) — 로깅/디버그용
    orient: tuple[str, str, str]  # display (D,H,W) 축 방향코드 (viz canonical 정렬용)


class BaseRunner(ABC):
    """모든 saliency arm 의 공통 인터페이스."""

    name: str = "base"

    def __init__(self, cfg) -> None:
        # cfg: cam.method / native_map / device / (fvlm) fvlm_mask 등을 가진 설정 객체.
        self.cfg = cfg

    @abstractmethod  # 이 메서드는 무조건 구현해야함
    def build(self) -> None:
        """백본 빌드 + weight 로드 (1회)."""

    @abstractmethod
    def preprocess(self, case: SaliencyCase) -> ModelInput:
        """케이스 → 모델 입력 (볼륨/마스크 로드 + 모델별 전처리)."""

    @abstractmethod
    def saliency(
        self, model_input: ModelInput, item: PromptItem
    ) -> SaliencyResult | None:
        """(model_input, prompt) → SaliencyResult. 적용 불가(organ 없음)면 None."""
