"""Saliency 실험 config 스키마 + 로드 + 런 셋업.

기본값을 한 곳(SaliencyConfig)에 모은다. OmegaConf structured-merge 라 YAML 의
**오타/미지의 키는 로드 시점에 에러**(struct 모드) — 런너에 흩어진 getattr-default 불필요.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class SaliencyConfig:
    # --- 필수 ---
    exp_name: str = MISSING  # results/<exp_name>/ 출력 폴더
    model: str = MISSING  # "ctclip" | "fvlm"

    # --- 공통 ---
    device: str = "cuda"
    cam: str = "layercam"  # cam.CAM_REGISTRY 키
    native_map: bool = False  # native token-text sim 맵 산출
    prompt_scheme: str = "fvlm"  # "fvlm" (16) | "ctclip" (18)
    abnormalities: Optional[List[str]] = None  # None = 스킴 전체
    n_cases: Optional[int] = None  # None = valid_v2 전체(1304)
    positive_only: bool = False  # abnormalities 라벨이 모두 1인 케이스만
    seed: int = 42
    out_root: str = "tests/saliency_map/results"

    # --- fVLM 전용 ---
    fvlm_crop: str = "organ"  # "organ" (장기 center_crop, eval 충실) | "whole"
    fvlm_mask: bool = True  # True (organ 토큰만) | False (전체 토큰)


def load_config(path: str | Path) -> DictConfig:
    """YAML → 검증된 DictConfig (스키마 병합, 미지의 키/타입 오류는 에러)."""
    schema = OmegaConf.structured(SaliencyConfig) # 정의한 구조만 따라야함, 잘못된 키/타입 못들어옴
    cfg = OmegaConf.merge(schema, OmegaConf.load(path))  # 미지의 키 → struct 에러
    for req in ("exp_name", "model"):
        if OmegaConf.is_missing(cfg, req):
            raise ValueError(f"config missing required field: {req!r}")
    if cfg.model not in ("ctclip", "fvlm"): # 나중에 모델 추가하려면 여기에 추가 해야함 
        raise ValueError(f"model must be 'ctclip'|'fvlm', got {cfg.model!r}")
    return cfg


def setup_run(cfg: DictConfig) -> Path:
    """results/<exp_name>/ 생성 + config 스냅샷 + run.log 로깅 셋업. out_dir 반환."""
    out = Path(cfg.out_root) / cfg.exp_name
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    logger = logging.getLogger("saliency")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(out / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return out
