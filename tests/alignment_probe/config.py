"""alignment_probe 설정 — config.yaml + CLI `key=value` 오버라이드 병합 (OmegaConf)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, DictConfig, OmegaConf

_DEFAULT_CFG = Path(__file__).with_name("config.yaml")


@dataclass
class Config:
    stage: str = MISSING  # embed | train | eval
    device: str = "cuda:0"
    encoder: str = "ctclip"  # embed: t5|report2ct|ctclip|text2ct|fvlm
    split: str = "valid_v2"  # embed: train|valid_v2
    epochs: int = 20  # train
    batch_size: int = 16  # train
    lr: float = 1e-3  # train
    weight_decay: float = 1e-2  # train (AdamW)
    num_workers: int = 8  # train
    exp_name: str = "default"  # train: results/<exp_name>/ 산출물 네임스페이스
    # eval: t-SNE 색칠 abnormality(ABNORMALITY_LABELS 중 택); 빈 리스트면 양성 빈도 상위 자동 선택
    tsne_labels: list[str] = field(default_factory=list)


def load_config() -> DictConfig:
    """config.yaml + CLI `key=value` 오버라이드 → 검증된 DictConfig."""
    cfg = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(_DEFAULT_CFG),
        OmegaConf.from_cli(),
    )
    if OmegaConf.is_missing(cfg, "stage"):
        raise ValueError("config missing required field: 'stage'")
    return cfg
