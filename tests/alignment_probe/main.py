"""config.yaml(+ CLI key=value)의 stage에 따라 실행.

python -m tests.alignment_probe.main                                  # config.yaml 그대로
python -m tests.alignment_probe.main stage=train epochs=30            # override
python -m tests.alignment_probe.main stage=embed encoder=ctclip split=train

stage:
  - embed: 각 텍스트 인코더별 embedding을 생성 (embed.py)
  - train: latent를 입력으로 받는 embedder를 학습 (train.py)
  - eval: probe 실험 진행 
  
"""

from __future__ import annotations

from tests.alignment_probe import embed, train
from tests.alignment_probe import eval as eval_stage
from tests.alignment_probe.config import load_config

_STAGES = {"embed": embed.run, "train": train.run, "eval": eval_stage.run}


def main() -> None:
    cfg = load_config()
    if cfg.stage not in _STAGES:
        raise ValueError(f"unknown stage: {cfg.stage!r} (embed|train|eval)")
    _STAGES[cfg.stage](cfg)


if __name__ == "__main__":
    main()
