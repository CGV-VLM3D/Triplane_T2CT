"""eval 스테이지 — Probe A(content) + Probe B(image-predictability) → results/probe.json."""

from __future__ import annotations

from pathlib import Path

from tests.alignment_probe.metric.plots import render_probe_figures
from tests.alignment_probe.probes import linear_probe

_RESULTS = Path("tests/alignment_probe/results")


def run(cfg) -> None:
    """
    Probe A + B (선형 probe, CPU) 실행 → results/<exp_name>/probe.json.
      - Probe A: text -> 18-label
      - Probe B: text -> img_feat
      - Probe C: img_feat -> 18-label (img_feat이 충분히 semantic한가?)

    probe.json (+ probe_a_confusion.json) 저장 후 시각화 PNG를 같은 디렉토리에 생성한다.
    """
    out_dir = _RESULTS / cfg.exp_name
    linear_probe.main(out_dir)
    render_probe_figures(
        out_dir, list(cfg.tsne_labels)
    )  # config 지정 라벨(빈 리스트면 자동 선택)
