"""Saliency 실험 엔트리포인트 (config 1개 = 모델 arm 1개).

    CUDA_VISIBLE_DEVICES=0 python tests/saliency_map/run_saliency.py \
        --config tests/saliency_map/configs/fvlm_mask.yaml [--smoke]

흐름: load_config → setup_run → select_cases → runner.build →
케이스 × 프롬프트 루프(saliency, None skip+log) → viz.save_result → index.json.
"""

from __future__ import annotations

import argparse
import json
import logging

from tests.saliency_map.config import load_config, setup_run
from tests.saliency_map.data import select_cases
from tests.saliency_map.prompts import get_prompt_items
from tests.saliency_map.runners.ctclip_runner import CTCLIPRunner
from tests.saliency_map.runners.fvlm_runner import FVLMRunner
from tests.saliency_map.viz import save_result

_RUNNERS = {"ctclip": CTCLIPRunner, "fvlm": FVLMRunner}


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 3 saliency runner")
    ap.add_argument("--config", required=True, help="path to a config YAML")
    ap.add_argument("--smoke", action="store_true", help="1 case × 1 prompt 만")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = setup_run(cfg)
    log = logging.getLogger("saliency")
    log.info(
        "arm=%s model=%s cam=%s scheme=%s",
        cfg.exp_name,
        cfg.model,
        cfg.cam,
        cfg.prompt_scheme,
    )

    names = list(cfg.abnormalities) if cfg.abnormalities else None
    positive_for = names if (cfg.positive_only and names) else None
    cases = select_cases(n=cfg.n_cases, seed=cfg.seed, positive_for=positive_for)
    items = get_prompt_items(cfg.prompt_scheme, names=names)
    if args.smoke:
        cases, items = cases[:1], items[:1]
    log.info("cases=%d prompts=%d", len(cases), len(items))

    runner = _RUNNERS[cfg.model](cfg)
    runner.build()

    index: list[dict] = []
    for ci, case in enumerate(cases):
        mi = runner.preprocess(case)
        for item in items:
            res = runner.saliency(mi, item)
            if res is None:
                log.info(
                    "[%d/%d] skip %s/%s (organ-absent / N/A)",
                    ci + 1,
                    len(cases),
                    case.scan_id,
                    item.name,
                )
                continue
            save_result(res, case.scan_id, out)
            index.append(
                {
                    "scan_id": case.scan_id,
                    "item": item.name,
                    "organ": res.organ,
                    "score": round(res.score, 4),
                    "pred_prob": round(res.pred_prob, 4),
                    "grid": list(res.grid_shape),
                }
            )
            log.info(
                "[%d/%d] %s/%s organ=%s score=%+.3f prob=%.3f",
                ci + 1,
                len(cases),
                case.scan_id,
                item.name,
                res.organ,
                res.score,
                res.pred_prob,
            )

    (out / "index.json").write_text(json.dumps(index, indent=2))
    log.info("done: %d saliency maps → %s", len(index), out)


if __name__ == "__main__":
    main()
