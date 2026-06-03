"""VLM3D evaluation entrypoint (Hydra).

Usage examples
--------------
# Task 4 — Report2CT best checkpoint, proxy GT (100 cases), GPU 3
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct \\
    model.ckpt_path=/workspace/data/report2ct_work_dir/checkpoints/epoch_079_valloss_1.3198.ckpt \\
    task.n_samples=100

# Quick smoke test (1 sample, no CLIPScore)
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct \\
    model.ckpt_path=<ckpt> \\
    task.n_samples=1 task.metrics.clip_score=false task.metrics.fid_2p5d=false

# GenerateCT baseline
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=generatect task.n_samples=100
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _get_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cfg.device)


def _run_ctgen(cfg: DictConfig) -> dict:
    """Task 4: CT generation pipeline (generate → proxy GT → evaluate)."""
    from src.eval.ct_rate_cases import (
        load_eval_cases,
        prepare_proxy_gt,
        write_prompt_xlsx,
    )
    from src.eval.tasks.ctgen import CTGenEvaluator

    device = _get_device(cfg)
    out_dir = Path(cfg.out_dir)
    pred_dir = out_dir / "predictions"
    gt_dir = Path(cfg.task.gt_dir) if cfg.task.gt_dir else out_dir / "proxy_gt"

    # 1. Load eval cases
    n = cfg.task.n_samples if cfg.task.n_samples else None
    cases = load_eval_cases(n_samples=n)
    if not cases:
        raise RuntimeError("No eval cases found — check CT-RATE dataset path.")
    log.info("Evaluating %d cases", len(cases))

    # 2. Prepare proxy GT (idempotent)
    if cfg.task.gt_mode == "proxy":
        prepare_proxy_gt(cases, gt_dir)
    elif cfg.task.gt_mode == "official":
        if not gt_dir.is_dir() or not any(gt_dir.glob("*.mha")):
            raise FileNotFoundError(
                f"official gt_dir={gt_dir} is empty or missing. "
                "Extract from docker: docker cp <cid>:/opt/app/ground-truth/ ."
            )

    # 2b. Auto-write prompt XLSX for CLIPScore (proxy mode only)
    prompt_xlsx = cfg.task.get("prompt_xlsx")
    if (
        prompt_xlsx is None
        and cfg.task.gt_mode == "proxy"
        and cfg.task.metrics.get("clip_score", False)
    ):
        prompt_xlsx = out_dir / "prompts.xlsx"
        write_prompt_xlsx(cases, prompt_xlsx)

    # 3. Generate predictions
    sampler = hydra.utils.instantiate(cfg.model)
    sampler.generate(cases, pred_dir, device)

    # 4. Evaluate
    evaluator = CTGenEvaluator(
        gt_dir=gt_dir,
        metrics=OmegaConf.to_container(cfg.task.metrics, resolve=True),
        ctclip_ckpt=cfg.task.get("ctclip_ckpt"),
        prompt_xlsx=prompt_xlsx,
    )
    results = evaluator.evaluate(pred_dir, out_dir)
    return results


_TASK_RUNNERS = {
    "ctgen": _run_ctgen,
}


@hydra.main(
    config_path="../configs/eval",
    config_name="default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    task_name = cfg.task.name
    runner = _TASK_RUNNERS.get(task_name)
    if runner is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(_TASK_RUNNERS.keys())}"
        )

    results = runner(cfg)

    # Pretty-print final metrics
    print("\n" + "=" * 60)
    print(f"  VLM3D Task 4 ({task_name}) — {cfg.model.name}")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<25s} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60 + "\n")

    # Also save a top-level results summary
    summary_path = Path(cfg.out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"task": task_name, "model": cfg.model.name, "metrics": results},
            f,
            indent=2,
        )
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
