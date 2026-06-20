"""Generate Text2CT predictions on the CT-RATE valid 1000-split (generation only).

Decoupled from the docker eval so a long run is not blocked on Docker availability and is
fully resumable (existing .mha files are skipped). After this finishes, score with:

    python scripts/run_eval.py model=text2ct task.n_samples=1000

which reuses these predictions (same out_dir layout) and runs valid-GT prep + VLM3D-Dockers.

Usage:
    CUDA_VISIBLE_DEVICES=2 HF_HOME=/workspace/data/checkpoints/hf_cache \
        python scripts/generate_text2ct_valid.py --n 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402

from src.eval.ct_rate_cases import load_eval_cases  # noqa: E402
from src.eval.samplers.text2ct import Text2CTSampler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_text2ct")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Text2CT generation on CT-RATE valid split"
    )
    ap.add_argument("--n", type=int, default=1000, help="number of valid cases (head)")
    ap.add_argument("--n-steps", type=int, default=30, help="RFlow sampling steps")
    ap.add_argument(
        "--out",
        type=str,
        default="/workspace/data/vlm3d_eval/ctgen/text2ct/predictions",
        help="output dir (matches scripts/run_eval.py model=text2ct so eval reuses it)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    cases = load_eval_cases(n_samples=args.n)
    done = len(list(out_dir.glob("*.mha"))) if out_dir.is_dir() else 0
    log.info("Loaded %d cases; %d already generated in %s", len(cases), done, out_dir)

    sampler = Text2CTSampler(n_steps=args.n_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written = sampler.generate(cases, out_dir, device)
    log.info("Done: %d/%d predictions in %s", len(written), len(cases), out_dir)


if __name__ == "__main__":
    main()
