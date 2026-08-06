#!/usr/bin/env python3
"""Generate report2ct predictions with a per-case FIXED noise seed.

Why this exists
---------------
The spacing study needs two prediction sets that differ ONLY in the declared/conditioned
voxel spacing. `Report2CTSampler._denoise` draws its initial latent with an unseeded
`torch.randn`, so two ordinary runs would also differ by sampling noise — which would be
confounded with the spacing effect we are trying to measure.

This subclass seeds the RNG with a stable per-case value immediately before `_denoise`
runs, so `gen@1.0` and `gen@0.8` start from the *same* noise for every case and the only
difference between the arms is the spacing conditioning.

Seeding point is safe: the only RNG consumer in the inference path is `torch.randn` in
`_denoise` (`RFlowScheduler.set_timesteps` is deterministic — its `torch.rand` lives in
the training-only `sample_timesteps`, rectified_flow.py:121).

Production code is NOT modified — this is a thin subclass living under tests/.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/gen_seeded.py \
        --spacing 1.0 1.0 1.5 --out-dir tests/spacing_fov/preds/gen_sp1.0
    CUDA_VISIBLE_DEVICES=3 python tests/spacing_fov/gen_seeded.py \
        --spacing 0.8 0.8 1.5 --out-dir tests/spacing_fov/preds/gen_sp0.8
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import zlib

import torch

sys.path.insert(0, "/workspace")

from src.eval.ct_rate_cases import load_eval_cases  # noqa: E402
from src.eval.samplers.base import EvalCase  # noqa: E402
from src.eval.samplers.report2ct import Report2CTSampler  # noqa: E402

# toy model the spacing study is run on (5k datalist, epoch 099)
DEFAULT_CKPT = "/workspace/outputs/report2ct/2026-06-20_toy_v2/checkpoints/last.ckpt"


def case_seed(scan_id: str) -> int:
    """Stable per-case seed. Uses crc32, not `hash()` — the latter is salted per process."""
    return zlib.crc32(scan_id.encode()) & 0x7FFFFFFF


class SeededReport2CTSampler(Report2CTSampler):
    """Report2CTSampler whose initial latent noise is a deterministic function of scan_id."""

    def _case_to_context(self, case: EvalCase) -> torch.Tensor:
        """Build the cross-attention context, then seed the RNG for this case's `_denoise`.

        Args:
            case: the evaluation case; its ``scan_id`` determines the seed.

        Returns:
            Cross-attention context ``(1, 2, 2560)`` — unchanged from the parent.
        """
        context = super()._case_to_context(case)  # (1, 2, 2560)
        # Seed AFTER the text encoders run, so the RNG state is pristine at the moment
        # `_denoise` calls torch.randn(1, 4, 120, 120, 64).
        torch.manual_seed(case_seed(case.scan_id))
        return context


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="voxel spacing (mm) — UNet conditioning AND saved .mha header (no default, on purpose)",
    )
    ap.add_argument("--out-dir", required=True, help="directory for the generated .mha")
    ap.add_argument(
        "--n", type=int, default=100, help="number of eval cases (seeded subset)"
    )
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="generate only these scan_ids (must be inside the --n subset). Use this instead "
        "of shrinking --n: load_eval_cases picks its subset by a seeded shuffle over all 1304, "
        "so a smaller --n is a DIFFERENT set of patients, not a prefix of this one.",
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="walk the case list backwards; lets a second GPU work the same out-dir from the "
        "other end (generate() skips files that already exist). Stop the helper before the "
        "two ends meet so they never write the same file at once.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    # cuDNN autotune picks algorithms based on timing, which can vary run to run; pin it off
    # so the two arms use the same kernels.
    torch.backends.cudnn.benchmark = False

    cases = load_eval_cases(n_samples=args.n)
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.scan_id in wanted]
        if len(cases) != len(wanted):
            found = {c.scan_id for c in cases}
            sys.exit(f"--cases not in the n={args.n} subset: {sorted(wanted - found)}")
    if args.reverse:
        cases = cases[::-1]
    print(
        f"[gen_seeded] {len(cases)} cases | spacing={args.spacing} | cfg={args.cfg_scale}"
    )

    sampler = SeededReport2CTSampler(
        ckpt_path=args.ckpt,
        n_steps=args.n_steps,
        spacing_mm=list(args.spacing),
        cfg_scale=args.cfg_scale,
    )
    written = sampler.generate(
        cases, pathlib.Path(args.out_dir), torch.device("cuda:0")
    )
    print(f"[gen_seeded] wrote {len(written)} volumes → {args.out_dir}")


if __name__ == "__main__":
    main()
