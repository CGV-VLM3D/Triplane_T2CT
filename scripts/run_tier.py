"""Tier orchestrator for the triplane-AE research pipeline.

Tiers
-----
0  sanity        pytest single-batch overfit (CPU, ~30 s, no GPU)
1  arch sweep    train at 2mm toy latents, 45 min wall-clock cap
2  mid val       train at 1mm full latents, 360 min (6 h) wall-clock cap

Tier 3 (full run) is intentionally not implemented; see plan.

Usage
-----
    # default: our arch only
    python scripts/run_tier.py --tier 1
    python scripts/run_tier.py --tiers 0,1

    # baseline is explicit opt-in
    python scripts/run_tier.py --tier 1 --arch baseline
    python scripts/run_tier.py --tier 1 --arch ours,baseline

    # preview without executing
    python scripts/run_tier.py --tiers 0,1,2 --arch ours,baseline --dry-run

All non-tier-0 invocations are prefixed with CUDA_VISIBLE_DEVICES=0 to keep
the project's single-GPU convention. Tier 0 runs pytest on CPU and ignores
that variable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (tier, arch) -> argv list (NOT shell string — we exec via subprocess.run with shell=False).
# Tier 0 ignores arch.
DISPATCH: dict[tuple[int, str | None], list[str]] = {
    (0, None): ["pytest", "-m", "tier0", "-v"],
    (1, "ours"): ["python", "scripts/train.py", "--config-name=trial_toy"],
    (1, "baseline"): ["python", "scripts/train.py", "--config-name=trial_toy_baseline"],
    (2, "ours"): ["python", "scripts/train.py", "--config-name=trial_mid"],
    (2, "baseline"): ["python", "scripts/train.py", "--config-name=trial_mid_baseline"],
}

VALID_TIERS = {0, 1, 2}
VALID_ARCHS = {"ours", "baseline"}


def _parse_tiers(spec: str) -> list[int]:
    tiers: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError as e:
            raise SystemExit(f"--tiers: cannot parse {tok!r} as int") from e
        if n == 3:
            raise NotImplementedError(
                "Tier 3 deferred — add trial_full*.yaml and extend the dispatch table."
            )
        if n not in VALID_TIERS:
            raise SystemExit(f"--tiers: {n} not in {sorted(VALID_TIERS)}")
        tiers.append(n)
    if not tiers:
        raise SystemExit("--tiers: empty list")
    return tiers


def _parse_archs(spec: str) -> list[str]:
    archs: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in VALID_ARCHS:
            raise SystemExit(f"--arch: {tok!r} not in {sorted(VALID_ARCHS)}")
        archs.append(tok)
    if not archs:
        raise SystemExit("--arch: empty list")
    return archs


def _plan(tiers: list[int], archs: list[str]) -> list[tuple[str, list[str]]]:
    """Return ordered list of (label, argv) pairs to execute."""
    out: list[tuple[str, list[str]]] = []
    for tier in tiers:
        if tier == 0:
            out.append(("tier0/sanity", DISPATCH[(0, None)]))
        else:
            for arch in archs:
                argv = list(DISPATCH[(tier, arch)])
                out.append((f"tier{tier}/{arch}", argv))
    return out


def _format_cmd(argv: list[str], env_extra: dict[str, str]) -> str:
    env_prefix = " ".join(f"{k}={v}" for k, v in env_extra.items())
    return (env_prefix + " " + " ".join(argv)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tier", type=int, help="single tier (0, 1, or 2)")
    group.add_argument(
        "--tiers", type=str, help="comma-separated tier list, e.g. 0,1,2"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ours",
        help="comma-separated archs: ours | baseline | ours,baseline (default: ours)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="append train.resume=off to train commands (Hydra override)",
    )
    args = parser.parse_args()

    if args.tier is not None:
        if args.tier == 3:
            raise NotImplementedError(
                "Tier 3 deferred — add trial_full*.yaml and extend the dispatch table."
            )
        if args.tier not in VALID_TIERS:
            raise SystemExit(f"--tier: {args.tier} not in {sorted(VALID_TIERS)}")
        tiers = [args.tier]
    else:
        tiers = _parse_tiers(args.tiers)

    archs = _parse_archs(args.arch)
    plan = _plan(tiers, archs)

    # Build env for non-tier-0 commands.
    gpu_env = {"CUDA_VISIBLE_DEVICES": "0"}

    print(f"[run_tier] {len(plan)} job(s) planned:")
    for i, (label, argv) in enumerate(plan, 1):
        env_extra = {} if argv[0] == "pytest" else gpu_env
        extra_args = (
            ["train.resume=off"] if (args.no_resume and "train.py" in argv[1]) else []
        )
        full_argv = argv + extra_args
        print(f"  [{i}] {label:24s}  {_format_cmd(full_argv, env_extra)}")

    if args.dry_run:
        return 0

    results: list[tuple[str, int, float]] = []
    for label, argv in plan:
        env_extra = {} if argv[0] == "pytest" else gpu_env
        extra_args = (
            ["train.resume=off"] if (args.no_resume and "train.py" in argv[1]) else []
        )
        full_argv = argv + extra_args

        env = os.environ.copy()
        env.update(env_extra)

        print(f"\n[run_tier] -> START  {label}: {_format_cmd(full_argv, env_extra)}")
        t0 = time.time()
        try:
            proc = subprocess.run(full_argv, cwd=str(ROOT), env=env, check=False)
            rc = int(proc.returncode)
        except FileNotFoundError as e:
            print(f"[run_tier] command not found: {e}")
            rc = 127
        dt = time.time() - t0
        results.append((label, rc, dt))
        print(f"[run_tier] <- END    {label}  exit={rc}  elapsed={dt / 60:.1f} min")
        if rc != 0:
            print(f"[run_tier] STOP: {label} returned {rc}; later jobs skipped.")
            break

    print("\n[run_tier] summary:")
    for label, rc, dt in results:
        status = "OK " if rc == 0 else f"FAIL({rc})"
        print(f"  {status}  {label:24s}  {dt / 60:.1f} min")

    return (
        0 if all(rc == 0 for _, rc, _ in results) and len(results) == len(plan) else 1
    )


if __name__ == "__main__":
    sys.exit(main())
