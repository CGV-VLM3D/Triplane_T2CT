"""Tier orchestrator for the triplane-AE research pipeline.

Tiers
-----
0  sanity        pytest single-batch overfit (CPU, ~30 s, no GPU)
1  arch sweep    train at toy scale (~60 min wall-clock cap, set in config)
2  mid val       train at mid scale (~6 h wall-clock cap, set in config)

Tier 3 (full run) is intentionally not implemented.

The orchestrator is config-driven: pass any yaml name under `src/configs/`
via `--configs`. To add a new model variant, drop a yaml in `src/configs/`
and reference it by name — no code change here.

The orchestrator does NOT enforce tier↔config matching (e.g. it won't stop
you from running `trial_mid` under `--tier 1`). The wall-clock cap, data
path, model size, etc. all live in the yaml. Tier is only a label — and
gates which pytest tests run at tier 0.

Usage
-----
    # tier 0 needs no config (runs the sanity pytest)
    python scripts/run_tier.py --tier 0

    # tier 1 / 2: pass any config name(s) under src/configs/
    python scripts/run_tier.py --tier 1 --configs trial_toy
    python scripts/run_tier.py --tier 1 --configs trial_toy,trial_toy_baseline,trial_toy_d3t
    python scripts/run_tier.py --tier 2 --configs trial_mid

    # multiple tiers, same config list reused per tier
    python scripts/run_tier.py --tiers 0,1 --configs trial_toy

    # preview without executing
    python scripts/run_tier.py --tier 1 --configs trial_toy,trial_toy_d3t --dry-run

    # ignore auto-resume (fresh start)
    python scripts/run_tier.py --tier 1 --configs trial_toy --no-resume

Convention
----------
- All non-tier-0 commands are prefixed with CUDA_VISIBLE_DEVICES=0
  (single-GPU project convention).
- Configs are validated against `src/configs/<name>.yaml` existence before
  any subprocess is launched, so typos fail fast at planning time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "src" / "configs"

VALID_TIERS = {0, 1, 2}


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
                "Tier 3 deferred — define a tier-3 config and pass it via --configs."
            )
        if n not in VALID_TIERS:
            raise SystemExit(f"--tiers: {n} not in {sorted(VALID_TIERS)}")
        tiers.append(n)
    if not tiers:
        raise SystemExit("--tiers: empty list")
    return tiers


def _parse_configs(spec: str) -> list[str]:
    """Parse comma-separated config names and validate each yaml exists."""
    if not spec:
        return []
    configs: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith(".yaml"):
            tok = tok[: -len(".yaml")]
        cfg_path = CONFIGS_DIR / f"{tok}.yaml"
        if not cfg_path.is_file():
            available = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
            raise SystemExit(
                f"--configs: {tok!r} not found at {cfg_path}\n  available: {available}"
            )
        configs.append(tok)
    return configs


def _build_command(tier: int, config: str | None) -> list[str]:
    if tier == 0:
        return ["pytest", "-m", "tier0", "-v"]
    if config is None:
        raise SystemExit(f"--tier {tier} requires --configs (none provided)")
    return ["python", "scripts/train.py", f"--config-name={config}"]


def _plan(tiers: list[int], configs: list[str]) -> list[tuple[str, list[str]]]:
    """Ordered list of (label, argv) pairs to execute."""
    out: list[tuple[str, list[str]]] = []
    for tier in tiers:
        if tier == 0:
            out.append(("tier0/sanity", _build_command(0, None)))
        else:
            if not configs:
                raise SystemExit(f"--tier {tier} requires --configs <name>[,<name>...]")
            for cfg in configs:
                out.append((f"tier{tier}/{cfg}", _build_command(tier, cfg)))
    return out


def _format_cmd(argv: list[str], env_extra: dict[str, str]) -> str:
    env_prefix = " ".join(f"{k}={v}" for k, v in env_extra.items())
    return (env_prefix + " " + " ".join(argv)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tier", type=int, help="single tier (0, 1, or 2)")
    group.add_argument(
        "--tiers", type=str, help="comma-separated tier list, e.g. 0,1,2"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help=(
            "comma-separated config names under src/configs/ "
            "(e.g. trial_toy,trial_toy_baseline). Ignored when tier=0."
        ),
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
                "Tier 3 deferred — define a tier-3 config and pass via --configs."
            )
        if args.tier not in VALID_TIERS:
            raise SystemExit(f"--tier: {args.tier} not in {sorted(VALID_TIERS)}")
        tiers = [args.tier]
    else:
        tiers = _parse_tiers(args.tiers)

    configs = _parse_configs(args.configs)
    plan = _plan(tiers, configs)

    gpu_env = {"CUDA_VISIBLE_DEVICES": "0"}

    print(f"[run_tier] {len(plan)} job(s) planned:")
    for i, (label, argv) in enumerate(plan, 1):
        env_extra = {} if argv[0] == "pytest" else gpu_env
        extra_args = (
            ["train.resume=off"]
            if (args.no_resume and len(argv) > 1 and "train.py" in argv[1])
            else []
        )
        full_argv = argv + extra_args
        print(f"  [{i}] {label:32s}  {_format_cmd(full_argv, env_extra)}")

    if args.dry_run:
        return 0

    results: list[tuple[str, int, float]] = []
    for label, argv in plan:
        env_extra = {} if argv[0] == "pytest" else gpu_env
        extra_args = (
            ["train.resume=off"]
            if (args.no_resume and len(argv) > 1 and "train.py" in argv[1])
            else []
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
        print(f"  {status}  {label:32s}  {dt / 60:.1f} min")

    return (
        0 if all(rc == 0 for _, rc, _ in results) and len(results) == len(plan) else 1
    )


if __name__ == "__main__":
    sys.exit(main())
