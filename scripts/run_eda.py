"""Day 4 EDA runner.

Usage:
    python scripts/run_eda.py [--split valid|train] [--hu-sample N]

Produces ≥4 PNGs in /workspace/figs/eda/ and prints a JSON digest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ct_rate_datamodule import load_records  # noqa: E402
from src.data.ct_rate_eda import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--hu-sample", type=int, default=50)
    parser.add_argument("--out-dir", default="/workspace/figs/eda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Loading {args.split} records...")
    records = load_records(args.split)
    print(f"Loaded {len(records)} records.")

    summary = run_all(records, out_dir, hu_sample_size=args.hu_sample)
    print("\n=== EDA summary ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
