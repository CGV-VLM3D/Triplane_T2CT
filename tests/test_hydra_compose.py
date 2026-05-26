"""Verify the root Hydra configs compose and resolve without error.

Phase A acceptance criterion: `python src/train.py --cfg job --resolve` exits 0.
This test reproduces that via subprocess so we capture the exit code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_train_cfg_resolves() -> None:
    result = _run([sys.executable, "src/train.py", "--cfg", "job", "--resolve"])
    assert result.returncode == 0, (
        f"train.py --cfg job --resolve failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )


def test_eval_cfg_resolves() -> None:
    result = _run([sys.executable, "src/eval.py", "--cfg", "job", "--resolve"])
    assert result.returncode == 0, (
        f"eval.py --cfg job --resolve failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )
