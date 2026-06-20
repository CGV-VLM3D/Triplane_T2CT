"""루트 Hydra config가 에러 없이 compose·resolve되는지 확인.

`python src/train.py --cfg job --resolve`(및 eval.py)가 exit 0인지를
subprocess로 재현해 종료 코드를 검사한다.
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
