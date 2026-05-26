"""VLM3D-Dockers Task 4 (ctgen) evaluation runner.

Subprocess-wraps `third_party/vlm3d_dockers/ctgen_evaluation/` so any generated
volume directory can be scored against the FVD_CTNet + CLIPScore +
FID_2p5D evaluation pipeline.

Phase A Day 5: provides the wrapper interface + a `--dry-run` mode for the
contract smoke. The actual `docker run` is gated by `DOCKER_AVAILABLE` env
detection — in container-without-DinD environments we skip the real run and
emit a placeholder metrics.json with NaNs so the downstream compare pipeline
can still be exercised. Phase B kickoff (6/1) flips DOCKER_AVAILABLE on.

Usage:
    python -m src.eval.vlm3d_runner --predictions <dir-or-zip> --out results/vlm3d/<model>/metrics.json
    python -m src.eval.vlm3d_runner --dry-run --out /tmp/smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

CTGEN_EVAL_DIR: Final[Path] = Path(
    "/workspace/third_party/vlm3d_dockers/ctgen_evaluation"
)
DEFAULT_IMAGE: Final[str] = "forithmus/ctgen-eval:latest"

# Schema from third_party/vlm3d_dockers/ctgen_evaluation/README.md
EXPECTED_KEYS: Final[tuple[str, ...]] = (
    "FVD_CTNet",
    "CLIPScore",
    "CLIPScore_I2I",
    "CLIPScore_mean",
    "FID_2p5D_Avg",
    "FID_2p5D_XY",
    "FID_2p5D_XZ",
    "FID_2p5D_YZ",
)


def docker_available() -> bool:
    """True iff `docker info` succeeds — meaning a docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def run_docker_eval(
    predictions: Path,
    out_dir: Path,
    image: str = DEFAULT_IMAGE,
) -> dict:
    """Run the ctgen_evaluation container against a predictions directory or zip.

    Args:
        predictions: directory containing .mha files (or a single .zip).
        out_dir: where metrics.json will be written by the container.
        image: docker image to use; default is the public one but a locally-built
            image (e.g. `vlm3d-ctgen-eval:local`) can be passed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{predictions}:/input",
        "-v",
        f"{out_dir}:/output",
        image,
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"Expected metrics.json at {metrics_path} after docker run")
    return json.loads(metrics_path.read_text())


def dry_run_metrics(out_path: Path) -> dict:
    """Emit a placeholder metrics.json with NaNs for every expected key.

    Used when docker daemon is unavailable (Phase A dev container) so downstream
    compare scripts can still be wired and unit-tested.
    """
    metrics = {k: math.nan for k in EXPECTED_KEYS}
    metrics["_status"] = "dry_run"
    metrics["_note"] = "Docker daemon unavailable; placeholder NaNs."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def validate_metrics(metrics: dict) -> None:
    """Assert all expected keys are present (NaNs OK for dry runs)."""
    missing = [k for k in EXPECTED_KEYS if k not in metrics]
    if missing:
        raise ValueError(f"metrics.json missing required keys: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM3D-Dockers Task 4 evaluation runner"
    )
    parser.add_argument("--predictions", type=Path, help="Predictions dir or .zip")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output metrics.json path"
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip docker, emit placeholder"
    )
    args = parser.parse_args()

    if args.dry_run or not docker_available():
        if not args.dry_run:
            print("docker daemon unavailable; falling back to --dry-run mode.")
        metrics = dry_run_metrics(args.out)
    else:
        if args.predictions is None:
            print("--predictions required when not in --dry-run mode")
            return 2
        out_dir = args.out.parent
        metrics = run_docker_eval(args.predictions, out_dir, image=args.image)

    validate_metrics(metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
