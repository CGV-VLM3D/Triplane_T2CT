"""Subprocess wrapper for compute_fid_2-5d_ct.py with numpy 2.x compat.

MONAI 1.4's `metrics/fid.py` uses `np.float_` which was removed in NumPy 2.0.
We can't patch site-packages (auto-blocked), so we monkey-patch numpy here
*before* importing the FID script and let fire parse argv from this process.
"""

from __future__ import annotations

import runpy
import sys

import numpy as np

# Shim: np.float_ was removed in NumPy 2.0; alias to np.float64
if not hasattr(np, "float_"):
    np.float_ = np.float64

# Re-execute the upstream FID script in the current namespace so fire.Fire(main)
# parses our argv (already trimmed to drop the wrapper path).
SCRIPT_PATH = (
    "/workspace/third_party/vlm3d_dockers/ctgen_evaluation/compute_fid_2-5d_ct.py"
)

# Pop our own path from argv so fire sees only the FID script's args
sys.argv[0] = SCRIPT_PATH
runpy.run_path(SCRIPT_PATH, run_name="__main__")
