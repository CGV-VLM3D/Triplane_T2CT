"""Re-export the bundled RFlowScheduler so Hydra `_target_:` can resolve it.

MONAI 1.4 ships without RFlowScheduler; the MAISI bundle vendors its own copy at
`third_party/maisi_bundle/scripts/rectified_flow.py`. This shim adds that scripts/
directory to sys.path once and re-exports `RFlowScheduler` under a stable import
path: `src.baselines.rflow.RFlowScheduler`.

When MONAI is bumped to ≥1.5 (which ships `monai.networks.schedulers.rectified_flow`),
this file can be replaced with a one-line re-export from MONAI proper.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BUNDLE_SCRIPTS = Path("/workspace/third_party/maisi_bundle/scripts")
if _BUNDLE_SCRIPTS.is_dir() and str(_BUNDLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_SCRIPTS))

from rectified_flow import RFlowScheduler  # noqa: E402

__all__ = ["RFlowScheduler"]
