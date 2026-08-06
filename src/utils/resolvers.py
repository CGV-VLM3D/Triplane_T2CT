"""Custom OmegaConf resolvers for run-output directory naming.

Registered on import (idempotent via ``replace=True``). Imported by
``src.utils`` (so ``src/train.py`` and ``src/eval.py`` pick them up) and directly
by ``scripts/run_eval.py``.

``dated_rundir`` replaces the old ``${now:%Y-%m-%d}_${now:%H-%M-%S}`` run-dir
pattern, which used the container's UTC clock (so the wall-clock time never
matched KST) and cluttered every folder name with seconds. Instead it names runs
by the **KST date** and appends ``_2``/``_3``… only when that date already exists:
``outputs/<task>/2026-06-22`` → ``2026-06-22_2`` → ``2026-06-22_3``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from omegaconf import OmegaConf

# KST is a fixed UTC+9 offset (no DST ever) — exact year-round without tzdata.
_KST = timezone(timedelta(hours=9))


def dated_rundir(parent: str, prefix: str = "") -> str:
    """Return ``<parent>/<prefix><KST-date>``, suffixing ``_2``/``_3``… on collision.

    Args:
        parent: Directory the run folder is created under.
        prefix: Optional leaf prefix (e.g. ``"eval_"`` for eval runs).

    Returns:
        A run-directory path whose leaf does not yet exist on disk.
    """
    date = datetime.now(_KST).strftime("%Y-%m-%d")
    base = os.path.join(parent, f"{prefix}{date}")
    if not os.path.exists(base):
        return base
    i = 2
    while os.path.exists(f"{base}_{i}"):
        i += 1
    return f"{base}_{i}"


def path_leaf(path: str | None) -> str:
    """Return the final path component (basename); ``""`` for empty/None input."""
    return os.path.basename(path.rstrip("/")) if path else ""


# use_cache=True on dated_rundir: within one process the same args resolve to one
# fixed path, so repeated resolution of hydra.run.dir never re-scans and bumps the
# suffix mid-run. A fresh process (the next run) re-scans and picks the next suffix.
OmegaConf.register_new_resolver(
    "dated_rundir", dated_rundir, use_cache=True, replace=True
)
OmegaConf.register_new_resolver("path_leaf", path_leaf, replace=True)
