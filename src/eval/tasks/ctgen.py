"""Task 4 (CT generation) evaluator — wraps vlm3d_dockers scripts via subprocess.

Script interfaces (verified by reading the actual source):
  evaluate_fvd.py      : argparse --generated_dir --gt_root --out_json
  evaluate_clip.py     : argparse --generated_dir --gt_root --prompt_xlsx --out_json
                         hardcodes /opt/app/models/{CT-CLIP_v2.pt,BiomedVLP-CXR-BERT-specialized}
  compute_fid_2-5d_ct.py : fire.Fire(main) with kwargs; logs results to logger.info;
                            expects real/synth file-lists of NIfTI/.mha filenames;
                            model downloaded from torch.hub "Warvito/radimagenet-models"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from src.eval._vlm3d_paths import ctclip_pkg_parents, ctgen_eval_dir
from src.eval.tasks._fid_refstats import refstats_path_for

log = logging.getLogger(__name__)


def _merge_metrics(
    out_dir: Path, results: dict[str, object], metric_names: list[str]
) -> dict[str, object]:
    """Merge one scoring pass into ``out_dir/metrics.json``, keeping what earlier passes wrote.

    A pass computes only the metrics it was asked for. Dumping that dict straight out — which
    is what this did before 2026-07-31 — drops everything already recorded: scoring FVD into a
    directory that held CLIP+FID left a **one-key file**. That is the only reason FVD-only
    re-scores had to be parked in a separate ``fvd/`` subdirectory instead of landing beside
    the numbers they belong with (docs/ctgen_local_eval.md).

    Metric keys stay **flat at the top level** — every aggregator reads them as
    ``metrics.get("FID_2p5D_Avg")``, so nesting them would break every table. The audit trail
    goes in ``_history``: one dated row per pass naming the metric set it ran, the keys it
    added, and any key it replaced together with the value it replaced.

    Args:
        out_dir: the scoring directory; ``metrics.json`` lives directly inside it.
        results: this pass's metric keys → values.
        metric_names: the metric set that actually ran (e.g. ``["clip_score", "fid_2p5d"]``).

    Returns:
        Every metric recorded for this directory, flat and without ``_history`` — the whole
        picture, not just this pass.

    Raises:
        json.JSONDecodeError: an existing ``metrics.json`` is unreadable. Failing loudly beats
            replacing a truncated-but-recoverable file with this pass's keys alone.
    """
    path = out_dir / "metrics.json"
    previous: dict = json.loads(path.read_text()) if path.is_file() else {}
    history = previous.pop("_history", [])

    added = sorted(k for k in results if k not in previous)
    replaced = {
        k: previous[k] for k in results if k in previous and previous[k] != results[k]
    }
    history.append(
        {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "metrics": list(metric_names),
            "added": added,
            "replaced": replaced,
        }
    )

    merged = {**previous, **results}
    with open(path, "w") as f:
        json.dump({**merged, "_history": history}, f, indent=2)
    log.info(
        "Metrics saved → %s (%d added, %d replaced, %d kept)",
        path,
        len(added),
        len(replaced),
        len(merged) - len(results),
    )
    return merged


# Resolved once at import: ct_challenges/ layout (a945900+) with old-path fallback.
_EVAL_DIR = ctgen_eval_dir()
_FVD_SCRIPT = _EVAL_DIR / "evaluate_fvd.py"
_CLIP_SCRIPT = _EVAL_DIR / "evaluate_clip.py"
_FID_SCRIPT = _EVAL_DIR / "compute_fid_2-5d_ct.py"

# evaluate_clip.py hardcodes these paths (docker convention; we symlink into them)
_CLIP_MODELS_DIR = Path("/opt/app/models")
_CLIP_CKPT_EXPECTED = _CLIP_MODELS_DIR / "CT-CLIP_v2.pt"
_CLIP_BERT_EXPECTED = _CLIP_MODELS_DIR / "BiomedVLP-CXR-BERT-specialized"

_CTCLIP_DEFAULT = Path("/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt")

# --- FID-2.5D preprocessing params -------------------------------------------------------- #
# Kept as constants so the subprocess command and the shared-GT-feature cache key can never
# drift: any change here changes the cache key, forcing recomputation (correct-by-construction).
# These four are identical in both FID profiles below.
_FID_TARGET_SHAPE = "512x512x512"
_FID_RESAMPLE = "1.0x1.0x1.0"
_FID_ENABLE_PADDING = True
_FID_ENABLE_CENTER_CROP = True

# --- FID-2.5D profiles ---------------------------------------------------------------------- #
# The official eval container runs the upstream FID script with `--model_name squeezenet1_1
# --num_images 100` (evaluation.py:229-230 at pin a945900); our harness historically used the
# script's own defaults (radimagenet_resnet50, every volume). "docker" reproduces the container
# and is the default; "research" reproduces every FID number recorded before 2026-07-29 and is
# what the subgroup-analysis assets (2048-d ref-stats) are built on.
#
# `feat_subdir` MUST differ per profile (not just per feature network): upstream names each
# feature file after the VOLUME only (compute_fid_2-5d_ct.py:622,662 — its `suffix` variable at
# :519,524 is assigned but never used), so two PROFILES sharing one directory would silently
# reuse each other's features via the `os.path.isfile(out_fp)` cache check at :626,666 — even
# "docker" and "docker_n300" (same model, different `num_images`), since a 100-volume dir and a
# 300-volume dir must stay distinguishable on disk, not just by their in-memory (mu, Sigma). Kept
# strict (one dir per profile, not one per model) so this invariant never depends on a downstream
# consumer's own staleness guard being correct.
#
# "docker_n300": same feature network as "docker" (so it reproduces the container's own FID
# numbers, unlike "research"'s radimagenet_resnet50) but scores 300 volumes instead of the
# container's 100 — added 2026-07-27 because subgroup breakdowns are unreliable at n=100 (most
# per-label axes would have single digits or zero GT matches). Promoted into the default list
# 2026-08-01, replacing "research" there (see _DEFAULT_FID_PROFILES below) — subgroup analysis
# now needs it available by default, and n=100 was too small for that use case anyway. Its own
# LOCAL feature dir is always freshly created (feat_subdir differs from "docker"'s, by the
# one-dir-per-profile invariant above), but the underlying GT forward pass is NOT wasted: the
# cross-run shared GT-feature cache (`_shared_gt_feat_dir`, below) is keyed by model only, so a
# "docker" pass scored first in the same run leaves its 100 GT features warm for "docker_n300"
# to hardlink, paying the forward pass only for the marginal ~200.
_FID_PROFILES: dict[str, dict] = {
    "docker": {
        "model": "squeezenet1_1",
        "num_images": 100,  # upstream truncates BOTH filelists to this
        "feat_subdir": "fid_features_squeezenet1_1",
    },
    "docker_n300": {
        "model": "squeezenet1_1",
        "num_images": 300,
        "feat_subdir": "fid_features_squeezenet1_1_n300",
    },
    "research": {
        "model": "radimagenet_resnet50",
        "num_images": None,  # every volume
        "feat_subdir": "fid_features",  # historical path — keeps existing run dirs valid
    },
}
_DEFAULT_FID_PROFILE = "docker"

# What `run_eval.py` scores when `task.fid_profile` is left alone: both squeezenet1_1 families,
# in this order (2026-08-01 — "research" was dropped from the default; still fully available via
# an explicit `task.fid_profile=research` or `task.fid_profile=[docker,docker_n300,research]`
# override). "docker" is the only value comparable to the official container; "docker_n300" is
# the same feature space at 300 volumes, needed by default now that subgroup analysis
# (`subgroup_fid_profile`, see below) reuses it. "research" (radimagenet_resnet50 vs the full
# 3001-volume GT census) remains the only profile comparable to anything recorded before
# 2026-07-29 or to the paper envelope (CLAUDE.md) — dropping it from the default trades that
# automatic comparability for a cheaper routine run; get it back with an explicit override.
# Order matters: the FIRST profile's pass is the one that also computes the profile-INDEPENDENT
# metrics (FVD / FVD_CTCLIP / CLIP), which the others copy rather than recompute; "docker" stays
# first for continuity. Putting "docker_n300" second lets it reuse "docker"'s just-populated
# shared GT-feature cache (both are squeezenet1_1) within the same run.
# CTGenEvaluator itself stays strictly single-profile (`_DEFAULT_FID_PROFILE`) — the loop lives in
# the caller, so one evaluator instance can never mix two feature spaces in one folder.
_DEFAULT_FID_PROFILES = ("docker", "docker_n300")

# Default for the SEPARATE `subgroup_fid_profile` knob (which FID profile subgroup_setlevel.py
# reuses cached features from) — independent of _DEFAULT_FID_PROFILES above, but the single
# place this value is written down in Python. Every Python call site (subgroup_setlevel.py,
# orchestrate.py, run_eval.py's fallback, score_subgroups.py, precompute_subgroup_fid_refstats.py)
# imports this constant instead of repeating the literal, so the default can never drift between
# them again the way it briefly could before 2026-08-01. The one place that CAN'T import it —
# configs/eval/task/ctgen.yaml's `subgroup_fid_profile` key, since YAML can't reference Python —
# is instead kept honest by test_config_defaults_match_the_code_defaults's parity assertion.
_DEFAULT_SUBGROUP_FID_PROFILE = "docker_n300"

# Model name for consumers that are pinned to the research profile (2048-d ref-stats assets).
_RESEARCH_FID_MODEL = _FID_PROFILES["research"]["model"]


def resolve_fid_profiles(value: str | Sequence[str] | None) -> list[str]:
    """Normalise a ``task.fid_profile`` value to an ordered list of known profile names.

    Accepts a single name (the historical form, still valid) or a sequence of them, so an
    existing `task.fid_profile=research` override keeps working unchanged.

    Args:
        value: profile name, sequence of names, or None (→ :data:`_DEFAULT_FID_PROFILES`).

    Returns:
        Profile names in the order given, duplicates dropped (a repeated profile would score
        the same folder twice and log a spurious "replaced" row in its ``_history``).

    Raises:
        ValueError: an unknown profile name, or an empty sequence (which would silently skip
            FID altogether instead of failing where the mistake was made).
    """
    if value is None:
        value = _DEFAULT_FID_PROFILES
    names = [value] if isinstance(value, str) else list(value)
    if not names:
        raise ValueError(
            "task.fid_profile is empty — give one or more of "
            f"{sorted(_FID_PROFILES)}, or set task.metrics.fid_2p5d=false to skip FID."
        )
    unknown = [n for n in names if n not in _FID_PROFILES]
    if unknown:
        raise ValueError(
            f"unknown fid_profile {unknown} — choose from {sorted(_FID_PROFILES)}"
        )
    return list(dict.fromkeys(names))


# A truncating profile scores the first `num_images` of this run's own sorted predictions,
# mirroring the container, which sorts both filelists and truncates (compute_fid_2-5d_ct.py:
# 555-556 real, :569-570 synth). Runs that share a task.n_samples share the scored set — the
# subsets load_eval_cases produces are nested — so they stay mutually comparable; runs at
# different n do not, which is what `scored_stems_sha1` in fid.json exists to expose.
# There is deliberately no "score the same fixed 100 regardless of n" option: it would not buy
# leaderboard equivalence (the server scores a hidden test split we do not have), and locally it
# only helps when comparing runs generated at different n — which the hash already flags.


# Shared GT FID-feature cache. GT features depend ONLY on the GT volume set + the preprocessing
# params above — NOT on the model under evaluation — so they are computed once and reused across
# every model. Without this, each model's _run_fid recomputes the (GPU-heavy) GT forward pass.
# (The volume load/resample still runs: that lives in the read-only upstream FID loop, which
# loads each volume before its `if os.path.isfile(out_fp)` cache check; only the forward is
# skipped when the feature file already exists.)
_SHARED_GT_FEAT_ROOT = Path("/workspace/data/vlm3d_eval/_shared_gt_fidfeat")


def _shared_gt_feat_dir(gt_dir: Path, model: str) -> Path:
    """Cache dir for a GT set's FID features, keyed on the GT-set name + all FID params.

    ``model`` is deliberately REQUIRED (no default): the two FID profiles produce
    incompatible features, and a defaulted model would let a caller silently follow a
    changed default and mix feature spaces inside one cache dir.

    Args:
        gt_dir: the GT ``*.mha`` directory this cache belongs to.
        model: FID feature-network name (``_FID_PROFILES[...]["model"]``).
    """
    key = (
        f"{gt_dir.name}__{model}__{_FID_TARGET_SHAPE}"
        f"__rs{_FID_RESAMPLE}__pad{int(_FID_ENABLE_PADDING)}_crop{int(_FID_ENABLE_CENTER_CROP)}"
    )
    return _SHARED_GT_FEAT_ROOT / key


def _link_shared_features(
    shared_dir: Path, local_feat_dir: Path, names: set[str] | None = None
) -> int:
    """Hardlink cached features from a shared cache into this run's local feature dir.

    Generic over GT and prediction features (neither's identity is baked into this function) —
    GT calls it against the cross-run ``_shared_gt_feat_dir`` cache, predictions against the
    per-run ``_shared_pred_feat_dir`` cache (see both below). The upstream loop skips its
    forward pass when the feature file already exists (compute_fid_2-5d_ct.py:626 real / :666
    synth), so pre-linking shared features makes this profile reuse another profile's work.
    Hardlinks share inodes (no extra disk); the loop only reads existing feature files, so the
    shared inode is never mutated. Falls back to copy across filesystems.

    Args:
        shared_dir: the shared feature cache to link from.
        local_feat_dir: this run's ``<feat_subdir>/{gt,pred}`` dir.
        names: when given, link ONLY these feature filenames. Used for GT under a truncating
            profile, whose GT set is a subset: the whole cache holds every GT volume, and
            linking all of them would leave this run's gt dir describing a different reference
            set than the one being scored (harmless to ``_fid_from_cached_features``, which
            takes explicit names, but it poisons the ref-stats npz auto-built from this dir).
            Predictions don't need this filter — the shared pred cache is already scoped to one
            run's own predictions (see ``_shared_pred_feat_dir``), so it never holds anything
            irrelevant to link in.
    """
    if not shared_dir.is_dir():
        return 0
    linked = 0
    for src in shared_dir.iterdir():
        if not src.is_file():
            continue
        if names is not None and src.name not in names:
            continue
        dst = local_feat_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        linked += 1
    return linked


def _populate_shared_features(local_feat_dir: Path, shared_dir: Path) -> int:
    """Persist freshly-computed features into the shared cache (first profile populates it).

    Hardlinks by default — ``src`` is already a complete, never-mutated file by the time this
    runs (same as `_link_shared_features`'s reasoning in the other direction), so linking is
    just as atomic as copy+rename and costs no extra disk. Falls back to copy across
    filesystems.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for src in local_feat_dir.iterdir():
        if not src.is_file():
            continue
        dst = shared_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)  # atomic: concurrent readers never see a partial file
        added += 1
    if added:
        log.info(
            "FID feature cache: populated %d new features into %s", added, shared_dir
        )
    return added


def _missing_names(local_feat_dir: Path, names: list[str]) -> list[str]:
    """Which of ``names`` are NOT already present as a feature file in ``local_feat_dir``.

    The upstream extraction loop loads + resamples every volume it's given before checking its
    own per-file cache (see ``_link_shared_features``'s docstring) — the only way to skip that
    I/O for an already-cached volume is to never hand it to the loop in the first place. Callers
    pass only this filtered list to the subprocess; whatever's already on disk (via a shared-cache
    hardlink or a prior partial run) is picked up as-is by ``_fid_from_cached_features`` after.
    """
    return [n for n in names if not (local_feat_dir / n).is_file()]


# Per-run prediction FID-feature cache: unlike GT (a fixed, permanent reference set shared
# across every model/run forever, hence the cross-run root above), predictions are unique to
# one run's own generated volumes — so this cache lives beside that run's own predictions and
# is keyed on the ACTUAL pred_dir, not just its parent. Two profiles scoring the same pred_dir
# (docker/docker_n300 within one eval run) share a key and correctly reuse each other's work —
# docker's stems are always the first N of docker_n300's (see resolve_fid_profiles' caller), so
# the larger profile's pass leaves the smaller one's features already sitting here. Keying on
# the parent alone would NOT be safe: some harnesses (tests/orientation_quant/run_quant.py,
# tests/spacing_fov/decompose_2x2.py) score several DIFFERENT prediction sets that share one
# parent directory and reuse filenames across them (same case ids, different volume content per
# model/cell) — exactly the "same basename, different content" hazard `_gt_set_matches` already
# guards against on the GT side. Hashing the full pred_dir path makes that collision
# structurally impossible rather than something a guard has to catch.
def _shared_pred_feat_dir(pred_dir: Path, model: str) -> Path:
    """Cache dir for one prediction directory's FID features, keyed on model + pred_dir identity."""
    key = hashlib.sha1(str(pred_dir).encode()).hexdigest()[:12]
    return pred_dir.parent / "_shared_pred_fidfeat" / model / key


def _fid_from_cached_features(
    features_dir: Path,
    gt_names: list[str],
    pred_names: list[str],
    ref_stats_path: Path | None = None,
) -> dict[str, float] | None:
    """Compute 2.5D-FID on CPU from the per-volume feature files the FID loop already wrote.

    The upstream loop extracts features to ``features_dir/{gt,pred}`` but computes the final
    FID statistics on the GPU, which OOMs at full scale (~512k slice-features per plane across
    6 matrices). Since every per-volume feature file is on disk, we always compute the FID here
    on CPU instead (the host has far more RAM than the GPU). Returns ``None`` if the cached
    features are incomplete (i.e. the subprocess died before finishing extraction).

    Precomputed-GT fast path: when ``ref_stats_path`` is given, the GT distribution's per-plane
    ``(mu, Sigma)`` is loaded from that npz (or built once from the cached GT features and saved),
    so the (large, model-independent) GT set is never re-stacked — only the prediction features
    are. Numerically identical to the full stack (verified ``<2e-9`` vs ``FIDMetric``). Falls back
    to the full ``FIDMetric`` stack when the npz is absent/stale and GT features are complete.

    Args:
        features_dir: dir holding ``gt/`` and ``pred/`` per-volume feature files.
        gt_names: exact GT feature filenames that make up the reference set.
        pred_names: exact prediction feature filenames to score.
        ref_stats_path: optional npz of precomputed GT ``(mu, Sigma)`` (see ``_fid_refstats``).

    The name lists are used verbatim rather than globbing the directories: a reused ``out_dir``
    can hold features from an earlier, different set (upstream names feature files after the
    volume only and skips extraction when one exists), and a glob would silently fold those
    extras into the distribution. Extras are reported so a stale dir is visible.
    """
    import numpy as _np  # noqa: PLC0415

    if not hasattr(
        _np, "float_"
    ):  # MONAI fid._sqrtm uses the NumPy-2-removed np.float_
        _np.float_ = _np.float64
    import torch  # noqa: PLC0415
    from monai.metrics import FIDMetric  # noqa: PLC0415

    gt_dir = features_dir / "gt"
    pred_dir = features_dir / "pred"
    n_gt, n_pred = len(gt_names), len(pred_names)
    gt_files = [gt_dir / n for n in gt_names]
    pred_files = [pred_dir / n for n in pred_names]

    missing_gt = [p.name for p in gt_files if not p.is_file()]
    missing_pred = [p.name for p in pred_files if not p.is_file()]
    if missing_pred:
        log.error(
            "CPU FID: incomplete cached pred features (%d/%d missing, first: %s).",
            len(missing_pred),
            n_pred,
            missing_pred[0],
        )
        return None
    for label, d, names in (("gt", gt_dir, gt_names), ("pred", pred_dir, pred_names)):
        extra = {p.name for p in d.glob("*") if p.is_file()} - set(names)
        if extra:
            log.warning(
                "CPU FID: %s feature dir holds %d file(s) outside the scored set (e.g. %s) — "
                "they are ignored here, but the dir is stale: %s",
                label,
                len(extra),
                sorted(extra)[0],
                d,
            )

    # Fast path: precomputed GT (mu, Sigma). GT is model-independent, so skip re-stacking it.
    if ref_stats_path is not None:
        from src.eval.tasks._fid_refstats import (  # noqa: PLC0415
            compute_ref_stats,
            fid_from_ref_stats,
            load_ref_stats,
            save_ref_stats,
        )

        ref = None
        if Path(ref_stats_path).is_file():
            r = load_ref_stats(ref_stats_path)
            if r["n_volumes"] == n_gt:
                ref = r
            else:  # stale (GT set changed) → rebuild below if GT features are present
                log.warning(
                    "FID ref-stats %s stale (n_volumes %d != gt %d) — rebuilding.",
                    ref_stats_path,
                    r["n_volumes"],
                    n_gt,
                )
        if ref is None and not missing_gt:
            log.info(
                "FID ref-stats: computing GT (mu, Sigma) over %d volumes → %s",
                n_gt,
                ref_stats_path,
            )
            # Explicit names, not a glob: this npz is persisted in the CROSS-MODEL shared
            # cache, so extras swept up here would poison the reference for every later run.
            save_ref_stats(compute_ref_stats(gt_dir, names=gt_names), ref_stats_path)
            ref = load_ref_stats(ref_stats_path)
            if ref["n_volumes"] != n_gt:
                log.error(
                    "FID ref-stats: built %s with n_volumes=%d but the scored GT set is %d — "
                    "refusing to score against a mis-sized reference.",
                    ref_stats_path,
                    ref["n_volumes"],
                    n_gt,
                )
                return None
        if ref is not None:
            fids = fid_from_ref_stats(pred_dir, ref, names=pred_names)
            log.info(
                "CPU FID (precomputed GT ref, %d vols): Avg=%.4f (XY=%.4f YZ=%.4f XZ=%.4f)",
                ref["n_volumes"],
                fids["FID_2p5D_Avg"],
                fids["FID_2p5D_XY"],
                fids["FID_2p5D_YZ"],
                fids["FID_2p5D_XZ"],
            )
            return fids
        # else: npz absent/stale AND GT features incomplete → fall through to full stack

    if missing_gt:
        log.error(
            "CPU FID fallback: incomplete cached features (gt %d/%d missing, first: %s; pred %d).",
            len(missing_gt),
            n_gt,
            missing_gt[0],
            n_pred,
        )
        return None

    # Per-volume feature files are tuples (xy, yz, zx); upstream reports zx as XZ.
    plane_keys = {0: "FID_2p5D_XY", 1: "FID_2p5D_YZ", 2: "FID_2p5D_XZ"}
    fids: dict[str, float] = {}
    for idx, key in plane_keys.items():
        real = torch.vstack(
            [torch.load(f, weights_only=True)[idx].float() for f in gt_files]
        )
        synth = torch.vstack(
            [torch.load(f, weights_only=True)[idx].float() for f in pred_files]
        )
        fids[key] = float(FIDMetric()(synth, real))
        del real, synth
    fids["FID_2p5D_Avg"] = (
        fids["FID_2p5D_XY"] + fids["FID_2p5D_YZ"] + fids["FID_2p5D_XZ"]
    ) / 3.0
    log.info(
        "CPU FID fallback: Avg=%.4f (XY=%.4f YZ=%.4f XZ=%.4f)",
        fids["FID_2p5D_Avg"],
        fids["FID_2p5D_XY"],
        fids["FID_2p5D_YZ"],
        fids["FID_2p5D_XZ"],
    )
    return fids


class CTGenEvaluator:
    """Run FVD / CLIPScore / FID-2.5D on a directory of predicted .mha files.

    Args:
        gt_dir: directory with ground-truth ``*.mha`` files (matching pred filenames).
        metrics: dict with boolean flags ``fvd``, ``clip_score``, ``fid_2p5d``.
        ctclip_ckpt: local path to ``CT-CLIP_v2.pt``.  evaluate_clip.py hardcodes
            ``/opt/app/models/CT-CLIP_v2.pt``; we create a symlink there automatically.
        prompt_xlsx: XLSX file with (Names, Text_prompts) columns for CLIPScore I2T.
            If None, CLIPScore I2T is skipped; only I2I is meaningful.
        fid_profile: which entry of ``_FID_PROFILES`` the 2.5D-FID uses. ``"docker"``
            (default) reproduces the official container (squeezenet1_1, first 100 volumes);
            ``"docker_n300"`` is the same feature network scored over 300 volumes instead of
            100 (more reliable for subgroup breakdowns); ``"research"`` reproduces the
            pre-2026-07-29 numbers (radimagenet_resnet50, every volume). Affects FID only —
            CLIP/FVD are untouched.
    """

    def __init__(
        self,
        gt_dir: str | Path,
        metrics: dict | None = None,
        ctclip_ckpt: str | Path | None = None,
        prompt_xlsx: str | Path | None = None,
        fid_profile: str = _DEFAULT_FID_PROFILE,
    ) -> None:
        """Store GT dir, metric flags, and weight paths; see class docstring for parameter details."""
        if fid_profile not in _FID_PROFILES:
            raise ValueError(
                f"unknown fid_profile {fid_profile!r} — choose from {sorted(_FID_PROFILES)}"
            )
        self.fid_profile = fid_profile
        # Absolute: every metric runs its upstream script as a subprocess with cwd set to the
        # vlm3d_dockers eval dir, so a relative path handed to the child resolves against THAT
        # directory and the child dies with a confusing FileNotFoundError.
        self.gt_dir = Path(gt_dir).resolve()
        self.metrics = metrics or {
            "fvd": True,
            "fvd_ctclip": True,
            "clip_score": True,
            "fid_2p5d": True,
        }
        self.ctclip_ckpt = Path(ctclip_ckpt) if ctclip_ckpt else _CTCLIP_DEFAULT
        self.prompt_xlsx = Path(prompt_xlsx).resolve() if prompt_xlsx else None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run the enabled metrics and merge them into ``out_dir/metrics.json``.

        Returns:
            Every metric recorded for ``out_dir`` — this pass's plus any earlier pass's — flat
            and without the ``_history`` bookkeeping. See :func:`_merge_metrics`.
        """
        pred_dir = Path(pred_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, float] = {}
        ran: list[
            str
        ] = []  # which metric set this pass covered — recorded in the history row

        if self.metrics.get(
            "fvd", False
        ):  # 지금 제공되는 가중치가 깨져있어서 일단 false로 해둠
            results.update(self._run_fvd(pred_dir, out_dir))
            ran.append("fvd")

        if self.metrics.get(
            "fvd_ctclip", False
        ):  # CTNet 손상 → CT-CLIP feature swap proxy
            results.update(self._run_fvd_ctclip(pred_dir, out_dir))
            ran.append("fvd_ctclip")

        if self.metrics.get("clip_score", True):
            results.update(self._maybe_run_clip(pred_dir, out_dir))
            ran.append("clip_score")

        if self.metrics.get("fid_2p5d", True):
            results.update(self._run_fid(pred_dir, out_dir))
            # Label the FID scale in metrics.json itself: the two profiles write the SAME
            # FID_2p5D_* keys on incomparable scales (squeezenet/100 vs radimagenet/all), and
            # every aggregator reads this file. Consumers tolerate a string value
            # (summary.py falls back to str(); compare_runs coerces with errors="coerce").
            results["fid_profile"] = self.fid_profile
            ran.append("fid_2p5d")

        return _merge_metrics(out_dir, results, ran)

    # ------------------------------------------------------------------ #
    #  FVD                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_fvd_paths(self) -> None:
        """FVD/fvd_pytorch.py hardcodes /opt/app/FVD/ctnet/... — symlink it.

        Re-points a stale symlink: after the a945900 reorg an existing
        /opt/app/FVD/ctnet may dangle at the old (pre-ct_challenges) path, where
        ``link.exists()`` is False but ``symlink_to`` still raises FileExistsError.
        """
        opt_fvd_dir = Path("/opt/app/FVD")
        opt_fvd_dir.mkdir(parents=True, exist_ok=True)
        target = (_EVAL_DIR / "FVD" / "ctnet").resolve()
        link = opt_fvd_dir / "ctnet"
        if not target.exists():
            return
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target:
                return  # already correct
            link.unlink()  # stale/wrong → replace
        link.symlink_to(target)
        log.info("Symlinked %s → %s", link, target)

    def _run_fvd(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run evaluate_fvd.py and return a dict with the ``FVD_CTNet`` key.

        Returns NaN for ``FVD_CTNet`` if the subprocess exits non-zero or the
        output JSON is not produced.
        """
        self._setup_fvd_paths()
        # cwd=_EVAL_DIR in the subprocess below — absolute paths only (see __init__).
        pred_dir, out_dir = Path(pred_dir).resolve(), Path(out_dir).resolve()
        out_json = out_dir / "fvd.json"
        cmd = [
            sys.executable,
            str(_FVD_SCRIPT),
            "--generated_dir",
            str(pred_dir),
            "--gt_root",
            str(self.gt_dir),
            "--out_json",
            str(out_json),
        ]
        # `fvd_pytorch.py` imports top-level `ctnet`, a namespace package living in FVD/.
        # The upstream Dockerfile `pip install -e`s it; locally that editable mapping breaks
        # whenever the submodule dir moves (e.g. the a945900 ct_challenges reorg). Putting
        # FVD/ on PYTHONPATH resolves `ctnet` by namespace discovery, independent of any
        # stale editable install.
        env = os.environ.copy()
        fvd_pkg_dir = str((_EVAL_DIR / "FVD").resolve())
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (fvd_pkg_dir, env.get("PYTHONPATH", "")) if p
        )
        rc = self._run(cmd, cwd=_EVAL_DIR, env=env)
        if rc != 0 or not out_json.is_file():
            log.error("FVD evaluation failed (rc=%d).", rc)
            return {"FVD_CTNet": float("nan")}
        with open(out_json) as f:
            return json.load(f)

    def _run_fvd_ctclip(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """CT-CLIP zero-shot 18-d FVD proxy — feature-extractor swap for the corrupt CTNet FVD.

        Runs the faithful docker FVD harness (pairing / CHUNK=4 / mean, imported unchanged from
        ``evaluate_fvd``) with CT-CLIP zero-shot 18-d features instead of CTNet. This is a LOCAL
        proxy, **not** the leaderboard ``FVD_CTNet`` (different feature space). In-process on
        cuda:0 (no subprocess). Returns ``{"FVD_CTCLIP": value}``; NaN if the computation raises.
        """
        out_json = out_dir / "fvd_ctclip.json"
        try:
            # Lazy import: pulls the CT-CLIP adapter + upstream FVD, heavy at module load.
            from src.eval.tasks._fvd_ctclip import compute_fvd_ctclip

            value = compute_fvd_ctclip(pred_dir, self.gt_dir)
        except Exception:
            log.exception("FVD_CTCLIP evaluation failed.")
            return {"FVD_CTCLIP": float("nan")}
        result = {"FVD_CTCLIP": float(value)}
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # ------------------------------------------------------------------ #
    #  CLIPScore                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_run_clip(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Guard-check prerequisites then delegate to ``_run_clip``; return NaN dict on skip.

        Returns NaN values for ``CLIPScore``, ``CLIPScore_I2I``, and ``CLIPScore_mean``
        when the CT-CLIP_v2.pt checkpoint is missing or ``/opt/app/models/`` cannot be
        set up; otherwise returns whatever ``_run_clip`` produces.
        """
        _nan = {
            "CLIPScore": float("nan"),
            "CLIPScore_I2I": float("nan"),
            "CLIPScore_mean": float("nan"),
        }
        if not self.ctclip_ckpt.is_file():
            log.warning(
                "CT-CLIP_v2.pt not found at %s — skipping CLIPScore.\n"
                '  Download: python -c "from huggingface_hub import hf_hub_download; '
                "hf_hub_download('ibrahimhamamci/CT-RATE', 'models/CT-CLIP-Related/CT-CLIP_v2.pt', "
                "repo_type='dataset')\"",
                self.ctclip_ckpt,
            )
            return _nan

        # evaluate_clip.py hardcodes /opt/app/models/ — set up symlinks
        if not self._setup_clip_paths():
            log.warning("Could not set up /opt/app/models/ paths — skipping CLIPScore.")
            return _nan

        return self._run_clip(pred_dir, out_dir)

    def _setup_clip_paths(self) -> bool:
        """Create /opt/app/models/ and symlink CT-CLIP weights + BiomedVLP-BERT."""
        try:
            _CLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Symlink CT-CLIP_v2.pt
            if not _CLIP_CKPT_EXPECTED.exists():
                _CLIP_CKPT_EXPECTED.symlink_to(self.ctclip_ckpt.resolve())
                log.info("Symlinked CT-CLIP_v2.pt → %s", self.ctclip_ckpt)

            # BiomedVLP-CXR-BERT: try HF cache snapshot first
            if not _CLIP_BERT_EXPECTED.exists():
                self._symlink_biomedvlp()

            return _CLIP_CKPT_EXPECTED.exists() and _CLIP_BERT_EXPECTED.exists()
        except PermissionError as e:
            log.warning("Cannot create /opt/app/models/: %s", e)
            return False

    def _symlink_biomedvlp(self) -> None:
        """Symlink BiomedVLP-CXR-BERT-specialized from HF cache or download it."""
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        # Try common HF hub cache patterns
        candidates = list(
            hf_home.glob(
                "hub/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/*/"
            )
        )
        if candidates:
            snapshot = sorted(candidates)[-1]  # latest snapshot
            _CLIP_BERT_EXPECTED.symlink_to(snapshot.resolve())
            log.info("Symlinked BiomedVLP-CXR-BERT-specialized → %s", snapshot)
        else:
            log.warning(
                "BiomedVLP-CXR-BERT-specialized not in HF cache. Downloading to %s …",
                _CLIP_BERT_EXPECTED,
            )
            from huggingface_hub import snapshot_download

            snapshot_download(
                "microsoft/BiomedVLP-CXR-BERT-specialized",
                local_dir=str(_CLIP_BERT_EXPECTED),
            )

    def _run_clip(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run evaluate_clip.py and return a dict with ``CLIPScore``, ``CLIPScore_I2I``, and ``CLIPScore_mean``.

        Appends ``--prompt_xlsx`` to the command when ``self.prompt_xlsx`` is set and
        the file exists; otherwise the upstream script uses its own default path.
        Returns NaN for all three keys if the subprocess exits non-zero or the output
        JSON is not produced.
        """
        # cwd=_EVAL_DIR in the subprocess below — absolute paths only (see __init__).
        pred_dir, out_dir = Path(pred_dir).resolve(), Path(out_dir).resolve()
        out_json = out_dir / "clip.json"
        cmd = [
            sys.executable,
            str(_CLIP_SCRIPT),
            "--generated_dir",
            str(pred_dir),
            "--gt_root",
            str(self.gt_dir),
            "--out_json",
            str(out_json),
        ]
        if self.prompt_xlsx is not None and self.prompt_xlsx.is_file():
            cmd += ["--prompt_xlsx", str(self.prompt_xlsx)]
        else:
            log.info(
                "No prompt_xlsx — CLIPScore I2T will use default xlsx path in script."
            )

        # ct_clip + transformer_maskgit must be importable from the submodule
        env = self._clip_env()
        log.info("Running CLIP eval: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=_EVAL_DIR, env=env, check=False)
        if result.returncode != 0 or not out_json.is_file():
            log.error("CLIPScore evaluation failed (rc=%d).", result.returncode)
            return {
                "CLIPScore": float("nan"),
                "CLIPScore_I2I": float("nan"),
                "CLIPScore_mean": float("nan"),
            }
        with open(out_json) as f:
            return json.load(f)

    @staticmethod
    def _clip_env() -> dict:
        """Env for CLIPScore subprocess.

        evaluate_clip.py imports top-level `transformer_maskgit` and `ct_clip`.
        Upstream `pip install -e`s them; locally that editable mapping breaks when
        the submodule moves (the a945900 reorg). We add each package's *parent*
        dir (not the package dir itself — that would put sub-packages at top level
        and break `import ct_clip.mlm`) to PYTHONPATH so the imports resolve
        regardless of any stale editable install.
        """
        env = os.environ.copy()
        parents = [str(p) for p in ctclip_pkg_parents()]
        if parents:
            env["PYTHONPATH"] = os.pathsep.join(
                [*parents, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        return env

    # ------------------------------------------------------------------ #
    #  FID-2.5D                                                            #
    # ------------------------------------------------------------------ #

    def _gt_set_matches(self, features_dir: Path) -> bool:
        """Bind a run's feature dir to one GT set, so a second GT set can't reuse its features.

        The feature dir is keyed on the PROFILE (network + n) but not on the GT set, while
        upstream names each feature file after the volume alone and skips extraction when that
        file exists (compute_fid_2-5d_ct.py:626/666). Two GT sets whose volumes share basenames
        but differ in content — ``tests/orientation_quant/gt_lps`` vs ``gt_ras`` are exactly
        that — would therefore reuse the first set's features and report the first set's FID,
        with no error and no NaN. The explicit name lists cannot catch it (the names match), so
        record the GT set on first use and refuse afterwards on a mismatch.

        The marker sits at ``features_dir/.gt_set``, NOT inside ``gt/``: everything in ``gt/``
        is copied into the cross-model shared cache and hardlinked into other runs.

        Returns:
            True to proceed. False (with an ERROR log) when this dir belongs to another GT set.
        """
        marker = features_dir / ".gt_set"
        current = str(self.gt_dir)
        if marker.is_file():
            recorded = marker.read_text().strip()
            if recorded != current:
                log.error(
                    "FID-2.5D refused: %s already holds features extracted from GT set\n"
                    "  %s\nbut this run's gt_dir is\n  %s\n"
                    "Feature files are named after the volume only, so same-named volumes from "
                    "a different GT set would silently reuse the first set's features (and "
                    "poison the shared GT cache). Use a separate out_dir per GT set.",
                    features_dir,
                    recorded,
                    current,
                )
                return False
            return True
        # Feature files keep the INPUT name — upstream's `.replace(".nii.gz", ".pt")` never
        # matches our `.mha`, so they are `<volume>.mha`, not `*.pt`. Match on any file.
        if any(p.is_file() for p in features_dir.glob("*/*")):
            # Pre-existing dir from before this marker existed: its provenance is unknowable,
            # so adopt the current GT set but say so rather than implying it was verified.
            log.warning(
                "FID-2.5D: %s already holds features but no .gt_set marker — assuming they "
                "came from %s (unverified; true for any run that used one gt_dir).",
                features_dir,
                current,
            )
        marker.write_text(current + "\n")
        return True

    def _run_fid(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run compute_fid_2-5d_ct.py via fire.Fire kwargs, following ``self.fid_profile``.

        The script does NOT support --out_json; results are in logger.info output, and at our
        scale its own GPU aggregation OOMs — so it serves purely as a feature extractor and the
        Frechet statistic is computed CPU-side from the cached per-volume features.

        Returns:
            ``{"FID_2p5D_{Avg,XY,XZ,YZ}": float}`` plus ``fid_num_images`` (how many volumes
            actually entered the statistic), NaN-filled on failure. Full provenance — profile,
            scored set, model, stem hash — goes to ``out_dir/fid.json``.
        """
        _nan = {
            "FID_2p5D_Avg": float("nan"),
            "FID_2p5D_XY": float("nan"),
            "FID_2p5D_XZ": float("nan"),
            "FID_2p5D_YZ": float("nan"),
        }
        # See __init__: the subprocess runs with cwd=_EVAL_DIR, so relative paths would break.
        pred_dir, out_dir = Path(pred_dir).resolve(), Path(out_dir).resolve()
        prof = _FID_PROFILES[self.fid_profile]
        model = prof["model"]

        # Build file lists (relative filenames, one per line)
        real_filelist = out_dir / "gt_filelist.txt"
        synth_filelist = out_dir / "pred_filelist.txt"

        gt_files = sorted(self.gt_dir.glob("*.mha"))
        pred_files = sorted(pred_dir.glob("*.mha"))
        if not gt_files or not pred_files:
            log.error(
                "FID-2.5D: empty gt_dir (%d) or pred_dir (%d).",
                len(gt_files),
                len(pred_files),
            )
            return _nan
        if len(gt_files) < 10 or len(pred_files) < 10:
            log.warning(
                "FID-2.5D may be unreliable with only %d GT / %d pred files (need ≥50).",
                len(gt_files),
                len(pred_files),
            )

        # Which volumes actually enter the FID.
        #   research: every GT and every prediction.
        #   docker:   the container truncates both filelists to the first `num_images` entries
        #             (compute_fid_2-5d_ct.py:556,570), and its /input/ground_truth holds exactly
        #             the prompted cases — so there sorted(GT)[:k] and sorted(pred)[:k] are the
        #             SAME stems. Our gt_dir is the 3001-volume census, so we reproduce that
        #             pairing explicitly; sorted(gt_dir)[:k] would pick scans unrelated to the
        #             predictions and be *less* faithful, not more.
        if prof["num_images"] is None:
            fid_gt_files, fid_pred_files = gt_files, pred_files
        else:
            k = prof["num_images"]
            fid_pred_files = pred_files[:k]
            if len(fid_pred_files) < k:
                # FID is strongly n-biased, so a short run's value is inflated but plausible
                # and carries the same `fid_profile` label as a full one. Make it loud, and
                # `fid_num_images` in metrics.json lets any table filter on it.
                log.error(
                    "FID-2.5D (%s profile): only %d of the profile's %d volumes are available "
                    "— the value is NOT comparable to a full %d-volume run.",
                    self.fid_profile,
                    len(fid_pred_files),
                    k,
                    k,
                )
            fid_gt_files = [self.gt_dir / f.name for f in fid_pred_files]
            missing = [p.name for p in fid_gt_files if not p.is_file()]
            if missing:
                log.error(
                    "FID-2.5D (%s profile): %d GT volumes missing for the scored stems (first: %s).",
                    self.fid_profile,
                    len(missing),
                    missing[0],
                )
                return _nan
            log.info(
                "%s profile: FID scores the first %d of %d predictions vs the same %d GT stems "
                "(CLIP/FVD use all %d).",
                self.fid_profile,
                len(fid_pred_files),
                len(pred_files),
                len(fid_gt_files),
                len(pred_files),
            )

        # Feature filenames == volume filenames (upstream's `.replace(".nii.gz", ".pt")` is a
        # no-op for our .mha), so these lists key both the linking and the CPU-side FID.
        from src.eval.tasks._fid_refstats import load_ref_stats  # noqa: PLC0415

        gt_names = [f.name for f in fid_gt_files]
        pred_names = [f.name for f in fid_pred_files]

        features_dir = out_dir / prof["feat_subdir"]
        features_dir.mkdir(parents=True, exist_ok=True)
        if not self._gt_set_matches(features_dir):
            return _nan

        # Reuse GT features across models: hardlink any cached features for this GT set into
        # this run's gt dir so the upstream loop skips their (GPU) forward pass.
        gt_feat_dir = features_dir / "gt"
        gt_feat_dir.mkdir(parents=True, exist_ok=True)
        shared_gt = _shared_gt_feat_dir(self.gt_dir, model)
        linked = _link_shared_features(
            shared_gt,
            gt_feat_dir,
            # docker: link ONLY the 100 scored stems. The scoring itself is safe either way
            # (explicit name lists), but the ref-stats npz is built from whatever this dir
            # holds — extras there would silently enlarge the cached reference distribution.
            names=None if prof["num_images"] is None else set(gt_names),
        )
        if linked:
            log.info(
                "FID GT cache: linked %d shared GT features from %s (forward skipped)",
                linked,
                shared_gt,
            )

        # Reuse prediction features ACROSS PROFILES scored in this same run: "docker" and
        # "docker_n300" score the same predictions/ dir, and docker's stems are always the
        # first N of docker_n300's (both sort it identically) — so whichever profile runs
        # first leaves the other's overlapping predictions already cached here. Scoped to this
        # run's own out_dir (not a cross-run root like GT's), since predictions aren't a stable
        # reference set the way GT is — see `_shared_pred_feat_dir`.
        pred_feat_dir = features_dir / "pred"
        pred_feat_dir.mkdir(parents=True, exist_ok=True)
        shared_pred = _shared_pred_feat_dir(pred_dir, model)
        linked_pred = _link_shared_features(
            shared_pred, pred_feat_dir, names=set(pred_names)
        )
        if linked_pred:
            log.info(
                "FID pred cache: linked %d shared pred features from %s (forward skipped)",
                linked_pred,
                shared_pred,
            )

        # GT-reload fast path. The upstream subprocess is, for us, only a FEATURE
        # extractor: the reported FID is recomputed CPU-side from the precomputed GT (mu, Sigma)
        # refstats npz (fid_from_ref_stats), NOT from the subprocess's real features. But the
        # subprocess's real loop still LOADS + resamples every volume through its DataLoader
        # (only the CNN forward is skipped when features are cached) — hours per run. When valid
        # refstats already exist for this GT set, feed the subprocess a SINGLE GT volume instead:
        # the real loop stays non-empty (its torch.vstack won't crash) but trivial, its own
        # downstream FID is ignored as always, and the pred features it extracts are unchanged
        # -> reported FID is bit-identical. A cold/stale cache falls back to a filtered GT list
        # (see below) so the npz can still bootstrap.
        #
        # npz identity differs per profile. research always scores the COMPLETE GT set, so
        # `n_volumes` identifies it. A truncating profile scores a SUBSET whose membership
        # depends on the run, and `n_volumes == 100` would not distinguish two different
        # 100-volume subsets — so its npz name carries the scored-stem hash.
        stems_sha1 = hashlib.sha1("\n".join(pred_names).encode()).hexdigest()
        ref_stats_path = (
            refstats_path_for(shared_gt)
            if prof["num_images"] is None
            else shared_gt.parent / f"{shared_gt.name}__refstats_{stems_sha1[:12]}.npz"
        )
        refstats_ok = ref_stats_path.is_file() and (
            load_ref_stats(ref_stats_path)["n_volumes"] == len(fid_gt_files)
        )
        if refstats_ok:
            log.info(
                "FID GT-reload fast path: refstats valid (%d vols, %s) — subprocess real loop "
                "cut to 1 GT volume (pred features + CPU FID unchanged).",
                len(fid_gt_files),
                ref_stats_path.name,
            )
            real_names = [gt_names[0]]
        else:
            # Partial-cache filter: whatever the two links above just placed in {gt,pred}_feat_dir
            # doesn't need the (I/O-heavy) load+resample either — only names truly missing locally
            # go to the subprocess. `_fid_from_cached_features` reads by explicit name afterward,
            # so it doesn't matter whether a given file arrived via hardlink or fresh extraction.
            real_names = _missing_names(gt_feat_dir, gt_names) or [gt_names[0]]
        synth_names = _missing_names(pred_feat_dir, pred_names) or [pred_names[0]]
        skipped_gt, skipped_pred = (
            len(gt_names) - len(real_names),
            len(pred_names) - len(synth_names),
        )
        if skipped_gt or skipped_pred:
            log.info(
                "FID partial-cache filter: skipping load+resample for %d/%d GT and %d/%d pred "
                "volumes already cached locally.",
                skipped_gt,
                len(gt_names),
                skipped_pred,
                len(pred_names),
            )
        real_filelist.write_text("\n".join(real_names))
        synth_filelist.write_text("\n".join(synth_names))

        # num_images truncates BOTH real_lines[:n] AND synth_lines[:n] upstream
        # (compute_fid_2-5d_ct.py:556,570).
        #   docker:   pass the profile's own 100 — the filelists already hold exactly that many,
        #             so the truncation is a no-op and the command matches the container.
        #   research: the GT reference set (3001) is larger than the generated set (≤1304), so
        #             this MUST be max(): min() would cap GT extraction at the pred count and, on
        #             a cold cache, leave the 3001-GT reference incomplete → NaN FID that can't
        #             self-bootstrap. Already-cached GT features are skipped via the loop's
        #             os.path.isfile check, so a warm cache pays nothing for the larger count.
        n = (
            prof["num_images"]
            if prof["num_images"] is not None
            else max(len(fid_gt_files), len(fid_pred_files))
        )
        # Use our wrapper that applies the np.float_ → np.float64 shim before
        # importing MONAI's compute_fid_2-5d_ct.py (numpy 2.x compatibility)
        fid_runner = Path("/workspace/src/eval/tasks/_fid_runner.py")
        import socket as _socket

        with _socket.socket() as _s:
            _s.bind(("", 0))
            _free_port = _s.getsockname()[1]
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            f"--master_port={_free_port}",
            str(fid_runner),
            f"--real_dataset_root={self.gt_dir}",
            f"--real_filelist={real_filelist}",
            "--real_features_dir=gt",
            f"--synth_dataset_root={pred_dir}",
            f"--synth_filelist={synth_filelist}",
            "--synth_features_dir=pred",
            f"--model_name={model}",
            f"--num_images={n}",
            f"--output_root={features_dir}",
            f"--target_shape={_FID_TARGET_SHAPE}",
            f"--enable_padding={_FID_ENABLE_PADDING}",
            f"--enable_center_cropping={_FID_ENABLE_CENTER_CROP}",
            f"--enable_resampling_spacing={_FID_RESAMPLE}",
        ]
        log.info("Running FID-2.5D: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=_EVAL_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = result.stdout + "\n" + result.stderr

        # The subprocess is used purely as a GPU feature *extractor*: it writes per-volume
        # features to features_dir/{gt,pred} incrementally (torch.save), then attempts the final
        # FID statistic on the GPU. At full scale that aggregation — ~512k slice-features per
        # plane stacked across 6 matrices, plus an all_gather copy — OOMs the GPU. We therefore
        # always compute the FID ourselves on CPU from those cached features: deterministic,
        # memory-safe (plane-by-plane), ~1-2 min for 1000. This avoids the GPU OOM by design
        # rather than depending on (and parsing) the subprocess's doomed GPU FID. Feature
        # extraction is the only step that needs the GPU.
        # Precomputed GT ref-stats live beside the shared GT feature cache: the fast path loads
        # GT (mu, Sigma) instead of re-stacking the full GT feature set. BOTH profile families
        # pass an npz path (built above) — research one per GT set + model + preprocessing;
        # a truncating profile a sibling npz whose name carries the scored-stem sha1. Its
        # dimensionality follows the feature net (research 2048-d, squeezenet 1000-d), which is
        # why the two never share a file. A missing npz is auto-built here from the cached
        # features, so a cold cache is a one-time cost, not a different code path.
        fid_data = _fid_from_cached_features(
            features_dir, gt_names, pred_names, ref_stats_path=ref_stats_path
        )

        if not fid_data:
            # Incomplete features -> the subprocess died *before* finishing extraction (a real
            # failure, distinct from the expected post-extraction GPU-FID OOM).
            log.error(
                "FID-2.5D: feature extraction did not complete; cannot compute FID.\n%s",
                combined[-2000:],
            )
            return _nan

        # Persist this run's features into their shared caches so the next profile/model reuses
        # them — GT across every run+model forever, predictions within this run's other profiles.
        _populate_shared_features(gt_feat_dir, shared_gt)
        _populate_shared_features(pred_feat_dir, shared_pred)

        # Sidecar carries the full provenance. Historical fid.json files have no "profile" key,
        # which reads as the research profile. `scored_stems_sha1` pins WHICH volumes were
        # scored: the docker profile takes the first 100 sorted prediction stems, so two runs
        # are only comparable when this hash matches (it does across runs generated with the
        # same task.n_samples, since load_eval_cases' subsets are nested).
        fid_data["fid_num_images"] = len(
            pred_names
        )  # numeric -> safe for every metrics.json parser
        out_json = out_dir / "fid.json"
        with open(out_json, "w") as f:
            json.dump(
                {
                    **fid_data,
                    "profile": self.fid_profile,
                    "model_name": model,
                    "num_images": len(pred_names),
                    "scored_stems_sha1": stems_sha1,
                },
                f,
                indent=2,
            )
        return fid_data

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
        """Run a subprocess command and return its exit code.

        Args:
            cmd: command and arguments list passed to ``subprocess.run``.
            cwd: working directory for the subprocess.
            env: environment mapping; inherits the current process environment when None.

        Returns:
            The subprocess return code (0 on success).
        """
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        if result.returncode != 0:
            log.error("Command failed (rc=%d)", result.returncode)
        return result.returncode


__all__ = ["CTGenEvaluator", "resolve_fid_profiles"]
