"""VLM3D evaluation entrypoint (Hydra).

Usage examples
--------------
# Task 4 — Report2CT best checkpoint, valid GT (100 cases), GPU 3
# (model=<name> == a yaml filename under configs/eval/model/)
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct \\
    model.ckpt_path=/workspace/outputs/report2ct/legacy_work_dir/checkpoints/epoch_079_valloss_1.3198.ckpt \\
    model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0 \\ # 필수로 넣어야함
    task.n_samples=100   # 1304가 최대 (spacing_mm/cfg_scale는 report2ct 계열 필수)

# Quick smoke test (1 sample, no CLIPScore)
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct \\
    model.ckpt_path=<ckpt> \\
    model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0 \\
    task.n_samples=1 task.metrics.clip_score=false task.metrics.fid_2p5d=false

# GenerateCT baseline
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=generatect task.n_samples=100

Common args (quick reference)
------------------------------
  task=ctgen model=<name>              <name> = a yaml filename under configs/eval/model/
  model.ckpt_path=<ckpt>               (most models)
  model.spacing_mm=[ip,ip,z] model.cfg_scale=<c>   REQUIRED, no default, for report2ct-family
                                        models (report2ct, report2ct_fvlm, report2ct_clip3d,
                                        report2ct_text2ct*, report2ct_wan*)
  model.cfg_scale_text=<t> model.cfg_scale_mask=<m>  dual-CFG models only (report2ct_wan_mask_v2)
  task.n_samples=<n>                   null = all ~1304 (valid_v2)
  task.metrics.{fvd,fvd_ctclip,clip_score,fid_2p5d}=true|false   existing VLM3D metrics
  task.fid_profile=<name>|[<name>,...]  which 2.5D-FID metric family/families. DEFAULT
                                        [docker,docker_n300] = both, each into its own
                                        <out_dir>/fid_<profile>/, plus a combined top-level
                                        summary.json. `docker` reproduces the official container
                                        (squeezenet1_1 over the first 100 predictions);
                                        `docker_n300` is docker's network at 300 volumes (needed
                                        by default so subgroup analysis always has it cached);
                                        `research` is the pre-2026-07-29 setting
                                        (radimagenet_resnet50, all predictions vs the 3001-GT),
                                        opt-in only — REQUIRED to compare against any FID
                                        recorded before then or the paper envelope.
                                        Affects FID only; CLIP/FVD are identical either way and
                                        are computed ONCE (first profile) then copied.
                                        To score an EXISTING run under another profile without
                                        regenerating, use scripts/rescore_predictions.py (it writes to
                                        <eval_dir>/fid_<profile>/ and never touches metrics.json).
  task.metrics.{per_sample,dice,hd95,subgroup,subgroup_setlevel}=true|false   NEW (all default
                                        false — additive; see docs below)
  task.qc_figures=true|false           NEW, default true (viz-only; task.qc_n=<n>, default 5)
  task.seg_backend=totalseg|vista3d    NEW, default totalseg (dice/hd95 backend)
  task.is_mask_model=true              NEW — set for report2ct_wan_mask(_v2)/
                                        report2ct_text2ct_mask so dice_to_input_mask is populated
  task.manifest=<jsonl>                NEW — mask-intervention run (docs/mask_intervention_manifest.md):
                                        cases come from the manifest ROWS, so predictions are named
                                        <sample_id>.mha and one target appears several times (one per
                                        conditioning-mask condition). The POOLED task.metrics.{fid_2p5d,
                                        fvd,fvd_ctclip} are REFUSED for such a run (repeated target
                                        breaks their premise) — use task.metrics.per_sample/dice for
                                        CLIP-T2I/Dice, and task.metrics.condition_fid=true for FID/FVD
                                        scored PER CONDITION (valid; lands in this run's SUMMARY.md).

New-flag examples
------------------
# Per-sample CLIP + subgroup summaries (cheap; no segmentation)
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct model.ckpt_path=<ckpt> \\
    model.spacing_mm=[0.757,0.757,1.5] model.cfg_scale=5.0 task.n_samples=300 \\
    task.metrics.per_sample=true task.metrics.subgroup=true

# Full per-sample analysis for a MASK-conditioned model: per-sample CLIP + Dice/HD95
# (dice_to_input_mask meaningful here) + subgroup + subgroup set-level FID
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct_wan_mask out_dir=<OUT> task.n_samples=300 \\
    task.is_mask_model=true \\
    task.metrics.per_sample=true task.metrics.dice=true task.metrics.hd95=true \\
    task.metrics.subgroup=true task.metrics.subgroup_setlevel=true

# Manual eval-folder naming convention (encodes eval-set/n/epoch/spacing/cfg; NOT applied
# automatically — dated eval_<KST-date> is still the default. Override via out_dir=, which
# Hydra's run dir follows; setting hydra.run.dir= instead splits the two and is refused):
CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py \\
    task=ctgen model=report2ct model.ckpt_path=<ckpt> \\
    model.spacing_mm=[0.757,0.757,1.5] model.cfg_scale=5.0 task.n_samples=300 \\
    out_dir=/workspace/outputs/report2ct/eval_toy_v2_n300_ep099_sp0.757_1.5_cfg5
    # dual-CFG models: use ..._cfgt5_cfgm1.0 instead of ..._cfg5 (two guidance scales)
    # An EXISTING eval dir is refused when its metrics.json holds another fid_profile —
    # re-score with scripts/rescore_predictions.py instead (it writes to <dir>/fid_<profile>/).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import resolvers  # noqa: F401  — registers OmegaConf resolvers (dated_rundir, path_leaf)

log = logging.getLogger(__name__)


def _get_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cfg.device)


def _refuse_split_run_dir(cfg: DictConfig, run_dir: str) -> None:
    """Stop when the results dir and Hydra's run dir are not the same folder.

    ``configs/eval/default.yaml`` wires ``hydra.run.dir: ${out_dir}``, so overriding ``out_dir=``
    moves both. Overriding ``hydra.run.dir=`` instead splits them the other way: metrics.json /
    predictions/ land in the dated default while ``.hydra/`` + run_eval.log land elsewhere. A split
    run dir is not wrong by itself — it is wrong because the results then carry no record of the
    command that produced them, and a results dir can end up holding a ``.hydra/`` from a DIFFERENT
    run. That is what hid the origin of the 2026-07-29 clobber for a day (docs/ctgen_local_eval.md).

    Task-agnostic, so it lives in ``main()`` rather than in a per-task runner.

    Args:
        run_dir: ``HydraConfig.get().runtime.output_dir`` — where .hydra/ and the log go.
    """
    out, run = Path(cfg.out_dir).resolve(), Path(run_dir).resolve()
    if out == run:
        return
    raise RuntimeError(
        f"out_dir and Hydra's run dir point at different folders — results would be written "
        f"without the .hydra/ + log that document them:\n"
        f"  results  (out_dir):      {out}\n"
        f"  .hydra/ + run_eval.log:  {run}\n"
        f"Override `out_dir=<dir>` (Hydra's run dir follows it); do not set `hydra.run.dir=`."
    )


# Metrics that depend on the predictions alone, not on the FID feature network — computed in the
# first profile's pass and copied into the rest (see the loop in _run_ctgen).
_PROFILE_INDEPENDENT_METRICS = ("fvd", "fvd_ctclip", "clip_score")

# Set-level metrics whose premise a mask-intervention run breaks (see _refuse_setlevel_metrics).
_SET_LEVEL_METRICS = ("fid_2p5d", "fvd", "fvd_ctclip")


def _refuse_setlevel_metrics(cfg: DictConfig) -> None:
    """Stop a manifest run from being scored with the POOLED FID/FVD default.

    Those metrics compare two *distributions* under a "1 volume = 1 independent sample" premise.
    A default fid_2p5d/fvd/fvd_ctclip pass scores the WHOLE predictions/ dir as one pool, and a
    mask-intervention run generates the SAME target several times in there (once per condition),
    so that pool is neither independent nor a sample of the CT-RATE distribution, and the GT
    reference would be counted once per repeat. The numbers would still compute — which is
    exactly why this refuses rather than warns.

    This is NOT a blanket "FID is invalid for manifest runs" claim: a SINGLE condition's subset
    (e.g. just the label_mismatched_swap volumes) has no repeated target and is exactly as valid
    as any other model's plain-run FID. That per-condition scoring is
    ``task.metrics.condition_fid=true`` (folded into this same run — see step 5b in
    ``_run_ctgen``) or, for a run already scored without it, the post-hoc
    ``scripts/score_condition_fid.py`` (both call ``src/eval/analysis/condition_setlevel.py``).
    The remaining, always-valid, same-run diagnostics are CLIPScore-T2I, dice_to_input_mask,
    dice_to_gt_mask.
    """
    if not cfg.task.get("manifest"):
        return
    requested = [m for m in _SET_LEVEL_METRICS if cfg.task.metrics.get(m, False)]
    if requested:
        raise RuntimeError(
            f"task.manifest is set, so this is a mask-intervention run — {requested} cannot be "
            "scored POOLED on it (one target has several generated volumes, breaking the "
            "'1 volume = 1 independent sample' premise those metrics rest on). Pass "
            + " ".join(f"task.metrics.{m}=false" for m in requested)
            + " and use the per-sample diagnostics (task.metrics.per_sample=true, "
            "task.metrics.dice=true) plus, for FID/FVD compared BY CONDITION, "
            "task.metrics.condition_fid=true (folded into this same run) or "
            "scripts/score_condition_fid.py on an already-scored run."
        )


def _is_fid_key(key: str) -> bool:
    """True for the metrics.json keys whose meaning depends on the FID profile.

    These are the keys that must NOT be copied between profile folders — they are the whole
    reason the folders are separate — and that the combined summary puts under
    ``fid.<profile>`` rather than beside the profile-independent metrics.
    """
    return key.startswith("FID_2p5D") or key in ("fid_profile", "fid_num_images")


class _Scored(NamedTuple):
    """What one multi-profile scoring pass produced.

    Attributes:
        profiles: the profiles scored, in the order they ran.
        shared: the profile-independent metrics (FVD / FVD_CTCLIP / CLIP), computed once.
        fid_by_profile: profile name → that profile's FID keys only.
        results_by_profile: profile name → everything recorded in that profile's folder.
    """

    profiles: list[str]
    shared: dict
    fid_by_profile: dict[str, dict]
    results_by_profile: dict[str, dict]


def _score_profiles(
    *,
    fid_profiles: list[str],
    score_dirs: list[Path],
    pred_dir: Path,
    gt_dir: Path,
    metrics_cfg: dict,
    ctclip_ckpt,
    prompt_xlsx,
    clip_agg: dict | None,
) -> _Scored:
    """Score the predictions once per FID profile, each into its own ``fid_<profile>/`` folder.

    FVD / FVD_CTCLIP / CLIP depend only on the predictions, not on the FID feature network, so
    only the FIRST profile's pass computes them (~24 min of GPU time at n=300) and the later
    passes get the identical values copied in. Copying rather than omitting keeps every
    ``fid_<profile>/metrics.json`` self-contained, which is what every aggregator already
    assumes; the raw sidecars (clip.json / fvd_ctclip.json) stay with the pass that wrote them.

    Args:
        clip_agg: aggregate CLIP from the per-sample runner, or None. Merged into the first
            profile's folder (and thus copied onward like any other shared metric).
    """
    from src.eval.tasks.ctgen import CTGenEvaluator, _merge_metrics

    shared: dict = {}
    fid_by_profile: dict[str, dict] = {}
    results_by_profile: dict[str, dict] = {}
    for i, (profile, score_dir) in enumerate(zip(fid_profiles, score_dirs)):
        pass_metrics = dict(metrics_cfg)
        if i:
            for key in _PROFILE_INDEPENDENT_METRICS:
                pass_metrics[key] = False
        evaluator = CTGenEvaluator(
            gt_dir=gt_dir,
            metrics=pass_metrics,
            ctclip_ckpt=ctclip_ckpt,
            prompt_xlsx=prompt_xlsx,
            fid_profile=profile,
        )
        results = evaluator.evaluate(pred_dir, score_dir)
        if i == 0:
            if clip_agg is not None:
                # Merge (never re-dump): a plain write here would drop `_history` and, when the
                # per-sample runner supplied CLIP, everything evaluate() just recorded.
                results = _merge_metrics(score_dir, clip_agg, ["clip_score_persample"])
            shared = {k: v for k, v in results.items() if not _is_fid_key(k)}
        else:
            results = _merge_metrics(
                score_dir, shared, [f"copied_from_{fid_profiles[0]}"]
            )
        results_by_profile[profile] = results
        fid_by_profile[profile] = {k: v for k, v in results.items() if _is_fid_key(k)}

    return _Scored(fid_profiles, shared, fid_by_profile, results_by_profile)


def _write_summary(path: Path, payload: dict) -> None:
    """Write one summary.json and log where it went."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Summary saved to %s", path)


def _write_summaries(
    out_dir: Path, score_dirs: list[Path], model_name: str, scored: _Scored
) -> None:
    """Write each profile's summary.json plus the combined top-level one.

    ``fid_<profile>/summary.json`` keeps its original schema (a flat ``metrics`` dict for that
    profile alone). The top-level file collects the whole run: the profile-independent metrics
    once, and every profile's FID keyed BY PROFILE. Nesting is what makes a top-level file safe
    here — the ambiguity the ``fid_<profile>/`` layout exists to prevent is an UNLABELLED FID
    (the 2026-07-29 clobber), and under ``fid.<profile>`` no number is unlabelled.
    """
    for profile, score_dir in zip(scored.profiles, score_dirs):
        _write_summary(
            score_dir / "summary.json",
            {
                "task": "ctgen",
                "model": model_name,
                "metrics": scored.results_by_profile[profile],
            },
        )
    _write_summary(
        out_dir / "summary.json",
        {
            "task": "ctgen",
            "model": model_name,
            "fid_profiles": scored.profiles,
            "metrics": scored.shared,
            "fid": scored.fid_by_profile,
        },
    )


def _run_ctgen(cfg: DictConfig) -> dict:
    """Task 4: CT generation pipeline (generate → valid GT → evaluate)."""
    from src.eval.ct_rate_cases import (
        load_eval_cases,
        load_manifest_cases,
        prepare_valid_gt,
        write_prompt_xlsx,
    )
    from src.eval.tasks.ctgen import _DEFAULT_SUBGROUP_FID_PROFILE, resolve_fid_profiles

    _refuse_setlevel_metrics(cfg)
    device = _get_device(cfg)
    out_dir = Path(cfg.out_dir)
    pred_dir = out_dir / "predictions"
    gt_dir = Path(cfg.task.gt_dir) if cfg.task.gt_dir else out_dir / "gt"

    # Each profile's metric artifacts land in their OWN profile-named folder — the same place
    # rescore_predictions.py writes. `docker` and `research` put the SAME FID_2p5D_* keys on
    # incomparable scales, so keeping them in separate folders makes a cross-profile clobber
    # structurally impossible instead of something a guard has to catch (it is what destroyed
    # two report2ct_wan_mask_v2 research FIDs on 2026-07-29 — docs/ctgen_local_eval.md).
    # Run-level artifacts (predictions/, latents/, analysis/, prompts.xlsx, .hydra/) are shared
    # across profiles and stay at the top; so does the combined summary.json written at the end.
    fid_profiles = resolve_fid_profiles(cfg.task.get("fid_profile"))
    score_dirs = [out_dir / f"fid_{p}" for p in fid_profiles]
    for d in score_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load eval cases — one per scan, or one per manifest ROW for a mask-intervention run
    # (same target several times, each with its own conditioning mask and output filename).
    manifest_path = cfg.task.get("manifest")
    if manifest_path:
        cases = load_manifest_cases(manifest_path)
    else:
        n = cfg.task.n_samples if cfg.task.n_samples else None
        cases = load_eval_cases(n_samples=n)
    if not cases:
        raise RuntimeError("No eval cases found — check CT-RATE dataset path.")
    log.info("Evaluating %d cases", len(cases))

    # 2. Prepare valid GT (idempotent)
    if cfg.task.gt_mode == "valid":
        # One GT per TARGET scan: a manifest run lists the same target once per condition, and
        # prepare_valid_gt is keyed by scan_id (identical list for a plain run, whose cases are
        # already unique).
        prepare_valid_gt(list({c.scan_id: c for c in cases}.values()), gt_dir)
    elif cfg.task.gt_mode == "official":
        if not gt_dir.is_dir() or not any(gt_dir.glob("*.mha")):
            raise FileNotFoundError(
                f"official gt_dir={gt_dir} is empty or missing. "
                "Extract from docker: docker cp <cid>:/opt/app/ground-truth/ ."
            )

    # 2a2. GT view for a manifest run. Upstream CLIP (and our QC figures) find a prediction's GT
    # by its FILENAME STEM, which is a sample_id here — a name the GT dir does not contain.
    # third_party/ is read-only, so they get a directory of symlinks to the same GT volumes under
    # the sample_id names instead (each link points at the row's TARGET). Everything that joins on
    # target_id — prepare_valid_gt above, run_metrics_analysis below — keeps the real gt_dir.
    # Safe to hand to CTGenEvaluator because FID/FVD are refused on a manifest run.
    clip_gt_dir = gt_dir
    if manifest_path:
        from src.eval.manifest import build_gt_view

        clip_gt_dir = build_gt_view(cases, gt_dir, out_dir / "gt_view")

    # 2b. Auto-write prompt XLSX for CLIPScore (valid mode only)
    prompt_xlsx = cfg.task.get("prompt_xlsx")
    if (
        prompt_xlsx is None
        and cfg.task.gt_mode == "valid"
        and cfg.task.metrics.get("clip_score", False)
    ):
        prompt_xlsx = out_dir / "prompts.xlsx"
        write_prompt_xlsx(cases, prompt_xlsx)

    # 3. Generate predictions
    sampler = hydra.utils.instantiate(cfg.model)
    sampler.generate(cases, pred_dir, device)

    # 3b. QC figures — independent of every metric flag below (viz-only, never touches
    # metrics.json/summary.json; renders even when all task.metrics.* are false). See
    # plan §3.10/§3.11 (repository-run-eval-jazzy-twilight.md).
    # Wrapped in try/except: a viz failure must never cost the run its metrics on
    # already-generated predictions (code review finding, 2026-07-27).
    if cfg.task.get("qc_figures", True):
        from src.eval.analysis.orchestrate import run_qc_figures

        try:
            run_qc_figures(
                pred_dir=pred_dir,
                gt_dir=clip_gt_dir,
                out_dir=out_dir,
                model_name=cfg.model.name,
                n=cfg.task.get("qc_n", 5),
            )
        except Exception:
            log.exception(
                "QC figures failed — continuing with metric evaluation (viz-only)."
            )

    # 4. Evaluate (existing VLM3D metrics — CTGenEvaluator itself is unmodified)
    metrics_cfg = OmegaConf.to_container(cfg.task.metrics, resolve=True)
    per_sample_flag = metrics_cfg.get("per_sample", False)

    # 4a. Per-sample CLIP routing: when per_sample + clip_score are both requested, run CLIP
    # ONCE via the per-sample runner (same model, identical aggregate formula) and tell
    # CTGenEvaluator to skip its own CLIP subprocess — a code-enforced dedup, not a manual
    # `clip_score=false` toggle the user has to remember (plan §3.10).
    clip_scores = None
    clip_agg = None
    if per_sample_flag and metrics_cfg.get("clip_score", False):
        from src.eval.analysis import clip_persample

        clip_scores, clip_agg = clip_persample.run(
            pred_dir=pred_dir,
            gt_dir=clip_gt_dir,
            prompt_xlsx=prompt_xlsx,
            out_dir=out_dir,
            ctclip_ckpt=cfg.task.get("ctclip_ckpt"),
        )
        metrics_cfg["clip_score"] = False

    # 4b. One scoring pass per FID profile, each into its own fid_<profile>/ folder.
    scored = _score_profiles(
        fid_profiles=fid_profiles,
        score_dirs=score_dirs,
        pred_dir=pred_dir,
        gt_dir=clip_gt_dir,
        metrics_cfg=metrics_cfg,
        ctclip_ckpt=cfg.task.get("ctclip_ckpt"),
        prompt_xlsx=prompt_xlsx,
        clip_agg=clip_agg,
    )

    # The analysis extension below is profile-independent and runs once, so point its
    # SUMMARY.md headline at the first profile's metrics.json.
    primary_dir = score_dirs[0]

    # 5. Per-sample / subgroup / segmentation analysis extension (additive; all-off by default
    # -> a no-op, byte-identical metrics.json/summary.json to before this extension existed).
    from src.eval.analysis.orchestrate import run_metrics_analysis

    run_metrics_analysis(
        scan_ids=[c.scan_id for c in cases],
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        out_dir=out_dir,
        subgroup_config_path=cfg.task.get("subgroup_config"),
        per_sample=per_sample_flag,
        dice=metrics_cfg.get("dice", False),
        hd95=metrics_cfg.get("hd95", False),
        subgroup_flag=metrics_cfg.get("subgroup", False),
        subgroup_setlevel_flag=metrics_cfg.get("subgroup_setlevel", False),
        is_mask_model=cfg.task.get("is_mask_model", False),
        manifest_path=cfg.task.get("manifest"),
        seg_backend=cfg.task.get("seg_backend", "totalseg"),
        seg_max_n=cfg.task.get("seg_max_n"),
        clip_scores=clip_scores,
        subgroup_setlevel_compute_fvd=cfg.task.get(
            "subgroup_setlevel_compute_fvd", False
        ),
        subgroup_fid_profile=cfg.task.get(
            "subgroup_fid_profile", _DEFAULT_SUBGROUP_FID_PROFILE
        ),
        metrics_path=primary_dir / "metrics.json",
    )

    # 5b. FID/FVD BY CONDITION (manifest runs only, opt-in) — the sanctioned alternative to the
    # pooled fid_2p5d/fvd/fvd_ctclip _refuse_setlevel_metrics blocks above: each condition's
    # OWN subset has no repeated target, so it is exactly as valid as any plain run's FID.
    # Folded into this same invocation (rather than left to the standalone
    # scripts/score_condition_fid.py) so `task.metrics.condition_fid=true` is the only thing
    # needed for the numbers to show up in this run's SUMMARY.md — no separate command.
    if manifest_path and metrics_cfg.get("condition_fid", False):
        from src.eval.analysis import condition_setlevel
        from src.eval.analysis.orchestrate import analysis_paths

        per_sample_csv = analysis_paths(out_dir)["per_sample_csv"]
        if not per_sample_csv.is_file():
            raise RuntimeError(
                "task.metrics.condition_fid=true needs analysis/per_sample.csv (for the "
                "sample_id/condition columns) — pass task.metrics.per_sample=true too."
            )
        condition_setlevel.run(
            pred_dir=pred_dir,
            gt_view_dir=clip_gt_dir,  # this run's gt_view/ (GT-by-stem, see build_gt_view)
            per_sample_csv=per_sample_csv,
            out_dir=out_dir / "condition_fid",
            fid_profile=cfg.task.get(
                "subgroup_fid_profile", _DEFAULT_SUBGROUP_FID_PROFILE
            ),
            ctclip_ckpt=cfg.task.get("ctclip_ckpt"),
        )

    # 6. Summaries — one per profile folder plus the combined top-level one.
    _write_summaries(out_dir, score_dirs, cfg.model.name, scored)

    return {**scored.shared, "fid": scored.fid_by_profile}


_TASK_RUNNERS = {
    "ctgen": _run_ctgen,
}


@hydra.main(
    config_path="../configs/eval",
    config_name="default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    _refuse_split_run_dir(cfg, HydraConfig.get().runtime.output_dir)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    task_name = cfg.task.name
    runner = _TASK_RUNNERS.get(task_name)
    if runner is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(_TASK_RUNNERS.keys())}"
        )

    results = runner(cfg)

    # Pretty-print final metrics
    print("\n" + "=" * 60)
    print(f"  VLM3D Task 4 ({task_name}) — {cfg.model.name}")
    print("=" * 60)
    for k, v in results.items():
        # ctgen returns the FID keys nested under `fid` (one block per profile), because the
        # same FID_2p5D_* names carry incomparable values across profiles.
        if isinstance(v, dict):
            for profile, block in v.items():
                print(f"  [{k}: {profile}]")
                for kk, vv in block.items():
                    print(
                        f"    {kk:<23s} {vv:.4f}"
                        if isinstance(vv, float)
                        else f"    {kk}: {vv}"
                    )
            continue
        print(f"  {k:<25s} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
