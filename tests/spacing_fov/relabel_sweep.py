#!/usr/bin/env python3
"""E4 — FID as a function of the DECLARED spacing, with the voxels held fixed.

Nothing is regenerated and no file is rewritten: the same prediction volumes are pushed
through the FID chain repeatedly, each time pretending the header declared a different
spacing. Any change in FID is therefore attributable purely to the declaration
(hypothesis A) — the pixels are byte-identical across every point on the curve.

Run it on both content sets (`gen_sp0.8` and `gen_sp1.0`) and the two curves answer the
question the optimum alone cannot: if both minimise at the same declared spacing, the
optimum is a property of the declaration; if they minimise at different places, the
conditioning changed the content in a way that shifts it.

Feature extraction reuses the upstream `get_features_2p5d` (loaded from the read-only
docker script), and the FID statistic is computed exactly as `CTGenEvaluator` does it —
plane-by-plane on CPU with MONAI's `FIDMetric`. GT features come from the production
shared cache, so they are the very same tensors the reported numbers were built from.

**Gate**: `--gate <FID_Avg>` checks that the config matching how the volumes were actually
generated reproduces the production number. Without that agreement the curve means nothing.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/relabel_sweep.py --content gen_sp0.8
    CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/relabel_sweep.py --content gen_sp1.0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# MONAI's fid._sqrtm uses np.float_, removed in NumPy 2 — shim before anything imports it
# (same fix as src/eval/tasks/_fid_runner.py:16-17).
if not hasattr(np, "float_"):
    np.float_ = np.float64

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/workspace")

from src.eval._vlm3d_paths import ctgen_eval_dir  # noqa: E402
from src.eval.tasks.ctgen import (  # noqa: E402
    _RESEARCH_FID_MODEL,
    _shared_gt_feat_dir,
)
from tests.spacing_fov._common import fid_preproc  # noqa: E402

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs"
RESULTS = HERE / "results"
FEATCACHE = HERE / "_sweep_featcache"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_lps_reeval/gt_lps_seed100")

# (in-plane mm, z mm). z=1.5 is how both arms were generated; the 1.31 rows probe whether the
# z declaration matters independently (training median z is 1.318).
CONFIGS: list[tuple[float, float]] = [
    (0.65, 1.5),
    (0.70, 1.5),
    (0.757, 1.5),
    (0.80, 1.5),
    (0.85, 1.5),
    (0.90, 1.5),
    (1.00, 1.5),
    (1.10, 1.5),
    (0.757, 1.31),
    (0.80, 1.31),
    (0.90, 1.31),
    (0.77, 1.5),  # index 11 — training MEAN in-plane (0.774), between 0.757 and 0.8
    (0.77, 1.31),  # index 12
    # z-sweep at the in-plane optimum (0.8), to locate the FID z-minimum. Predicted ~1.32
    # (GT z-FOV median 339 mm / 256 = 1.324). Existing 0.8_1.31 and 0.8_1.5 fill the rest.
    (0.8, 1.2),  # index 13
    (0.8, 1.3),  # index 14
    (0.8, 1.35),  # index 15
    (0.8, 1.4),  # index 16
    (0.8, 1.6),  # index 17
    (
        0.8,
        1.7,
    ),  # index 18 — high-z tail (FID keeps rising; measured to show the CLIP trade)
    (0.8, 1.8),  # index 19
    (0.8, 2.0),  # index 20
]
PREFETCH = 6  # preprocessed 512^3 float32 volumes held in flight (~0.5 GB each)


def _load_upstream_features_fn():
    """Load `get_features_2p5d` from the read-only docker FID script.

    `runpy.run_path` with a non-``__main__`` run name executes the module body (imports,
    function defs) without triggering its ``fire.Fire(main)`` entrypoint. The filename
    contains a hyphen, so a normal import statement is not an option.
    """
    import logging  # noqa: PLC0415

    script = ctgen_eval_dir() / "compute_fid_2-5d_ct.py"
    ns = runpy.run_path(str(script), run_name="_fid_upstream")
    # The upstream helper logs 2 INFO lines per plane per volume; at 11 configs x 100 volumes
    # that is ~3300 lines of noise.
    logging.getLogger("fid_2-5d_ct").setLevel(logging.WARNING)
    return ns["get_features_2p5d"]


def _build_feature_network(device: torch.device):
    """The same RadImageNet ResNet-50 the production FID uses (compute_fid_2-5d_ct.py:515-519)."""
    net = torch.hub.load(
        "Warvito/radimagenet-models",
        model="radimagenet_resnet50",
        verbose=False,
        trust_repo=True,
    )
    return net.to(device).eval()


def _preproc_job(args: tuple[str, float, float]) -> np.ndarray:
    """Worker: FID-preprocess one volume under a substituted spacing. Returns ``(512,512,512)``."""
    path, sx, sz = args
    return fid_preproc(path, spacing_override=(sx, sx, sz))


def _features_for_config(
    files: list[pathlib.Path],
    sx: float,
    sz: float,
    get_features_2p5d,
    net,
    device: torch.device,
    workers: int,
) -> list[pathlib.Path]:
    """Extract (and cache) 2.5D features for every file under one declared spacing.

    Returns the list of cached ``.pt`` paths, one per input file.
    """
    cache = FEATCACHE / f"sp{sx}_{sz}__{files[0].parent.name}"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (cache / f"{f.stem}.pt").is_file()]
    if todo:
        print(f"  extracting {len(todo)}/{len(files)} → {cache.name}")
        # "spawn", not the default fork: this process has already initialised a CUDA context
        # (the feature network is on the GPU), and forking that context deadlocks the children.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            pending: list = []
            it = iter(todo)
            for f in it:
                pending.append((f, ex.submit(_preproc_job, (str(f), sx, sz))))
                if len(pending) < PREFETCH:
                    continue
                _drain_one(pending, cache, get_features_2p5d, net, device)
            while pending:
                _drain_one(pending, cache, get_features_2p5d, net, device)
    return [cache / f"{f.stem}.pt" for f in files]


def _ensure_gt_features(
    gt_dir: pathlib.Path,
    cache: pathlib.Path,
    get_features_2p5d,
    net,
    device: torch.device,
    workers: int,
) -> None:
    """Extract GT 2.5D features into ``cache`` at the GT's NATIVE spacing, if not present.

    Mirrors the prediction feature path exactly (same fid_preproc + upstream feature fn),
    with ``spacing_override=None`` so each GT volume keeps its real header. Idempotent: a
    volume already in the cache is skipped, so this reuses a prior production run's features.
    """
    gt_vols = sorted(gt_dir.glob("*.mha"))
    if not gt_vols:
        sys.exit(f"no GT volumes in {gt_dir}")
    cache.mkdir(parents=True, exist_ok=True)
    todo = [f for f in gt_vols if not (cache / f"{f.stem}.pt").is_file()]
    if not todo:
        return
    print(f"[E4] extracting GT features {len(todo)}/{len(gt_vols)} → {cache.name}")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        pending: list = []
        for f in todo:
            # native spacing: pass the sentinel by reusing _preproc_job with the real header
            pending.append((f, ex.submit(_preproc_native_job, str(f))))
            if len(pending) >= PREFETCH:
                _drain_one(pending, cache, get_features_2p5d, net, device)
        while pending:
            _drain_one(pending, cache, get_features_2p5d, net, device)


def _preproc_native_job(path: str) -> np.ndarray:
    """Worker: FID-preprocess one volume at its native declared spacing. ``(512,512,512)``."""
    return fid_preproc(path, spacing_override=None)


def _drain_one(
    pending: list, cache: pathlib.Path, get_features_2p5d, net, device
) -> None:
    """Pop the oldest pending preprocessing job, run the GPU forward, and cache its features."""
    f, fut = pending.pop(0)
    vol = fut.result()  # (512, 512, 512)
    img = torch.from_numpy(vol)[None, None].to(device)  # (1, 1, 512, 512, 512)
    feats = get_features_2p5d(img, net, center_slices=False, xy_only=False)
    # Saved on the device they were produced on, exactly like the upstream loop
    # (compute_fid_2-5d_ct.py:640) — the cached GT features are CUDA tensors, and mixing a
    # CPU-saved prediction with them makes FIDMetric raise a device mismatch.
    torch.save(feats, cache / f"{f.stem}.pt")
    del img, vol


def _fid_from_features(
    pred_files: list[pathlib.Path], gt_files: list[pathlib.Path]
) -> dict[str, float]:
    """Plane-wise FID on CPU — mirrors `src/eval/tasks/ctgen.py::_fid_from_cached_features`."""
    from monai.metrics import FIDMetric  # noqa: PLC0415

    # Load onto a fixed device rather than trusting each file's save-time device: a stray file
    # written by a different run (e.g. a CPU-only smoke test) sharing this cache key would
    # otherwise raise a device-mismatch error deep inside vstack/FIDMetric.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    keys = {0: "FID_2p5D_XY", 1: "FID_2p5D_YZ", 2: "FID_2p5D_XZ"}
    out: dict[str, float] = {}
    for idx, key in keys.items():
        real = torch.vstack(
            [
                torch.load(f, weights_only=True, map_location=device)[idx].float()
                for f in gt_files
            ]
        )
        synth = torch.vstack(
            [
                torch.load(f, weights_only=True, map_location=device)[idx].float()
                for f in pred_files
            ]
        )
        out[key] = float(FIDMetric()(synth, real))
        del real, synth
    out["FID_2p5D_Avg"] = sum(out[k] for k in keys.values()) / 3.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True, help="prediction dir name under preds/")
    ap.add_argument(
        "--gt-dir",
        default=str(GT_DIR),
        help="GT directory. Its FID feature cache (keyed on the dir name) is extracted on "
        "first use if empty — so a fresh GT set (e.g. an n=300 build) works out of the box.",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--files-limit", type=int, default=None, help="smoke run: use N volumes"
    )
    ap.add_argument(
        "--configs-limit",
        type=int,
        default=None,
        help="smoke run: use the first N spacings",
    )
    ap.add_argument(
        "--configs",
        type=int,
        nargs="*",
        default=None,
        help="indices into CONFIGS to run — lets the sweep be split across GPUs",
    )
    ap.add_argument(
        "--config-pairs",
        nargs="*",
        default=None,
        metavar="SX,SZ",
        help="explicit 'inplane,z' spacings to sweep (overrides CONFIGS) — for a model whose "
        "grid differs from report2ct's, e.g. Wan's 512^2 grid wants a different optimum",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="plumbing check only: score fewer predictions than there are GT volumes. The "
        "resulting FID is on its own scale and must not be plotted or reported.",
    )
    ap.add_argument(
        "--gate-source",
        default=None,
        help="where the --gate number came from (e.g. a production metrics.json path), "
        "recorded alongside the result so the certification is auditable",
    )
    ap.add_argument(
        "--gate",
        type=float,
        default=None,
        help="production FID_2p5D_Avg for this content set at its native declared spacing",
    )
    ap.add_argument(
        "--gate-config",
        type=float,
        nargs=2,
        default=None,
        metavar=("SX", "SZ"),
        help="the declared spacing the content set was generated with (for --gate)",
    )
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    pred_dir = HERE / "preds" / args.content
    files = sorted(pred_dir.glob("*.mha"))
    if args.files_limit:
        files = files[: args.files_limit]
    if not files:
        sys.exit(f"no volumes in {pred_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_features_2p5d = _load_upstream_features_fn()
    net = _build_feature_network(device)

    # GT features: extract into the shared cache (keyed on the GT dir name) if not already
    # there, so a fresh GT set works without a prior production run. Predictions and GT go
    # through the IDENTICAL preprocessing + feature path, only GT uses native spacing.
    gt_dir = pathlib.Path(args.gt_dir).resolve()
    # Pinned to research: _build_feature_network() below builds the RadImageNet ResNet50
    # directly, so the cache key must name that same network.
    gt_cache = _shared_gt_feat_dir(gt_dir, _RESEARCH_FID_MODEL)
    _ensure_gt_features(gt_dir, gt_cache, get_features_2p5d, net, device, args.workers)

    # Use the WHOLE cached GT feature set, exactly like production: `_run_fid` hardlinks every
    # cached GT feature into the run dir and `_fid_from_cached_features` globs that directory
    # (ctgen.py:498, :130), so the real distribution is always the full GT set regardless of how
    # many predictions there are. Selecting GT by prediction stem instead would silently compare
    # against a smaller real distribution, putting the curve on a different scale from every
    # reported number — and from any other curve plotted beside it.
    gt_files = sorted(p for p in gt_cache.iterdir() if p.is_file())
    if not gt_files:
        sys.exit(f"no cached GT features in {gt_cache}")
    if len(files) != len(gt_files) and not args.allow_partial:
        sys.exit(
            f"refusing to run: {len(files)} predictions vs {len(gt_files)} GT volumes.\n"
            "A partial prediction set gives a FID that is comparable to nothing — not to the "
            "production numbers, not to a full-set curve. Wait for generation to finish, or "
            "pass --allow-partial for a throwaway plumbing check."
        )
    print(
        f"[E4] {len(files)} predictions from {pred_dir.name}, GT features from {gt_cache.name}"
    )

    if args.config_pairs is not None:
        selected = [tuple(float(x) for x in p.split(",")) for p in args.config_pairs]
    elif args.configs is not None:
        selected = [CONFIGS[i] for i in args.configs]
    else:
        selected = CONFIGS[: args.configs_limit]

    # Merge with any earlier partial run for this content set, so the sweep can be split
    # across GPUs (each invocation handles a subset of --configs) and so a crash mid-sweep
    # (e.g. a stray cache file from an unrelated run) does not lose configs already computed.
    suffix = "__PARTIAL" if len(files) != len(gt_files) else ""
    out_path = RESULTS / f"sweep_fid__{args.content}{suffix}.json"
    out = {
        "content": args.content,
        "n_pred": len(files),
        "n_gt": len(gt_files),
        "results": {},
    }
    if out_path.is_file():
        prev = json.loads(out_path.read_text())
        if prev.get("n_pred") == len(files) and prev.get("n_gt") == len(gt_files):
            out["results"] = prev.get("results", {})
            # Carry the gate verdict forward: a split run whose --configs subset excludes the
            # gate config must not erase a previously passing certification.
            if "gate" in prev:
                out["gate"] = prev["gate"]
        else:
            print(
                f"[E4] discarding previous results for {out_path.name}: they were computed on "
                f"{prev.get('n_pred')} pred / {prev.get('n_gt')} GT volumes, not "
                f"{len(files)} / {len(gt_files)}"
            )

    for sx, sz in selected:
        print(f"[E4] declared spacing ({sx}, {sx}, {sz})")
        pred_feat = _features_for_config(
            files, sx, sz, get_features_2p5d, net, device, args.workers
        )
        out["results"][f"{sx}_{sz}"] = _fid_from_features(pred_feat, gt_files)
        print(f"       FID Avg {out['results'][f'{sx}_{sz}']['FID_2p5D_Avg']:.4f}")
        # Save after EVERY config, not just at the end — a later config (e.g. a stray-cache
        # device mismatch) must not cost the configs already computed in this run.
        out_path.write_text(json.dumps(out, indent=2))

    if args.gate is not None and args.gate_config is not None:
        key = f"{args.gate_config[0]}_{args.gate_config[1]}"
        got = out["results"].get(key, {}).get("FID_2p5D_Avg")
        out["gate"] = {
            "expected": args.gate,
            "expected_source": args.gate_source,
            "got": got,
            "config": key,
        }
        if got is None:
            print(f"[E4] GATE ERROR: config {key} not in the sweep")
        else:
            delta = abs(got - args.gate)
            out["gate"]["passed"] = bool(delta < 0.005)
            print(
                f"[E4] GATE {'PASS' if delta < 0.005 else 'FAIL'}: "
                f"production {args.gate:.4f} vs sweep {got:.4f} (|Δ|={delta:.4f})"
            )

    out_path.write_text(json.dumps(out, indent=2))
    _plot()
    print(f"[E4] → {RESULTS / f'sweep_fid__{args.content}.json'}")


def _plot() -> None:
    """Overlay every content set's FID-vs-declared-spacing curve that has been computed."""
    # __PARTIAL files are plumbing checks scored against a different-sized real distribution;
    # overlaying them on a full-set curve would compare incomparable numbers.
    files = [
        p
        for p in sorted(RESULTS.glob("sweep_fid__*.json"))
        if "__PARTIAL" not in p.name
    ]
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for f in files:
        data = json.loads(f.read_text())
        for z in sorted({float(k.split("_")[1]) for k in data["results"]}):
            pts = sorted(
                (float(k.split("_")[0]), v["FID_2p5D_Avg"])
                for k, v in data["results"].items()
                if float(k.split("_")[1]) == z
            )
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", label=f"{data['content']}  z={z}")
    ax.axvline(0.7646, color="tab:blue", ls=":", label="train median 0.765")
    ax.set_xlabel("declared in-plane spacing (mm) — voxels unchanged")
    ax.set_ylabel("2.5D FID Avg (lower is better)")
    ax.set_title("E4: FID vs declared spacing, content held fixed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "F5_fid_vs_spacing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
