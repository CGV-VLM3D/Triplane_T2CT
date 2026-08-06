"""Aggregate report2ct_wan epoch-sweep metrics into a table + FID/CLIP-vs-epoch figure.

Scans outputs/report2ct_wan/ep_sweep/eval_ep*_n300_sp0.75_1.3_cfg1/ for one FID profile's
metrics.json (``--profile``, default research; see _find_metrics for the layouts covered), sorts
by epoch, and writes:
  - results/wan_epoch_sweep.csv
  - figs/wan_epoch_sweep/fid_clip_vs_epoch.png

An eval dir can hold several profiles at once (``fid_research/`` next to ``fid_docker/``), and
they are different metric families on ~35x-apart scales — hence one profile per invocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

SWEEP_DIR = Path("/workspace/outputs/report2ct_wan/ep_sweep")
DIR_RE = re.compile(r"\Aeval_ep(\d+)_n300_sp0\.75_1\.3_cfg1\Z")
OUT_CSV = Path("/workspace/results/wan_epoch_sweep.csv")
OUT_FIG = Path("/workspace/figs/wan_epoch_sweep/fid_clip_vs_epoch.png")
EXPECTED_N = 300

# ep299/cfg5/n300 anchor (established final operating point, outputs/report2ct_wan/cfg_sweep/
# eval_ep299_n300_sp0.75_1.3_cfg5/metrics.json) — different cfg than this sweep, reference only.
CFG5_ANCHOR = {"FID_2p5D_Avg": 1.61370, "CLIPScore": 65.16698}


def _find_metrics(eval_dir: Path, profile: str) -> Path | None:
    """Locate this run's ``metrics.json`` for ``profile``, across the layouts this repo produced.

    Since 2026-07-31 every scoring pass — ``run_eval.py`` and ``rescore_predictions.py`` alike —
    writes to ``<dir>/fid_<profile>/``, so a dir routinely holds ``fid_research/`` **and**
    ``fid_docker/`` side by side. Which one this table wants is not guessable from the tree; it
    is the caller's ``--profile``. Older layouts are then tried in recency order:

    1. ``<dir>/fid_<profile>/``         — current (both scorers)
    2. ``<dir>/metrics/fid_<profile>/`` — sweep_wan_epochs.sh before 2026-07-31
    3. ``<dir>/metrics/``               — rescore_predictions.py before 2026-07-29
    4. ``<dir>/metrics.json``           — run_eval.py before 2026-07-31

    (3) and (4) predate the ``fid_profile`` key, so they are only returned when ``profile`` is
    ``research`` — the setting every pre-2026-07-29 number was recorded under.
    """
    profiled = (
        eval_dir / f"fid_{profile}" / "metrics.json",
        eval_dir / "metrics" / f"fid_{profile}" / "metrics.json",
    )
    for candidate in profiled:
        if candidate.is_file():
            return candidate
    if profile != "research":
        return None
    for candidate in (eval_dir / "metrics" / "metrics.json", eval_dir / "metrics.json"):
        if candidate.is_file():
            return candidate
    return None


def collect(profile: str) -> pd.DataFrame:
    """Scan the sweep's eval dirs and build an epoch-sorted table of ``profile``'s metrics."""
    rows = []
    for d in sorted(SWEEP_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        metrics_path = _find_metrics(d, profile)
        if metrics_path is None:
            print(f"[skip] {d.name}: no {profile}-profile metrics.json yet")
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] {d.name}: metrics.json unreadable (mid-write?)")
            continue
        pred_dir = d / "predictions"
        n_preds = len(list(pred_dir.glob("*.mha"))) if pred_dir.is_dir() else None
        rows.append(
            {
                "epoch": int(m.group(1)),
                "FID_2p5D_Avg": metrics.get("FID_2p5D_Avg"),
                "FID_2p5D_XY": metrics.get("FID_2p5D_XY"),
                "FID_2p5D_YZ": metrics.get("FID_2p5D_YZ"),
                "FID_2p5D_XZ": metrics.get("FID_2p5D_XZ"),
                "CLIPScore": metrics.get("CLIPScore"),
                "CLIPScore_I2I": metrics.get("CLIPScore_I2I"),
                "n_samples": n_preds,
                # Absent in every metrics.json written before 2026-07-29 -> research.
                "fid_profile": metrics.get("fid_profile", "research"),
                "fid_num_images": metrics.get("fid_num_images"),
            }
        )
    if not rows:
        raise SystemExit(
            f"No completed {profile}-profile results under {SWEEP_DIR} "
            "(try --profile docker)"
        )
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def check_single_fid_profile(df: pd.DataFrame) -> None:
    """Refuse to tabulate/plot two FID profiles together — they are different metric families.

    ``docker`` (squeezenet1_1, 100 volumes) and ``research`` (radimagenet_resnet50, full set)
    write the SAME ``FID_2p5D_*`` keys on scales that differ by ~35x, so one mixed sweep would
    put both in one column and draw both against the 4.04 paper anchor.
    """
    profiles = sorted(df["fid_profile"].dropna().unique())
    if len(profiles) > 1:
        by_profile = {
            p: df.loc[df["fid_profile"] == p, "epoch"].tolist() for p in profiles
        }
        raise SystemExit(
            f"FID profiles are mixed across this sweep: {by_profile}. Re-score the odd ones out "
            "with the same task.fid_profile before aggregating (see docs/ctgen_local_eval.md)."
        )


def warn_bad_rows(df: pd.DataFrame) -> None:
    """Print loud warnings for rows that are NaN (a failed metric) or short of EXPECTED_N."""
    nan_rows = df[df[["FID_2p5D_Avg", "CLIPScore"]].isna().any(axis=1)]
    if not nan_rows.empty:
        print(
            f"[warn] {len(nan_rows)} epoch(s) have NaN metrics (failed run?): "
            f"{nan_rows['epoch'].tolist()}"
        )

    short_rows = df[df["n_samples"] != EXPECTED_N]
    if not short_rows.empty:
        print(
            f"[warn] {len(short_rows)} epoch(s) don't have {EXPECTED_N} predictions: "
            f"{list(zip(short_rows['epoch'], short_rows['n_samples']))}"
        )


def plot(df: pd.DataFrame, out_path: Path) -> None:
    """2-panel epoch vs FID (Avg + per-axis XY/YZ/XZ) / epoch vs CLIPScore (T2I + I2I),
    cfg=5 anchor shown as reference (Avg / T2I only, to keep the anchor legend uncluttered)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_fid, ax_clip) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_fid.plot(df["epoch"], df["FID_2p5D_Avg"], marker="o", color="black", label="Avg")
    ax_fid.plot(
        df["epoch"], df["FID_2p5D_XY"], marker="^", color="tab:blue", label="XY"
    )
    ax_fid.plot(
        df["epoch"], df["FID_2p5D_YZ"], marker="s", color="tab:green", label="YZ"
    )
    ax_fid.plot(df["epoch"], df["FID_2p5D_XZ"], marker="d", color="tab:red", label="XZ")
    ax_fid.axhline(
        CFG5_ANCHOR["FID_2p5D_Avg"],
        ls="--",
        color="gray",
        label="ep299/cfg5/n300 Avg (ref, different cfg)",
    )
    ax_fid.set_xlabel("epoch")
    ax_fid.set_ylabel("2.5D-FID")
    ax_fid.set_title("FID vs epoch (cfg=1)")
    ax_fid.legend(fontsize=8)

    ax_clip.plot(
        df["epoch"], df["CLIPScore"], marker="o", color="tab:orange", label="T2I"
    )
    ax_clip.plot(
        df["epoch"], df["CLIPScore_I2I"], marker="^", color="tab:purple", label="I2I"
    )
    ax_clip.axhline(
        CFG5_ANCHOR["CLIPScore"],
        ls="--",
        color="gray",
        label="ep299/cfg5/n300 T2I (ref, different cfg)",
    )
    ax_clip.set_xlabel("epoch")
    ax_clip.set_ylabel("CLIPScore")
    ax_clip.set_title("CLIPScore vs epoch (cfg=1)")
    ax_clip.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Not guessable from the tree: one eval dir now legitimately holds several profiles.
    ap.add_argument(
        "--profile", default="research", choices=("research", "docker", "docker_n300")
    )
    args = ap.parse_args()

    df = collect(args.profile)
    check_single_fid_profile(df)  # hard stop before anything is tabulated or plotted
    warn_bad_rows(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    plot(df, OUT_FIG)
    print(f"\nwrote {OUT_CSV} and {OUT_FIG}")


if __name__ == "__main__":
    main()
