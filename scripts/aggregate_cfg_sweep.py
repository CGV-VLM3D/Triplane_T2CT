"""Aggregate the report2ct / report2ct_wan cfg sweeps into a table + FID/CLIP/FVD-vs-cfg figure.

Reads the n=300 cfg sweep metrics.json produced this session (fixed epoch, cfg in {1,3,5,7,9})
for report2ct (toy_v2, sp0.757/0.757/1.5) and report2ct_wan (ep299, sp0.75/0.75/1.3), and writes:
  - results/cfg_sweep.csv
  - figs/cfg_sweep/fid_clip_fvd_vs_cfg.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_CSV = Path("/workspace/results/cfg_sweep.csv")
OUT_FIG = Path("/workspace/figs/cfg_sweep/fid_clip_fvd_vs_cfg.png")

CFGS = [1, 3, 5, 7, 9]
SWEEPS = {
    "report2ct (toy_v2, sp0.757)": {
        cfg: f"/workspace/outputs/report2ct/eval_toy_v2_n300_ep099_sp0.757_1.5_cfg{cfg}/metrics.json"
        for cfg in CFGS
    },
    "report2ct_wan (ep299)": {
        1: "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg1_priorsweep/metrics.json",
        3: "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg3/metrics.json",
        5: "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg5/metrics.json",
        7: "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg7/metrics.json",
        9: "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg9/metrics.json",
    },
}


def collect() -> pd.DataFrame:
    """Load each sweep's metrics.json into one cfg-sorted, model-labeled table."""
    rows = []
    for model, cfg_to_path in SWEEPS.items():
        for cfg, path in cfg_to_path.items():
            metrics = json.loads(Path(path).read_text())
            rows.append(
                {
                    "model": model,
                    "cfg": cfg,
                    "FID_2p5D_Avg": metrics["FID_2p5D_Avg"],
                    "CLIPScore": metrics["CLIPScore"],
                    "FVD_CTCLIP": metrics["FVD_CTCLIP"],
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "cfg"]).reset_index(drop=True)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    """3-panel cfg vs FID / CLIPScore-T2I / FVD_CTCLIP figure, one line per model."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_fid, ax_clip, ax_fvd) = plt.subplots(1, 3, figsize=(15, 4.5))

    for model, sub in df.groupby("model"):
        ax_fid.plot(sub["cfg"], sub["FID_2p5D_Avg"], marker="o", label=model)
        ax_clip.plot(sub["cfg"], sub["CLIPScore"], marker="o", label=model)
        ax_fvd.plot(sub["cfg"], sub["FVD_CTCLIP"], marker="o", label=model)

    ax_fid.set_xlabel("cfg_scale")
    ax_fid.set_ylabel("2.5D-FID (Avg)")
    ax_fid.set_title("FID vs cfg")
    ax_fid.legend(fontsize=8)

    ax_clip.set_xlabel("cfg_scale")
    ax_clip.set_ylabel("CLIPScore (T2I)")
    ax_clip.set_title("CLIPScore-T2I vs cfg")
    ax_clip.legend(fontsize=8)

    ax_fvd.set_xlabel("cfg_scale")
    ax_fvd.set_ylabel("FVD_CTCLIP (local proxy)")
    ax_fvd.set_title("FVD_CTCLIP vs cfg")
    ax_fvd.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = collect()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    plot(df, OUT_FIG)
    print(f"\nwrote {OUT_CSV} and {OUT_FIG}")


if __name__ == "__main__":
    main()
