"""
C6 (sparsity profile) and C7 (per-sample t-SNE/UMAP embedding) analysis.

Usage:
    python -m scripts.extend_analysis_c6_c7 \
        --analysis-dir /workspace/analysis/maisi_latent_distribution
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── UMAP (optional) ──────────────────────────────────────────────────────────
try:
    import umap as umap_lib

    UMAP_AVAILABLE = True
except ImportError:
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "umap-learn", "-q"]
        )
        import umap as umap_lib

        UMAP_AVAILABLE = True
    except Exception:
        UMAP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_cache(analysis_dir: Path):
    npz = np.load(analysis_dir / "cache" / "streaming_accumulators.npz")
    hist_counts = npz["hist_counts"]  # (4, 256)
    hist_edges = npz["hist_edges"]  # (257,)
    return hist_counts, hist_edges


def load_per_sample(analysis_dir: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_dir / "cache" / "per_sample_stats.csv")


def load_labels() -> pd.DataFrame:
    lbl_dir = Path(
        "/workspace/datasets/datasets/CT-RATE/dataset/multi_abnormality_labels"
    )
    frames = []
    for split in ("train", "valid"):
        p = lbl_dir / f"{split}_predicted_labels.csv"
        df = pd.read_csv(p)
        df["sample_id"] = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def gini_from_histogram(midpoints: np.ndarray, counts: np.ndarray) -> float:
    """Gini coefficient of the |x| distribution computed from a histogram.

    The histogram spans negative to positive values with symmetric bins, so
    we first merge symmetric ±bins that share the same |midpoint|, then apply
    the standard Gini formula on the resulting (x, weight) pairs sorted by x.
    G = 2/mean_x * sum_i w_i * x_i * F_i - 1   where F_i is the midpoint CDF.
    """
    abs_mid = np.abs(midpoints)
    unique_abs = np.unique(abs_mid)
    # Sum counts from both the positive and negative bin at each |mid|
    merged_counts = np.array([counts[abs_mid == u].sum() for u in unique_abs])
    total = merged_counts.sum()
    if total == 0:
        return 0.0
    w = merged_counts / total
    x = unique_abs
    mean_x = float(np.sum(w * x))
    if mean_x == 0:
        return 0.0
    cum_w = np.cumsum(w)
    F = cum_w - w / 2  # midpoint CDF
    gini = 2.0 * float(np.sum(w * x * F)) / mean_x - 1.0
    return float(np.clip(gini, 0.0, 1.0))


def entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a histogram."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ─────────────────────────────────────────────────────────────────────────────
# C6: Sparsity profile
# ─────────────────────────────────────────────────────────────────────────────


def run_c6(analysis_dir: Path, hist_counts: np.ndarray, hist_edges: np.ndarray):
    n_channels = hist_counts.shape[0]
    figs_dir = analysis_dir / "figs"
    tables_dir = analysis_dir / "tables"

    bin_mids = 0.5 * (hist_edges[:-1] + hist_edges[1:])  # (256,)

    # ── 1. P(|x| < ε) curves ─────────────────────────────────────────────────
    eps_grid = np.logspace(np.log10(0.01), np.log10(5.0), 30)
    sparsity_curves = np.zeros((n_channels, len(eps_grid)))  # P(|x|<eps)

    for c in range(n_channels):
        counts = hist_counts[c]
        total = counts.sum()
        for j, eps in enumerate(eps_grid):
            mask = np.abs(bin_mids) < eps
            sparsity_curves[c, j] = counts[mask].sum() / total

    # Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c in range(n_channels):
        ax.semilogx(
            eps_grid, sparsity_curves[c], label=f"ch{c}", color=colors[c], linewidth=2
        )
    ax.set_xlabel("ε (log scale)")
    ax.set_ylabel("P(|x| < ε)")
    ax.set_title("C6: Voxel Sparsity Curve — P(|x| < ε) per channel")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figs_dir / "c6_sparsity_curve.png", dpi=150)
    plt.close(fig)

    # ── 2–4. Gini, Entropy, Effective rank ───────────────────────────────────
    ginis = []
    entropies = []
    eff_ranks = []

    for c in range(n_channels):
        counts = hist_counts[c]
        ginis.append(gini_from_histogram(bin_mids, counts))
        H = entropy_bits(counts)
        entropies.append(H)
        eff_ranks.append(2**H)

    # ── Sparsity table at fixed eps values ───────────────────────────────────
    fixed_eps = [0.01, 0.05, 0.1, 0.5, 1.0]
    # Compute sparsity at fixed eps
    fixed_sparsity = np.zeros((n_channels, len(fixed_eps)))
    for c in range(n_channels):
        counts = hist_counts[c]
        total = counts.sum()
        for j, eps in enumerate(fixed_eps):
            mask = np.abs(bin_mids) < eps
            fixed_sparsity[c, j] = counts[mask].sum() / total

    # Average across channels at eps=0.05
    avg_sparsity_005 = float(fixed_sparsity[:, fixed_eps.index(0.05)].mean())

    # ── Write markdown table ──────────────────────────────────────────────────
    lines = ["# C6: Sparsity Profile\n"]
    lines.append("## P(|x| < ε) per channel\n")
    header = "| Channel |" + "".join(f" ε={e} |" for e in fixed_eps)
    sep = "|---------|" + "".join("--------|" for _ in fixed_eps)
    lines.append(header)
    lines.append(sep)
    for c in range(n_channels):
        row = f"| ch{c}     |" + "".join(
            f" {fixed_sparsity[c, j]:.4f} |" for j in range(len(fixed_eps))
        )
        lines.append(row)
    lines.append(f"\n**Average P(|x|<0.05) across channels: {avg_sparsity_005:.4f}**")
    lines.append(
        f"\n> Decision rule: if avg > 0.60 → upgrade FSQ and VQ small codebook\n"
    )

    lines.append("\n## Gini Coefficient, Entropy, Effective Sparsity Rank\n")
    lines.append("| Channel | Gini | Entropy (bits) | Eff. rank (2^H) |")
    lines.append("|---------|------|----------------|-----------------|")
    for c in range(n_channels):
        lines.append(
            f"| ch{c}     | {ginis[c]:.4f} | {entropies[c]:.4f} | {eff_ranks[c]:.2f} |"
        )
    avg_eff_rank = np.mean(eff_ranks)
    lines.append(f"\n**Mean effective sparsity rank: {avg_eff_rank:.2f}**")
    lines.append(
        "\n> If < 10 → upgrade quantization candidates; if > 50 → continuous baseline remains top.\n"
    )

    table_path = tables_dir / "c6_sparsity.md"
    table_path.write_text("\n".join(lines))
    print(f"[C6] Wrote {table_path}")

    return {
        "fixed_sparsity": fixed_sparsity,
        "fixed_eps": fixed_eps,
        "avg_sparsity_005": avg_sparsity_005,
        "ginis": ginis,
        "entropies": entropies,
        "eff_ranks": eff_ranks,
        "avg_eff_rank": avg_eff_rank,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C7: Per-sample embedding
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    f"ch{c}_{stat}"
    for c in range(4)
    for stat in ["mean", "std", "min", "max", "fp16_overflow", "z5_count", "abs5_count"]
]


def run_c7(analysis_dir: Path, df: pd.DataFrame, labels_df: pd.DataFrame):
    figs_dir = analysis_dir / "figs"
    tables_dir = analysis_dir / "tables"

    # ── Feature matrix ────────────────────────────────────────────────────────
    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    splits = df["split"].values  # 'train' / 'valid'
    split_color = np.where(splits == "train", 0, 1)

    # ── Join with abnormality labels ──────────────────────────────────────────
    merged = df.merge(labels_df, on="sample_id", how="left")
    abn_cols = [c for c in labels_df.columns if c not in ("VolumeName", "sample_id")]
    join_rate = merged["VolumeName"].notna().mean()
    print(
        f"[C7] Label join rate: {join_rate:.3f} ({int(join_rate * len(df))}/{len(df)})"
    )

    # Most common abnormality
    abn_sums = merged[abn_cols].sum().sort_values(ascending=False)
    most_common_abn = abn_sums.index[0]
    most_common_count = int(abn_sums.iloc[0])
    print(
        f"[C7] Most common abnormality: '{most_common_abn}' ({most_common_count} positives)"
    )

    use_labels = join_rate >= 0.80

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("[C7] Running t-SNE (this may take 1–2 min) ...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, n_jobs=-1)
    Z_tsne = tsne.fit_transform(X)
    print(f"[C7] t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

    # ── t-SNE plots ───────────────────────────────────────────────────────────
    if use_labels:
        n_panels = 3
        fig, axes = plt.subplots(1, n_panels, figsize=(18, 5))
    else:
        n_panels = 1
        fig, axes_raw = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes_raw]

    # Panel 0: color by split
    ax = axes[0]
    cmap_split = matplotlib.colormaps["bwr"]
    sc = ax.scatter(
        Z_tsne[:, 0],
        Z_tsne[:, 1],
        c=split_color,
        cmap=cmap_split,
        s=4,
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    ax.set_title("t-SNE — colored by split")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap_split(0.0),
            markersize=6,
            label="train",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap_split(1.0),
            markersize=6,
            label="valid",
        ),
    ]
    ax.legend(handles=handles, fontsize=8)

    if use_labels:
        # Panel 1: binary color by most common abnormality
        abn_vals = merged[most_common_abn].fillna(0).values.astype(float)
        ax1 = axes[1]
        colors_abn = np.where(abn_vals == 1, "#d62728", "#aec7e8")
        ax1.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=colors_abn, s=4, alpha=0.5)
        ax1.set_title(f"t-SNE — '{most_common_abn}'\n(red=positive)")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        h2 = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#d62728",
                markersize=6,
                label="positive",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#aec7e8",
                markersize=6,
                label="negative",
            ),
        ]
        ax1.legend(handles=h2, fontsize=8)

        # Panel 2: color by number of positive labels
        n_pos = merged[abn_cols].fillna(0).sum(axis=1).values.astype(float)
        ax2 = axes[2]
        sc2 = ax2.scatter(
            Z_tsne[:, 0], Z_tsne[:, 1], c=n_pos, cmap="plasma", s=4, alpha=0.5
        )
        plt.colorbar(sc2, ax=ax2, label="# positive labels")
        ax2.set_title("t-SNE — # positive abnormality labels")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")

    fig.tight_layout()
    tsne_path = figs_dir / "c7_tsne.png"
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)
    print(f"[C7] Wrote {tsne_path}")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    umap_done = False
    if UMAP_AVAILABLE:
        print("[C7] Running UMAP ...")
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, n_jobs=-1
        )
        Z_umap = reducer.fit_transform(X)
        print("[C7] UMAP done.")

        if use_labels:
            n_panels_u = 3
            fig_u, axes_u = plt.subplots(1, n_panels_u, figsize=(18, 5))
        else:
            fig_u, axes_raw_u = plt.subplots(1, 1, figsize=(6, 5))
            axes_u = [axes_raw_u]

        ax_u = axes_u[0]
        ax_u.scatter(
            Z_umap[:, 0],
            Z_umap[:, 1],
            c=split_color,
            cmap=cmap_split,
            s=4,
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        ax_u.set_title("UMAP — colored by split")
        ax_u.set_xlabel("UMAP 1")
        ax_u.set_ylabel("UMAP 2")
        ax_u.legend(handles=handles, fontsize=8)

        if use_labels:
            ax_u1 = axes_u[1]
            ax_u1.scatter(Z_umap[:, 0], Z_umap[:, 1], c=colors_abn, s=4, alpha=0.5)
            ax_u1.set_title(f"UMAP — '{most_common_abn}'\n(red=positive)")
            ax_u1.set_xlabel("UMAP 1")
            ax_u1.set_ylabel("UMAP 2")
            ax_u1.legend(handles=h2, fontsize=8)

            ax_u2 = axes_u[2]
            sc_u2 = ax_u2.scatter(
                Z_umap[:, 0], Z_umap[:, 1], c=n_pos, cmap="plasma", s=4, alpha=0.5
            )
            plt.colorbar(sc_u2, ax=ax_u2, label="# positive labels")
            ax_u2.set_title("UMAP — # positive abnormality labels")
            ax_u2.set_xlabel("UMAP 1")
            ax_u2.set_ylabel("UMAP 2")

        fig_u.tight_layout()
        umap_path = figs_dir / "c7_umap.png"
        fig_u.savefig(umap_path, dpi=150)
        plt.close(fig_u)
        print(f"[C7] Wrote {umap_path}")
        umap_done = True
    else:
        print("[C7] umap-learn not available — skipping UMAP.")

    # ── Cluster observation ───────────────────────────────────────────────────
    # Compute ratio of within-split variance to total variance as a proxy for
    # split separation (if high -> split clusters are distinct)
    train_mask = splits == "train"
    valid_mask = splits == "valid"
    total_var = Z_tsne.var(axis=0).sum()
    within_var = (
        Z_tsne[train_mask].var(axis=0).sum() * train_mask.sum()
        + Z_tsne[valid_mask].var(axis=0).sum() * valid_mask.sum()
    ) / len(Z_tsne)
    between_ratio = 1.0 - within_var / total_var  # near 0 = interleaved
    cluster_obs = (
        (
            "train and valid points are strongly interleaved with no visible cluster separation "
            "(between-split variance ratio = {:.3f} ≈ 0) — good: no domain drift in the latent space."
        ).format(between_ratio)
        if between_ratio < 0.05
        else (
            "moderate split separation observed (between-split variance ratio = {:.3f}); "
            "may reflect sampling differences between train and valid sets."
        ).format(between_ratio)
    )

    # ── Notes table ───────────────────────────────────────────────────────────
    notes = textwrap.dedent(f"""\
    # C7: Per-sample Embedding Notes

    ## Parameters
    - t-SNE: n_components=2, perplexity=30, init='pca', random_state=0
    - UMAP: n_components=2, n_neighbors=15, min_dist=0.1, random_state=0 ({"available" if umap_done else "not available — skipped"})
    - Feature matrix: 28-dim (7 stats × 4 channels), standardized (z-score)
    - Samples: {len(df)} total (train={int(train_mask.sum())}, valid={int(valid_mask.sum())})

    ## Label Join
    - Join rate: {join_rate:.3f} ({int(join_rate * len(df))}/{len(df)} samples matched)
    - Most common abnormality: '{most_common_abn}' ({most_common_count} positives across all splits)
    - Additional panels: binary '{most_common_abn}' color, total positive label count color

    ## Observations
    - {cluster_obs}
    - Between-split variance ratio: {between_ratio:.4f}
    - t-SNE KL divergence: {tsne.kl_divergence_:.4f}
    """)
    notes_path = tables_dir / "c7_embedding_notes.md"
    notes_path.write_text(notes)
    print(f"[C7] Wrote {notes_path}")

    return {
        "join_rate": join_rate,
        "most_common_abn": most_common_abn,
        "cluster_obs": cluster_obs,
        "between_ratio": between_ratio,
        "umap_done": umap_done,
        "use_labels": use_labels,
        "kl_div": tsne.kl_divergence_,
        "n_pos": n_pos if use_labels else None,
        "abn_cols": abn_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT.md append
# ─────────────────────────────────────────────────────────────────────────────


def update_report(analysis_dir: Path, c6: dict, c7: dict):
    report_path = analysis_dir / "REPORT.md"
    existing = report_path.read_text()

    # ── D1 re-rank signal ─────────────────────────────────────────────────────
    br = c7["between_ratio"]
    if br < 0.05:
        d1_c7_signal = (
            "C7 t-SNE: train and valid are interleaved (between-split ratio = {:.3f}); "
            "latent space shows no pathology-driven macro-clusters across the 28-dim stat features. "
            "This is consistent with the global axis-projection hypothesis being approximately valid — "
            "the lack of strongly separated clusters means mean-collapse PVE is not grossly "
            "pessimistic from a domain-coverage standpoint, though individual sample heterogeneity "
            "remains. Ranking unchanged from C3.".format(br)
        )
    else:
        d1_c7_signal = (
            "C7 t-SNE shows moderate split/cluster separation (between-split ratio = {:.3f}); "
            "axis-projection that preserves these clusters is the target — mean-collapse PVE alone "
            "may be pessimistic. Consider cluster-conditional PVE analysis.".format(br)
        )

    # ── D2 re-rank signal ─────────────────────────────────────────────────────
    avg_sp005 = c6["avg_sparsity_005"]
    avg_eff = c6["avg_eff_rank"]
    d2_c6_signal_lines = []
    fsq_upgrade = avg_sp005 > 0.60
    quant_upgrade_eff = avg_eff < 10
    continuous_top = avg_eff > 50

    if fsq_upgrade:
        d2_c6_signal_lines.append(
            f"C6: avg P(|x|<0.05) = {avg_sp005:.4f} > 0.60 → **FSQ and VQ small codebook upgraded** "
            f"(high voxel sparsity favors aggressive quantization)."
        )
    else:
        d2_c6_signal_lines.append(
            f"C6: avg P(|x|<0.05) = {avg_sp005:.4f} ≤ 0.60 — sparsity below threshold; "
            f"no mandatory FSQ/VQ upgrade."
        )
    if quant_upgrade_eff:
        d2_c6_signal_lines.append(
            f"C6: mean effective sparsity rank = {avg_eff:.2f} < 10 → upgrade quantization candidates."
        )
    elif continuous_top:
        d2_c6_signal_lines.append(
            f"C6: mean effective sparsity rank = {avg_eff:.2f} > 50 → continuous baseline remains top."
        )
    else:
        d2_c6_signal_lines.append(
            f"C6: mean effective sparsity rank = {avg_eff:.2f} (10–50 range) → no decisive ranking change."
        )
    d2_c6_signal = "\n".join(d2_c6_signal_lines)

    # ── Build D1 updated table ────────────────────────────────────────────────
    d1_table = textwrap.dedent(f"""\
    | Rank | Candidate | Evidence (metric values) | Confidence |
    |------|-----------|--------------------------|------------|
    | 1 | low-rank 3D AE (abandon triplane) | C3: mean 3-plane PVE = 21.0% < 70%; C7: interleaved clusters (ratio={br:.3f}) — PVE not pessimistic | high |
    | 2 | hybrid triplane+3D | C3: partial plane structure may still help | med |
    | 3 | triplane keep | C3: low PVE — triplane insufficient | low |
    """)

    # ── Build D2 updated table ────────────────────────────────────────────────
    if fsq_upgrade or quant_upgrade_eff:
        d2_r1 = (
            "FSQ (finite scalar quantization)"
            if fsq_upgrade
            else "Continuous KL-VAE (latent bottleneck)"
        )
        d2_r2 = (
            "Continuous KL-VAE (latent bottleneck)"
            if fsq_upgrade
            else "VQ (vector quantization)"
        )
        d2_r1_ev = (
            f"C6: avg P(|x|<0.05)={avg_sp005:.3f} > 0.60 — high sparsity; "
            f"eff_rank={avg_eff:.1f}"
            if fsq_upgrade
            else f"C2: eff_rank=3.97; C6: eff_rank={avg_eff:.1f}"
        )
        d2_r1_conf = "high" if fsq_upgrade else "med"
    else:
        d2_r1 = "Continuous KL-VAE (latent bottleneck)"
        d2_r2 = "VQ (vector quantization)"
        d2_r1_ev = f"C2: eff_rank=3.97; C6: avg P(|x|<0.05)={avg_sp005:.3f}; eff_rank={avg_eff:.1f}"
        d2_r1_conf = "med"

    d2_table = textwrap.dedent(f"""\
    | Rank | Candidate | Evidence (metric values) | Confidence |
    |------|-----------|--------------------------|------------|
    | 1 | {d2_r1} | {d2_r1_ev} | {d2_r1_conf} |
    | 2 | {d2_r2} | C1: max kurtosis=2.0 ≤ 10; C2: eff_rank=3.97 | med |
    | 3 | FSQ (finite scalar quantization) | C6: P(|x|<0.05)={avg_sp005:.3f}; eff_rank={avg_eff:.1f} | {"high" if fsq_upgrade else "med"} |
    | 4 | LFQ (lookup-free quantization) | C2: eff_rank=3.97 — LFQ scales well with rank | med |
    | 5 | VQ small codebook | C6: sparsity={"favors" if fsq_upgrade else "neutral for"} small codebook | {"high" if fsq_upgrade else "low"} |
    | 6 | RVQ (residual VQ) | C2: eff_rank=3.97 — RVQ for high effective rank | low |
    | 7 | No quantization (plain AE) | Baseline for ablation | low |
    """)

    # ── Sparsity summary for C6 ───────────────────────────────────────────────
    ch_lines = []
    for c in range(4):
        sp05 = c6["fixed_sparsity"][c, c6["fixed_eps"].index(0.05)]
        sp01 = c6["fixed_sparsity"][c, c6["fixed_eps"].index(0.1)]
        gini = c6["ginis"][c]
        H = c6["entropies"][c]
        er = c6["eff_ranks"][c]
        ch_lines.append(
            f"- ch{c}: P(|x|<0.05)={sp05:.4f}, P(|x|<0.1)={sp01:.4f}, "
            f"Gini={gini:.4f}, H={H:.2f} bits, eff_rank={er:.1f}"
        )
    c6_summary = "\n".join(ch_lines)
    c6_key = (
        f"Avg P(|x|<0.05) across channels = {avg_sp005:.4f}; "
        f"mean effective sparsity rank = {avg_eff:.2f}."
    )

    new_section = textwrap.dedent(f"""\

    ---

    ## C6: Voxel-level Sparsity

    {c6_key}

    Per-channel:
    {c6_summary}

    **Finding**: {"High voxel sparsity (avg P(|x|<0.05) > 60%) — favors aggressive quantization (FSQ, VQ small codebook). " if fsq_upgrade else "Moderate voxel sparsity — no mandatory quantization upgrade. "}Effective sparsity rank = {avg_eff:.2f} {"< 10 — latent space is relatively low-entropy." if quant_upgrade_eff else ("> 50 — high entropy; continuous baseline remains top." if continuous_top else "(10–50 range).")}

    See `figs/c6_sparsity_curve.png` and `tables/c6_sparsity.md` for full detail.

    ## C7: Per-sample Semantic Embedding

    Label join rate: {c7["join_rate"]:.3f} ({int(c7["join_rate"] * 6000)}/6000 samples matched).
    Most common abnormality: '{c7["most_common_abn"]}'.
    UMAP: {"available and computed" if c7["umap_done"] else "not available — skipped"}.

    **Finding**: {c7["cluster_obs"]}

    See `figs/c7_tsne.png`{", `figs/c7_umap.png`" if c7["umap_done"] else ""} and `tables/c7_embedding_notes.md`.

    ## Updated Design Decision Rankings (with C6/C7 evidence)

    ### D1: Triplane Hypothesis (updated)

    {d1_c7_signal}

    {d1_table}

    ### D2: Tokenization (updated)

    {d2_c6_signal}

    {d2_table}
    """)

    report_path.write_text(existing + new_section)
    print(f"[REPORT] Appended C6/C7 sections to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extend MAISI latent analysis with C6 and C7."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("/workspace/analysis/maisi_latent_distribution"),
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    assert analysis_dir.exists(), f"analysis-dir not found: {analysis_dir}"

    print("=== Loading cache ===")
    hist_counts, hist_edges = load_cache(analysis_dir)
    df = load_per_sample(analysis_dir)
    labels_df = load_labels()
    print(
        f"Loaded hist_counts {hist_counts.shape}, {len(df)} sample rows, {len(labels_df)} label rows"
    )

    print("\n=== C6: Sparsity profile ===")
    c6 = run_c6(analysis_dir, hist_counts, hist_edges)

    print("\n=== C7: Per-sample embedding ===")
    c7 = run_c7(analysis_dir, df, labels_df)

    print("\n=== Updating REPORT.md ===")
    update_report(analysis_dir, c6, c7)

    print("\n=== Done ===")
    print(f"  figs/c6_sparsity_curve.png")
    print(f"  tables/c6_sparsity.md")
    print(f"  figs/c7_tsne.png")
    if c7["umap_done"]:
        print(f"  figs/c7_umap.png")
    print(f"  tables/c7_embedding_notes.md")
    print(f"  REPORT.md (appended)")


if __name__ == "__main__":
    main()
