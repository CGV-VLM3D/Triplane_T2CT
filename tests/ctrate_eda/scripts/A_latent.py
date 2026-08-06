"""A_latent — MAISI latent-space EDA on ctrate_toy_v2 valid_v2 latents.

Self-contained, re-runnable. Read-only on datasets.

Outputs under /workspace/tests/ctrate_eda/:
  tables/latent_stats.csv, tables/latent_raw.json
  figures/latent_channels.png, figures/latent_umap_label.png,
  figures/latent_umap_vendor.png, figures/latent_decode_example.png

Analysis unit = VOLUME (one reconstruction / one latent .nii.gz).
Latent = MAISI VAE emb, on-disk NIfTI shape (H=120, W=120, D=64, C=4);
we transpose to (C=4, H=120, W=120, D=64). Canonical latent = *_emb (std~0.98);
mu.pt (std~0.67) is a different, mismatched source — noted if present.
"""

import sys

sys.path.insert(0, "/workspace")

import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/tests/ctrate_eda"
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
SMP = os.path.join(OUT, "samples")
for d in (TAB, FIG, SMP):
    os.makedirs(d, exist_ok=True)

LAT_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"
META_CSV = (
    "/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv"
)
N_STATS = 400  # >=200 latents for channel stats + UMAP
DECODE_N = 3  # latents to GPU-decode

TRAIN_C = "#1f77b4"  # convention: train=blue (unused here; valid set)
VALID_C = "#ff7f0e"  # valid=orange


def emb_files():
    """Sorted list of *_emb.nii.gz latent paths (excludes sidecar .json)."""
    fs = sorted(glob.glob(os.path.join(LAT_DIR, "*_emb.nii.gz")))
    return [f for f in fs if not f.endswith(".json")]


def vol_name(path):
    """emb path -> VolumeName (valid_1000_a_1_emb.nii.gz -> valid_1000_a_1.nii.gz)."""
    b = os.path.basename(path)
    return b.replace("_emb.nii.gz", ".nii.gz")


def load_latent(path):
    """Load one latent NIfTI -> (4,120,120,64) float32 (transpose channel-last)."""
    a = np.asanyarray(nib.load(path).dataobj).astype(np.float32)  # (120,120,64,4)
    return np.transpose(a, (3, 0, 1, 2))  # (4,120,120,64)


# ------------------------------------------------------------------ metadata
def load_meta():
    """VolumeName -> (vendor_family, kernel_str)."""
    df = pd.read_csv(META_CSV)

    def fam(m):
        m = str(m).upper()
        if "SIEMENS" in m:
            return "Siemens"
        if "PHILIPS" in m:
            return "Philips"
        if "PNMS" in m:
            return "PNMS"
        return "Other"

    df["vendor"] = df["Manufacturer"].map(fam)
    df["kernel"] = df["ConvolutionKernel"].astype(str)
    return df.set_index("VolumeName")[["vendor", "kernel"]].to_dict("index")


# ------------------------------------------------------------------ labels
def load_labels():
    """VolumeName -> dict(label->0/1) via repo datamodule (valid split)."""
    from src.data.ct_rate_datamodule import load_records

    recs = load_records("valid")
    return {r.volume_name: r.labels for r in recs}


def main():
    files = emb_files()
    print(f"[info] {len(files)} emb latents on disk")
    rng = np.random.default_rng(0)
    if len(files) > N_STATS:
        idx = rng.choice(len(files), N_STATS, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"[info] using {len(files)} latents for stats/UMAP")

    meta = load_meta()
    labels = load_labels()

    # ---- accumulate per-channel stats + pooled feature vectors -------------
    C = 4
    # running per-channel value collections via reservoir of flattened voxels
    # (exact percentiles need values; sample voxels to bound memory)
    per_ch_samples = [[] for _ in range(C)]  # subsampled voxel values / channel
    ch_min = np.full(C, np.inf, np.float32)
    ch_max = np.full(C, -np.inf, np.float32)
    ch_sum = np.zeros(C, np.float64)
    ch_sqsum = np.zeros(C, np.float64)
    ch_cnt = 0
    spatial_mean = np.zeros((C, 120, 120), np.float64)  # mean over D and samples

    feats = []  # (N, 1024) pooled features for UMAP
    vnames = []
    voxel_rng = np.random.default_rng(1)

    for k, f in enumerate(files):
        z = load_latent(f)  # (4,120,120,64)
        vnames.append(vol_name(f))
        ch_cnt += 1
        for c in range(C): # 채널
            zc = z[c]  # (120,120,64)
            ch_sum[c] += zc.sum()
            ch_sqsum[c] += (zc.astype(np.float64) ** 2).sum()
            ch_min[c] = min(ch_min[c], zc.min())
            ch_max[c] = max(ch_max[c], zc.max())
            spatial_mean[c] += zc.mean(axis=2)  # mean over D -> (120,120)
            # subsample ~2000 voxels/channel/volume for percentile estimate
            flat = zc.ravel()
            sel = voxel_rng.choice(flat.size, 2000, replace=False)
            per_ch_samples[c].append(flat[sel])
        # pooled feature: adaptive avg pool (4,120,120,64) -> (4,8,8,4) = 1024
        # simple block-mean pooling
        zp = z.reshape(4, 8, 15, 8, 15, 4, 16).mean(axis=(2, 4, 6))  # (4,8,8,4)
        feats.append(zp.ravel())
        if (k + 1) % 100 == 0:
            print(f"  loaded {k + 1}/{len(files)}")

    spatial_mean /= ch_cnt
    n_vox = ch_cnt * 120 * 120 * 64
    ch_mean = ch_sum / n_vox
    ch_std = np.sqrt(ch_sqsum / n_vox - ch_mean**2)

    # per-channel percentiles from subsampled voxels
    rows = []
    raw = {
        "source": "ctrate_toy_v2/valid_v2/latents/*_emb.nii.gz",
        "n_latents": int(ch_cnt),
        "unit": "volume",
        "channels": {},
    }
    for c in range(C):
        vals = np.concatenate(per_ch_samples[c])
        pct = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
        row = dict(
            channel=c,
            mean=float(ch_mean[c]),
            std=float(ch_std[c]),
            min=float(ch_min[c]),
            max=float(ch_max[c]),
            p01=float(pct[0]),
            p05=float(pct[1]),
            p25=float(pct[2]),
            p50=float(pct[3]),
            p75=float(pct[4]),
            p95=float(pct[5]),
            p99=float(pct[6]),
        )
        rows.append(row)
        raw["channels"][str(c)] = row
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(TAB, "latent_stats.csv"), index=False)
    print("[write] tables/latent_stats.csv")
    print(stats_df[["channel", "mean", "std", "min", "max"]].to_string(index=False))

    # overall std across all channels (scale confirmation)
    overall_std = float(
        np.sqrt((ch_sqsum.sum() / (n_vox * C)) - (ch_sum.sum() / (n_vox * C)) ** 2)
    )
    raw["overall_std_all_channels"] = overall_std
    raw["scale_note"] = (
        f"emb latent overall std={overall_std:.3f} (~0.98 expected); "
        "canonical *_emb source. mu.pt (std~0.67) NOT present in toy_v2 "
        "valid_v2 (checked); it is a different, mismatched VAE-mu source."
    )
    print(f"[info] overall std (all channels) = {overall_std:.4f}")

    # ---- figure 1: per-channel spatial mean maps ---------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for c in range(C):
        im = axes[c].imshow(spatial_mean[c], cmap="RdBu_r")
        axes[c].set_title(f"channel {c}\nmean={ch_mean[c]:.3f} std={ch_std[c]:.3f}")
        axes[c].set_xlabel("W")
        axes[c].set_ylabel("H")
        plt.colorbar(im, ax=axes[c], fraction=0.046)
    fig.suptitle(
        f"MAISI latent per-channel spatial mean (mean over D & {ch_cnt} valid_v2 volumes) "
        f"| unit=volume",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "latent_channels.png"), dpi=220)
    plt.close(fig)
    print("[write] figures/latent_channels.png")

    # ---- build color labels for UMAP ---------------------------------------
    LABELS = [
        "Medical material",
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]
    # global positive frequency among sampled volumes
    freq = {L: 0 for L in LABELS}
    for vn in vnames:
        lab = labels.get(vn, {})
        for L in LABELS:
            if lab.get(L, 0) == 1:
                freq[L] += 1
    top6 = [L for L, _ in sorted(freq.items(), key=lambda x: -x[1])[:6]]
    raw["label_freq_sampled"] = freq
    raw["top6_labels"] = top6

    def primary_label(vn):
        lab = labels.get(vn)
        if lab is None:
            return "Unknown"
        pos = [L for L in LABELS if lab.get(L, 0) == 1]
        if not pos:
            return "Normal"
        pos_top = [L for L in top6 if L in pos]
        if pos_top:
            return max(pos_top, key=lambda L: freq[L])
        return "Other"

    lab_cat = np.array([primary_label(vn) for vn in vnames])
    ven_cat = np.array([meta.get(vn, {}).get("vendor", "Unknown") for vn in vnames])

    # ---- dimensionality reduction: PCA -> UMAP -----------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = np.asarray(feats, np.float32)  # (N, 1024)
    Xs = StandardScaler().fit_transform(X)
    npca = min(50, Xs.shape[0] - 1, Xs.shape[1])
    Xp = PCA(n_components=npca, random_state=0).fit_transform(Xs)
    pca_full = PCA(n_components=min(10, npca), random_state=0).fit(Xs)
    raw["pca_explained_var_ratio_top10"] = [
        float(v) for v in pca_full.explained_variance_ratio_
    ]

    method = "UMAP"
    try:
        import umap

        emb2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xp)
    except Exception as e:
        from sklearn.manifold import TSNE

        method = f"TSNE(fallback:{type(e).__name__})"
        emb2 = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Xp)
    raw["reduction_method"] = method

    def scatter_by(cats, title, fname, palette):
        fig, ax = plt.subplots(figsize=(8, 7))
        for cat in palette:
            m = cats == cat
            if m.sum() == 0:
                continue
            ax.scatter(
                emb2[m, 0],
                emb2[m, 1],
                s=16,
                alpha=0.75,
                label=f"{cat} (n={int(m.sum())})",
                c=palette[cat],
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
        ax.legend(fontsize=8, markerscale=1.5, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, fname), dpi=220)
        plt.close(fig)
        print(f"[write] figures/{fname}")

    # label palette
    lab_order = ["Normal"] + top6 + ["Other", "Unknown"]
    lab_order = [c for c in lab_order if c in set(lab_cat)]
    cmap = plt.get_cmap("tab10")
    lab_pal = {c: cmap(i % 10) for i, c in enumerate(lab_order)}
    scatter_by(
        lab_cat,
        f"MAISI latent {method} (pooled 1024-d -> PCA{npca}) colored by primary abnormality | "
        f"unit=volume, n={ch_cnt} valid_v2",
        "latent_umap_label.png",
        lab_pal,
    )

    ven_order = [
        c
        for c in ["Philips", "Siemens", "PNMS", "Other", "Unknown"]
        if c in set(ven_cat)
    ]
    ven_pal = {c: cmap(i % 10) for i, c in enumerate(ven_order)}
    scatter_by(
        ven_cat,
        f"MAISI latent {method} (pooled 1024-d -> PCA{npca}) colored by manufacturer | "
        f"unit=volume, n={ch_cnt} valid_v2",
        "latent_umap_vendor.png",
        ven_pal,
    )

    # ---- confound quantification: silhouette by label vs vendor ------------
    from sklearn.metrics import silhouette_score

    def safe_sil(cats):
        mask = ~np.isin(cats, ["Unknown"])
        c = cats[mask]
        if len(set(c)) < 2:
            return None
        try:
            return float(silhouette_score(emb2[mask], c))
        except Exception:
            return None

    raw["silhouette_umap_by_label"] = safe_sil(lab_cat)
    raw["silhouette_umap_by_vendor"] = safe_sil(ven_cat)
    print(
        f"[info] silhouette by label={raw['silhouette_umap_by_label']} "
        f"vendor={raw['silhouette_umap_by_vendor']}"
    )

    # ================= GPU decode (device 1 only) ===========================
    decode_ok = False
    decode_err = None
    try:
        import torch
        from monai.inferers.inferer import SlidingWindowInferer
        from src.baselines.maisi import load_frozen

        # reuse the canonical memory-bounded tiled decode (roi=80^3, gaussian, overlap 0.4);
        # a direct full-volume decode OOMs on 480^3 activations
        from src.eval.samplers.report2ct import (
            _ReconModel,
            _dynamic_infer,
            _SW_ROI,
            _SW_OVERLAP,
        )

        dev = "cuda:0"  # physical GPU selected via CUDA_VISIBLE_DEVICES in the run cmd
        vae = load_frozen(device=dev)
        # round-trip decode: NO scale_factor (encode->decode identity) => sf=1.0
        recon = _ReconModel(vae, scale_factor=1.0).to(dev)
        dfiles = emb_files()[:DECODE_N]
        fig, axes = plt.subplots(1, DECODE_N, figsize=(5 * DECODE_N, 5))
        if DECODE_N == 1:
            axes = [axes]
        with torch.inference_mode():
            for j, f in enumerate(dfiles):
                z = load_latent(f)  # (4,120,120,64)
                zt = (
                    torch.from_numpy(z).unsqueeze(0).to(dev).float()
                )  # (1,4,120,120,64)
                inferer = SlidingWindowInferer(
                    roi_size=list(_SW_ROI),
                    sw_batch_size=1,
                    progress=False,
                    mode="gaussian",
                    overlap=_SW_OVERLAP,
                    sw_device=dev,
                    device=dev,
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    vol = _dynamic_infer(inferer, recon, zt)  # (1,1,480,480,256) ~[0,1]
                v = vol.float().squeeze().cpu().numpy()  # (480,480,256)
                hu = np.clip(v, 0, 1) * 2000 - 1000  # back to ~HU
                mid = hu[:, :, hu.shape[2] // 2]  # mid-axial
                axes[j].imshow(
                    mid.T, cmap="gray", vmin=-1000, vmax=1000, origin="lower"
                )
                axes[j].set_title(
                    f"{vol_name(f).replace('.nii.gz', '')}\nmid-axial (decoded)"
                )
                axes[j].axis("off")
        fig.suptitle(
            f"MAISI decode of valid_v2 latents (no scale_factor; round-trip) | HU window [-1000,1000]",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "latent_decode_example.png"), dpi=220)
        plt.close(fig)
        decode_ok = True
        raw["decoded_volumes"] = [vol_name(f) for f in dfiles]
        print("[write] figures/latent_decode_example.png")
    except Exception as e:
        import traceback

        decode_err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        # placeholder so the figure path exists
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            f"GPU decode skipped:\n{decode_err}",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.axis("off")
        fig.savefig(os.path.join(FIG, "latent_decode_example.png"), dpi=220)
        plt.close(fig)
        print(f"[warn] decode skipped: {decode_err}")

    raw["decode_ok"] = decode_ok
    raw["decode_err"] = decode_err

    with open(os.path.join(TAB, "latent_raw.json"), "w") as fh:
        json.dump(raw, fh, indent=2)
    print("[write] tables/latent_raw.json")
    print("[done]")


if __name__ == "__main__":
    main()
