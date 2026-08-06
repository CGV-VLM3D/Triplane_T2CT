"""A_latent_sep — Is abnormality separable in the CT latent space? (MAISI vs Wan)

Fixes the methodology of A_latent.py for the specific question "does abnormality
separate", then runs it for BOTH latent spaces over the SAME paired valid_v2 volumes.

Why this exists (what A_latent.py got wrong for this question):
  * A_latent computed silhouette on the UMAP-2D embedding. UMAP is nonlinear and does
    NOT preserve distances, so that silhouette measures UMAP distortion, not latent
    geometry. Here silhouette is computed in FAITHFUL metric spaces (standardized
    features + PCA-50); the UMAP-2D silhouette is reported only as a cautionary baseline.
  * Silhouette alone answers "does this labeling DOMINATE the geometry", not "is the
    signal PRESENT". We add a per-label linear probe (5-fold CV ROC-AUC): the decisive
    "is abnormality linearly decodable" test, with vendor as a positive control.
  * Mean pooling washes out focal findings -> we pool mean+std (focal contrast survives
    std). Abnormality is tested 3 ways: Normal-vs-Abnormal, primary-abnormality
    (multiclass, abnormal-only), and per-label probe.

Feature (per latent, block mean+std pooling -> ravel; equal 2048-d budget):
    MAISI (4,120,120,64) -> grid (8,8,4) -> 4*8*8*4*2  = 2048
    Wan   (16,64,64,64)  -> grid (4,4,4) -> 16*4*4*4*2 = 2048
  grids matched in RELATIVE coarseness (~1/15 of native in-plane).

Outputs under /workspace/tests/ctrate_eda/:
  tables/latent_sep.json           all silhouettes + per-label AUC + PCA EVR + Ns
  tables/latent_sep_auc.csv        label, prevalence, maisi_auc, wan_auc
  figures/latent_sep_umap_maisi.png / _wan.png   1x3 UMAP (vendor | norm/abn | primary)
  figures/latent_sep_auc.png       per-label probe AUC, MAISI vs Wan
  figures/latent_sep_silhouette.png  silhouette by grouping/space/latent
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/workspace")

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score

OUT = "/workspace/tests/ctrate_eda"
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
CACHE = os.path.join(TAB, "_cache")
for d in (TAB, FIG, CACHE):
    os.makedirs(d, exist_ok=True)

MAISI_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"
WAN_DIR = "/workspace/data/report2ct_wan/latents_512x512x253"
META_CSV = (
    "/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv"
)

GRID = {"maisi": (8, 8, 4), "wan": (4, 4, 4)}  # -> 2048-d each with mean+std

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


# --------------------------------------------------------------- feature build
def _emb_files(space):
    """Sorted *_emb.nii.gz latent paths for a space (excludes .json sidecars)."""
    if space == "maisi":
        fs = glob.glob(os.path.join(MAISI_DIR, "*_emb.nii.gz"))
    else:
        fs = glob.glob(os.path.join(WAN_DIR, "valid_*_emb.nii.gz"))
    return sorted(f for f in fs if f.endswith("_emb.nii.gz"))


def _vol_name(path):
    """emb path -> VolumeName (valid_1000_a_1_emb.nii.gz -> valid_1000_a_1.nii.gz)."""
    return os.path.basename(path).replace("_emb.nii.gz", ".nii.gz")


def _load_latent(path, space):
    """Load one latent NIfTI -> (C,H,W,D) float32 (transpose channel-last)."""
    a = np.asanyarray(nib.load(path).dataobj).astype(np.float32)  # (H,W,D,C)
    return np.transpose(a, (3, 0, 1, 2))  # (C,H,W,D)


def _pool_meanstd(z, grid):
    """(C,H,W,D) latent -> 1D feature via block mean+std on a (gh,gw,gd) grid.

    Returns 1D ``(C*gh*gw*gd*2,)`` = concat[mean-pooled ravel, std-pooled ravel].
    """
    C, H, W, D = z.shape
    gh, gw, gd = grid
    bh, bw, bd = H // gh, W // gw, D // gd
    zr = z.reshape(C, gh, bh, gw, bw, gd, bd)  # (C,gh,bh,gw,bw,gd,bd)
    m = zr.mean(axis=(2, 4, 6))  # (C,gh,gw,gd)
    s = zr.std(axis=(2, 4, 6))  # (C,gh,gw,gd)
    return np.concatenate([m.ravel(), s.ravel()]).astype(np.float32)


def build_features(space):
    """Pool all emb latents for a space -> (X (N,2048), vnames). Cached to disk."""
    cx = os.path.join(CACHE, f"feat_{space}.npy")
    cv = os.path.join(CACHE, f"feat_{space}_vnames.json")
    if os.path.exists(cx) and os.path.exists(cv):
        with open(cv) as fh:
            return np.load(cx), json.load(fh)
    files = _emb_files(space)
    print(f"[{space}] pooling {len(files)} latents ...", flush=True)
    X, vnames = [], []
    for k, f in enumerate(files):
        X.append(_pool_meanstd(_load_latent(f, space), GRID[space]))
        vnames.append(_vol_name(f))
        if (k + 1) % 200 == 0:
            print(f"  [{space}] {k + 1}/{len(files)}", flush=True)
    X = np.asarray(X, np.float32)
    np.save(cx, X)
    with open(cv, "w") as fh:
        json.dump(vnames, fh)
    print(f"[{space}] cached features {X.shape}", flush=True)
    return X, vnames


# ------------------------------------------------------------------ groupings
def load_vendor():
    """VolumeName -> vendor family {Siemens, Philips, PNMS, Other}."""
    df = pd.read_csv(META_CSV)

    def fam(m):
        m = str(m).upper()
        return (
            "Siemens"
            if "SIEMENS" in m
            else "Philips"
            if "PHILIPS" in m
            else "PNMS"
            if "PNMS" in m
            else "Other"
        )

    df["v"] = df["Manufacturer"].map(fam)
    return df.set_index("VolumeName")["v"].to_dict()


def load_labels():
    """VolumeName -> {label: 0/1} via the repo datamodule (valid split)."""
    from src.data.ct_rate_datamodule import load_records

    return {r.volume_name: r.labels for r in load_records("valid")}


def build_groupings(vnames, vendor_map, label_map):
    """Return dict of arrays aligned to vnames: vendor, norm_abn, primary, + label matrix."""
    vendor = np.array([vendor_map.get(v, "Unknown") for v in vnames])

    Y = np.zeros((len(vnames), len(LABELS)), np.int8)  # (N,18) binary label matrix
    known = np.ones(len(vnames), bool)
    for i, v in enumerate(vnames):
        lab = label_map.get(v)
        if lab is None:
            known[i] = False
            continue
        for j, L in enumerate(LABELS):
            Y[i, j] = 1 if lab.get(L, 0) == 1 else 0

    npos = Y.sum(axis=1)  # (N,) # positive labels per volume
    norm_abn = np.where(~known, "Unknown", np.where(npos == 0, "Normal", "Abnormal"))

    # primary abnormality = most population-frequent of a volume's positive top-6 labels
    freq = Y[known].sum(axis=0)  # (18,) prevalence among known
    top6 = list(np.argsort(-freq)[:6])
    primary = []
    for i in range(len(vnames)):
        if not known[i]:
            primary.append("Unknown")
        elif npos[i] == 0:
            primary.append("Normal")
        else:
            pos_top = [j for j in top6 if Y[i, j] == 1]
            if pos_top:
                primary.append(LABELS[max(pos_top, key=lambda j: freq[j])])
            else:
                primary.append("Other")
    return dict(
        vendor=vendor,
        norm_abn=norm_abn,
        primary=np.array(primary),
        Y=Y,
        known=known,
        top6=[LABELS[j] for j in top6],
    )


# --------------------------------------------------------------------- probes
def probe_auc(X, y):
    """5-fold stratified CV ROC-AUC of an L2 logistic probe (None if too rare)."""
    y = y.astype(int)
    if y.sum() < 20 or (len(y) - y.sum()) < 20:
        return None
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def sil(rep, cats, drop=("Unknown",)):
    """Silhouette of cats in representation `rep`, dropping `drop` categories."""
    mask = ~np.isin(cats, list(drop))
    c = cats[mask]
    if len(set(c)) < 2:
        return None
    return float(silhouette_score(rep[mask], c))


# ----------------------------------------------------------------------- viz
VENDOR_PAL = {
    "Philips": "#1f77b4",
    "Siemens": "#ff7f0e",
    "PNMS": "#2ca02c",
    "Other": "#7f7f7f",
    "Unknown": "#cccccc",
}
NA_PAL = {"Normal": "#4c72b0", "Abnormal": "#c44e52", "Unknown": "#cccccc"}


def umap_panels(space, emb2, g, sils, method):
    """1x3 UMAP scatter: vendor | Normal-vs-Abnormal | primary abnormality."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    cmap = plt.get_cmap("tab10")

    def scat(ax, cats, palette, title, order):
        for i, c in enumerate(order):
            m = cats == c
            if m.sum() == 0:
                continue
            col = palette[c] if isinstance(palette, dict) else cmap(i % 10)
            ax.scatter(
                emb2[m, 0],
                emb2[m, 1],
                s=14,
                alpha=0.7,
                label=f"{c} (n={int(m.sum())})",
                c=[col],
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
        ax.legend(fontsize=7, markerscale=1.4, loc="best")

    scat(
        axes[0],
        g["vendor"],
        VENDOR_PAL,
        f"by vendor  (sil PCA50={sils['vendor']['pca50']:+.3f})",
        ["Philips", "Siemens", "PNMS", "Other"],
    )
    scat(
        axes[1],
        g["norm_abn"],
        NA_PAL,
        f"Normal vs Abnormal  (sil PCA50={sils['norm_abn']['pca50']:+.3f})",
        ["Normal", "Abnormal"],
    )
    prim_order = ["Normal"] + g["top6"] + ["Other"]
    prim_order = [c for c in prim_order if c in set(g["primary"])]
    scat(
        axes[2],
        g["primary"],
        None,
        f"by primary abnormality  (sil PCA50 abn-only={sils['primary']['pca50']:+.3f})",
        prim_order,
    )

    fig.suptitle(
        f"{space.upper()} latent {method} (pooled 2048-d -> PCA50) | "
        f"n={len(emb2)} paired valid_v2 | UMAP=viz only, silhouettes computed in PCA-50",
        fontsize=12,
    )
    fig.tight_layout()
    p = os.path.join(FIG, f"latent_sep_umap_{space}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[write] {p}", flush=True)


# ---------------------------------------------------------------------- main
def analyze(space, X, g, seed, n):
    """Full separability analysis for one latent space; returns a results dict."""
    idx = np.arange(len(X))
    if 0 < n < len(X):
        idx = np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))
    Xn = X[idx]
    gv = {
        k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(X),) else v)
        for k, v in g.items()
    }

    Xs = StandardScaler().fit_transform(Xn)  # (n,2048)
    npca = min(50, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=npca, random_state=0)
    Xp = pca.fit_transform(Xs)  # (n,npca)
    cum_evr = float(np.cumsum(pca.explained_variance_ratio_)[-1])

    method = "UMAP"
    try:
        import umap

        emb2 = umap.UMAP(
            n_neighbors=15, min_dist=0.1, random_state=0, n_jobs=1
        ).fit_transform(Xp)
    except Exception as e:
        from sklearn.manifold import TSNE

        method = f"TSNE({type(e).__name__})"
        emb2 = TSNE(2, perplexity=30, random_state=0).fit_transform(Xp)

    # silhouettes in three representations; primary restricted to abnormal-only
    prim_drop = ("Unknown", "Normal")
    sils = {
        "vendor": {
            "full": sil(Xs, gv["vendor"]),
            "pca50": sil(Xp, gv["vendor"]),
            "umap2d": sil(emb2, gv["vendor"]),
        },
        "norm_abn": {
            "full": sil(Xs, gv["norm_abn"]),
            "pca50": sil(Xp, gv["norm_abn"]),
            "umap2d": sil(emb2, gv["norm_abn"]),
        },
        "primary": {
            "full": sil(Xs, gv["primary"], prim_drop),
            "pca50": sil(Xp, gv["primary"], prim_drop),
            "umap2d": sil(emb2, gv["primary"], prim_drop),
        },
    }

    # per-label linear probe (raw pooled features; pipeline standardizes per fold)
    Y = gv["Y"]
    aucs = {LABELS[j]: probe_auc(Xn, Y[:, j]) for j in range(len(LABELS))}
    prev = {LABELS[j]: float(Y[:, j].mean()) for j in range(len(LABELS))}
    # vendor positive-control probes (one-vs-rest for the two big vendors)
    vendor_auc = {
        v: probe_auc(Xn, (gv["vendor"] == v).astype(int))
        for v in ("Siemens", "Philips")
    }

    umap_panels(space, emb2, gv, sils, method)
    return dict(
        n=int(len(Xn)),
        pca_dim=int(npca),
        pca_cum_evr=cum_evr,
        reduction=method,
        silhouette=sils,
        label_auc=aucs,
        label_prev=prev,
        vendor_auc=vendor_auc,
        top6=gv["top6"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="subsample size (0=all paired)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xm, vm = build_features("maisi")
    Xw, vw = build_features("wan")

    # pair on common VolumeNames so MAISI/Wan see the exact same volumes
    common = sorted(set(vm) & set(vw))
    print(f"[pair] maisi={len(vm)} wan={len(vw)} common={len(common)}", flush=True)
    im = {v: i for i, v in enumerate(vm)}
    iw = {v: i for i, v in enumerate(vw)}
    Xm = Xm[[im[v] for v in common]]
    Xw = Xw[[iw[v] for v in common]]

    vendor_map, label_map = load_vendor(), load_labels()
    g = build_groupings(common, vendor_map, label_map)

    res = {
        "n_common": len(common),
        "seed": args.seed,
        "maisi": analyze("maisi", Xm, g, args.seed, args.n),
        "wan": analyze("wan", Xw, g, args.seed, args.n),
    }

    with open(os.path.join(TAB, "latent_sep.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("[write] tables/latent_sep.json", flush=True)

    # ---- AUC csv + comparison figure ----
    rows = []
    for L in LABELS:
        rows.append(
            dict(
                label=L,
                prevalence=res["maisi"]["label_prev"][L],
                maisi_auc=res["maisi"]["label_auc"][L],
                wan_auc=res["wan"]["label_auc"][L],
            )
        )
    auc_df = pd.DataFrame(rows).sort_values(
        "maisi_auc", ascending=False, na_position="last"
    )
    auc_df.to_csv(os.path.join(TAB, "latent_sep_auc.csv"), index=False)
    print("[write] tables/latent_sep_auc.csv", flush=True)

    d = auc_df.dropna(subset=["maisi_auc", "wan_auc"])
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y - 0.2, d["maisi_auc"], 0.4, label="MAISI", color="#4c72b0")
    ax.barh(y + 0.2, d["wan_auc"], 0.4, label="Wan", color="#dd8452")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance (0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{L}  ({p:.0%})" for L, p in zip(d["label"], d["prevalence"])], fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV ROC-AUC (linear probe on pooled latent)")
    ax.set_xlim(
        0.45, max(0.9, float(np.nanmax(d[["maisi_auc", "wan_auc"]].values)) + 0.03)
    )
    va = res["maisi"]["vendor_auc"]
    ax.set_title(
        "Is each abnormality LINEARLY decodable from the latent?\n"
        f"(label = prevalence; vendor positive-control AUC "
        f"Siemens={va['Siemens']:.2f} Philips={va['Philips']:.2f} [MAISI])"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "latent_sep_auc.png"), dpi=200)
    plt.close(fig)
    print("[write] figures/latent_sep_auc.png", flush=True)

    # ---- silhouette comparison figure (PCA-50 = the honest metric) ----
    groups = [
        ("vendor", "vendor"),
        ("norm_abn", "Normal/Abn"),
        ("primary", "primary(abn-only)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(groups))
    for off, sp, col in [(-0.2, "maisi", "#4c72b0"), (0.2, "wan", "#dd8452")]:
        vals = [res[sp]["silhouette"][k]["pca50"] or 0.0 for k, _ in groups]
        ax.bar(x + off, vals, 0.4, label=sp.upper(), color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in groups])
    ax.set_ylabel("silhouette in PCA-50 space")
    ax.set_title(
        "Latent silhouette by grouping (PCA-50; higher = tighter clusters)\n"
        "vendor is the confound baseline; abnormality groupings near 0 = no clustering"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "latent_sep_silhouette.png"), dpi=200)
    plt.close(fig)
    print("[write] figures/latent_sep_silhouette.png", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
