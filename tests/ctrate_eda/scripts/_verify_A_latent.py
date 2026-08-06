"""Independent re-computation of A_latent headline numbers.

Different code path: independent file selection (seed 7, 400 vols) + exact
full-voxel per-channel stats; replicate pooling only to check PCA determinism.
"""

import sys

sys.path.insert(0, "/workspace")
import os, glob, json
import numpy as np
import nibabel as nib

LAT_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"

files = sorted(
    f
    for f in glob.glob(os.path.join(LAT_DIR, "*_emb.nii.gz"))
    if not f.endswith(".json")
)
print(f"[disk] n emb latents = {len(files)}")

# ---- independent subset (different seed than script's seed=0) ----
rng = np.random.default_rng(7)
sub = [files[i] for i in sorted(rng.choice(len(files), 400, replace=False))]

# on-disk shape check
a0 = np.asanyarray(nib.load(sub[0]).dataobj)
print(f"[shape] on-disk latent0 = {a0.shape} dtype={a0.dtype}")

C = 4
csum = np.zeros(C)
csq = np.zeros(C)
cmin = np.full(C, np.inf)
cmax = np.full(C, -np.inf)
n = 0
for f in sub:
    a = np.asanyarray(nib.load(f).dataobj).astype(np.float32)  # (120,120,64,4)
    z = np.transpose(a, (3, 0, 1, 2))  # (4,120,120,64)
    for c in range(C):
        zc = z[c].astype(np.float64)
        csum[c] += zc.sum()
        csq[c] += (zc**2).sum()
        cmin[c] = min(cmin[c], zc.min())
        cmax[c] = max(cmax[c], zc.max())
    n += 1
nv = n * 120 * 120 * 64
cmean = csum / nv
cstd = np.sqrt(csq / nv - cmean**2)
overall = np.sqrt(csq.sum() / (nv * C) - (csum.sum() / (nv * C)) ** 2)
print(f"[indep 400@seed7] per-ch mean = {np.round(cmean, 4).tolist()}")
print(f"[indep 400@seed7] per-ch std  = {np.round(cstd, 4).tolist()}")
print(f"[indep 400@seed7] per-ch min  = {np.round(cmin, 3).tolist()}")
print(f"[indep 400@seed7] per-ch max  = {np.round(cmax, 3).tolist()}")
print(f"[indep 400@seed7] overall std = {overall:.4f}")

# ---- exact full over ALL 1304 for the truest reference (min/max especially) ----
csum2 = np.zeros(C)
csq2 = np.zeros(C)
n2 = 0
gmin = np.full(C, np.inf)
gmax = np.full(C, -np.inf)
for f in files:
    a = np.asanyarray(nib.load(f).dataobj).astype(np.float32)
    z = np.transpose(a, (3, 0, 1, 2))
    for c in range(C):
        zc = z[c].astype(np.float64)
        csum2[c] += zc.sum()
        csq2[c] += (zc**2).sum()
        gmin[c] = min(gmin[c], zc.min())
        gmax[c] = max(gmax[c], zc.max())
    n2 += 1
nv2 = n2 * 120 * 120 * 64
cmean2 = csum2 / nv2
cstd2 = np.sqrt(csq2 / nv2 - cmean2**2)
overall2 = np.sqrt(csq2.sum() / (nv2 * C) - (csum2.sum() / (nv2 * C)) ** 2)
print(f"[ALL {n2}] per-ch std = {np.round(cstd2, 4).tolist()} overall={overall2:.4f}")
print(f"[ALL {n2}] per-ch min = {np.round(gmin, 3).tolist()}")
print(f"[ALL {n2}] per-ch max = {np.round(gmax, 3).tolist()}")

# ---- PCA determinism check: replicate script pooling on the SAME seed-0 subset ----
rng0 = np.random.default_rng(0)
sub0 = [files[i] for i in sorted(rng0.choice(len(files), 400, replace=False))]
feats = []
for f in sub0:
    a = np.asanyarray(nib.load(f).dataobj).astype(np.float32)
    z = np.transpose(a, (3, 0, 1, 2))  # (4,120,120,64)
    zp = z.reshape(4, 8, 15, 8, 15, 4, 16).mean(axis=(2, 4, 6))  # (4,8,8,4)
    feats.append(zp.ravel())
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = np.asarray(feats, np.float32)
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=10, random_state=0).fit(Xs)
print(
    f"[PCA seed0 repl] top5 evr = {np.round(pca.explained_variance_ratio_[:5], 4).tolist()}"
)

# ---- valid_v2 is one-per-patient (scan/volume unit sanity) ----
import re

pats = set()
recs = set()
for f in files:
    b = os.path.basename(f).replace("_emb.nii.gz", "")
    m = re.match(r"(valid_\d+)_([a-z])_(\d+)", b)
    pats.add(m.group(1))
    recs.add(f"{m.group(1)}_{m.group(2)}")
print(f"[unit] {len(files)} volumes / {len(recs)} scans / {len(pats)} patients")

# ---- output/figure sanity ----
FIG = "/workspace/tests/ctrate_eda/figures"
for fn in [
    "latent_channels.png",
    "latent_umap_label.png",
    "latent_umap_vendor.png",
    "latent_decode_example.png",
]:
    p = os.path.join(FIG, fn)
    sz = os.path.getsize(p) if os.path.exists(p) else -1
    print(f"[fig] {fn}: {sz} bytes  ok={sz > 10240}")

out = dict(
    disk_n=len(files),
    shape=list(a0.shape),
    indep400_std=cstd.tolist(),
    indep400_overall=float(overall),
    all_std=cstd2.tolist(),
    all_overall=float(overall2),
    all_min=gmin.tolist(),
    all_max=gmax.tolist(),
    pca_top5=pca.explained_variance_ratio_[:5].tolist(),
    n_scans=len(recs),
    n_patients=len(pats),
)
with open("/workspace/tests/ctrate_eda/tables/_verify_A_latent_indep.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("[done]")
