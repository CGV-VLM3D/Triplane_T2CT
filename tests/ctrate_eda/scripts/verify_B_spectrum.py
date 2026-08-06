"""Independent re-computation of B_spectrum headline numbers (verifier).

Different code path from B_spectrum.py: own RAPSD via np.hypot radial binning,
own log-grid interp, own polyfit. Matches sampling params so alphas are comparable.
"""

import sys

sys.path.insert(0, "/workspace")
import glob
import numpy as np
import nibabel as nib

HU_LO, HU_HI = -1000.0, 1000.0
SLICES_PER_VOL = 20
N_LATENTS = 150
ZPLANES = 16
NBINS = 60


def rapsd(img):
    img = img - img.mean()
    wy = np.hanning(img.shape[0])
    wx = np.hanning(img.shape[1])
    img = img * np.outer(wy, wx)
    P = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - nx // 2, yy - ny // 2).astype(int)
    tbin = np.bincount(r.ravel(), P.ravel())
    nbin = np.bincount(r.ravel())
    radial = tbin / np.maximum(nbin, 1)
    n = min(ny, nx)
    kmax = n // 2
    freq = np.arange(kmax + 1) / n
    return freq[1:], radial[1 : kmax + 1]


def loginterp(fr, pw, grid):
    return np.interp(
        np.log10(grid),
        np.log10(fr),
        np.log10(np.maximum(pw, 1e-30)),
        left=np.nan,
        right=np.nan,
    )


def alpha_of(grid, logmean):
    m = np.isfinite(logmean)
    slope = np.polyfit(np.log10(grid)[m], logmean[m], 1)[0]
    return -slope


# ---- CT ----
ct_files = sorted(glob.glob("/workspace/tests/ctrate_eda_bundle/files/*/*.nii.gz"))
grid = np.logspace(np.log10(1.5e-3), np.log10(0.5), NBINS)
rows = []
nsl = 0
for f in ct_files:
    v = np.clip(nib.load(f).get_fdata(), HU_LO, HU_HI)
    v = (v - HU_LO) / (HU_HI - HU_LO)
    Z = v.shape[2]
    for z in np.unique(
        np.clip(np.linspace(Z * 0.1, Z * 0.9, SLICES_PER_VOL).astype(int), 0, Z - 1)
    ):
        sl = v[:, :, z]
        if sl.std() < 1e-4:
            continue
        fr, pw = rapsd(sl)
        rows.append(loginterp(fr, pw, grid))
        nsl += 1
ct_alpha = alpha_of(grid, np.nanmean(np.stack(rows), 0))
print(f"CT: n_vol={len(ct_files)} n_slices={nsl} alpha={ct_alpha:.4f}")

# ---- LATENT ----
lat_files = sorted(
    glob.glob("/workspace/data/ctrate_toy_v2/valid_v2/latents/*_emb.nii.gz")
)[:N_LATENTS]
lgrid = np.logspace(np.log10(1.0 / 120), np.log10(0.5), NBINS)
chrows = {c: [] for c in range(4)}
stds = []
for f in lat_files:
    a = nib.load(f).get_fdata()
    stds.append(float(a.std()))
    Zc = a.shape[2]
    zs = np.unique(np.linspace(0, Zc - 1, ZPLANES).astype(int))
    for c in range(4):
        for z in zs:
            pl = a[:, :, z, c]
            if pl.std() < 1e-8:
                continue
            fr, pw = rapsd(pl)
            chrows[c].append(loginterp(fr, pw, lgrid))
lat_alpha = {c: alpha_of(lgrid, np.nanmean(np.stack(chrows[c]), 0)) for c in range(4)}
print(f"LATENT: n={len(lat_files)} mean_std={np.mean(stds):.4f}")
for c in range(4):
    print(f"  ch{c}: alpha={lat_alpha[c]:.4f}  planes={len(chrows[c])}")
