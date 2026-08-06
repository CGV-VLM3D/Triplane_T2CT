"""B_spectrum — Power-spectrum RAPSD of CT pixels vs MAISI latents (single-pass EDA addition).

Ref: Dieleman, "Diffusion is spectral autoregression" (2024). Natural images have a
radially-averaged power spectral density (RAPSD) ~ 1/f^alpha with log-log slope alpha ~ 2
(measured ~2.45 on Imagenette). We ask: what is alpha for chest-CT axial slices, and what
is the per-channel alpha of the MAISI VAE latent space that report2ct/text2ct diffuse in?

Outputs:
  tables/rapsd.csv        freq + power_ct + power_latent_ch0..3 (common normalized-freq grid) + a header note w/ alphas
  tables/rapsd_raw.json   raw pre-rounding arrays + fitted alphas + metadata
  figures/rapsd.png       log-log CT RAPSD vs ideal 1/f^2, plus 4 latent-channel RAPSD curves
"""

import sys

sys.path.insert(0, "/workspace")

import glob
import json
import os

import numpy as np
import nibabel as nib

OUT = "/workspace/tests/ctrate_eda"
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

BUNDLE = "/workspace/tests/ctrate_eda_bundle/files"
LATENT_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"

HU_LO, HU_HI = -1000.0, 1000.0
SLICES_PER_VOL = 20  # evenly spaced axial slices per bundle volume
N_LATENTS = 150  # latent samples for the latent RAPSD
ZPLANES_PER_LATENT = 16  # evenly spaced z-planes per latent sample
NBINS = 60  # common normalized-frequency grid resolution


def hann2d(ny, nx):
    """Separable 2D Hann window ``(ny, nx)`` to suppress FFT edge leakage."""
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx)


def rapsd(img):
    """Radially-averaged power spectral density of a 2D array.

    Args:
        img: real 2D array ``(H, W)``.
    Returns:
        (freq, power): freq in cycles/pixel (0..0.5 Nyquist, DC excluded),
        power the radially-averaged |FFT|^2 at each frequency. 1D arrays.
    """
    img = img - img.mean()  # kill DC before windowing
    img = img * hann2d(*img.shape)  # (H, W)
    F = np.fft.fftshift(np.fft.fft2(img))  # (H, W) complex
    P = np.abs(F) ** 2  # (H, W) power
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)  # (H, W) radius in pixels
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), P.ravel())  # summed power per radial bin
    nbin = np.bincount(r_int.ravel())  # counts per radial bin
    radial = tbin / np.maximum(nbin, 1)  # mean power per radial bin
    n = min(ny, nx)
    kmax = n // 2  # Nyquist radial index
    radial = radial[: kmax + 1]
    freq = np.arange(kmax + 1) / n  # cycles/pixel; index k -> k/N
    return freq[1:], radial[1:]  # drop DC (freq=0)


def interp_log(freq, power, grid):
    """Interpolate log10(power) onto a common log-freq grid; returns log10 power on grid."""
    lf, lp = np.log10(freq), np.log10(np.maximum(power, 1e-30))
    return np.interp(np.log10(grid), lf, lp, left=np.nan, right=np.nan)


def fit_alpha(grid, logpow):
    """Fit log10(power) = b - alpha*log10(freq); return (alpha, intercept) over valid bins."""
    m = np.isfinite(logpow)
    lg = np.log10(grid)[m]
    lp = logpow[m]
    slope, intercept = np.polyfit(lg, lp, 1)
    return -slope, intercept  # RAPSD ~ 1/f^alpha => slope = -alpha


# ----------------------------------------------------------------------------- PIXEL / CT
ct_files = sorted(glob.glob(f"{BUNDLE}/*/*.nii.gz"))
print(f"[CT] {len(ct_files)} bundle volumes")

# common normalized-freq grid (cycles/pixel): from a few bins in to Nyquist, log-spaced
grid = np.logspace(np.log10(1.5e-3), np.log10(0.5), NBINS)

ct_logpows = []
n_ct_slices = 0
for f in ct_files:
    vol = nib.load(f).get_fdata()  # (X, Y, Z)
    vol = np.clip(vol, HU_LO, HU_HI)
    vol = (vol - HU_LO) / (HU_HI - HU_LO)  # -> [0, 1]
    Z = vol.shape[2]
    zs = np.linspace(Z * 0.1, Z * 0.9, SLICES_PER_VOL).astype(int)
    zs = np.unique(np.clip(zs, 0, Z - 1))
    for z in zs:
        sl = vol[:, :, z]  # (X, Y) axial slice
        if sl.std() < 1e-4:  # skip flat/all-air slices
            continue
        fr, pw = rapsd(sl)
        ct_logpows.append(interp_log(fr, pw, grid))
        n_ct_slices += 1

ct_logpow_mean = np.nanmean(np.stack(ct_logpows), axis=0)  # (NBINS,) mean in log domain
ct_power = 10**ct_logpow_mean
ct_alpha, ct_b = fit_alpha(grid, ct_logpow_mean)
print(f"[CT] {n_ct_slices} slices; alpha = {ct_alpha:.3f}")

# ----------------------------------------------------------------------------- LATENT
lat_files = sorted(glob.glob(f"{LATENT_DIR}/*_emb.nii.gz"))[:N_LATENTS]
print(f"[LATENT] {len(lat_files)} latent samples")

# latent is fixed 120x120 spatial; its own Nyquist grid (cycles/pixel)
lat_grid = np.logspace(np.log10(1.0 / 120), np.log10(0.5), NBINS)
lat_logpows = {c: [] for c in range(4)}
n_lat_planes = 0
for f in lat_files:
    arr = nib.load(f).get_fdata()  # (120, 120, 64, 4)
    Zc = arr.shape[2]
    zs = np.linspace(0, Zc - 1, ZPLANES_PER_LATENT).astype(int)
    zs = np.unique(zs)
    for c in range(4):
        for z in zs:
            plane = arr[:, :, z, c]  # (120, 120)
            if plane.std() < 1e-8:
                continue
            fr, pw = rapsd(plane)
            lat_logpows[c].append(interp_log(fr, pw, lat_grid))
    n_lat_planes += len(zs)

lat_power = {}
lat_alpha = {}
for c in range(4):
    lp = np.nanmean(np.stack(lat_logpows[c]), axis=0)  # (NBINS,)
    lat_power[c] = 10**lp
    a, _b = fit_alpha(lat_grid, lp)
    lat_alpha[c] = a
    print(f"[LATENT] ch{c}: alpha = {a:.3f}")

# ----------------------------------------------------------------------------- SAVE TABLES
import csv

with open(f"{OUT}/tables/rapsd.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
        [
            f"# CT alpha={ct_alpha:.4f} (n_slices={n_ct_slices}); "
            f"latent alpha ch0..3="
            f"{lat_alpha[0]:.4f},{lat_alpha[1]:.4f},{lat_alpha[2]:.4f},{lat_alpha[3]:.4f}; "
            f"CT freq grid != latent freq grid (both cycles/pixel, Nyquist=0.5)"
        ]
    )
    w.writerow(
        [
            "freq_ct",
            "power_ct",
            "freq_latent",
            "power_latent_ch0",
            "power_latent_ch1",
            "power_latent_ch2",
            "power_latent_ch3",
        ]
    )
    for i in range(NBINS):
        w.writerow(
            [
                f"{grid[i]:.6e}",
                f"{ct_power[i]:.6e}",
                f"{lat_grid[i]:.6e}",
                f"{lat_power[0][i]:.6e}",
                f"{lat_power[1][i]:.6e}",
                f"{lat_power[2][i]:.6e}",
                f"{lat_power[3][i]:.6e}",
            ]
        )

raw = {
    "meta": {
        "n_ct_volumes": len(ct_files),
        "n_ct_slices": int(n_ct_slices),
        "n_latent_samples": len(lat_files),
        "n_latent_planes_per_channel": int(
            sum(len(v) for v in lat_logpows.values()) // 4
        ),
        "hu_clip": [HU_LO, HU_HI],
        "freq_units": "cycles/pixel, Nyquist=0.5, DC excluded",
        "note_natural_image_alpha": "~2 (Dieleman 2024; ~2.45 Imagenette)",
    },
    "ct": {
        "freq": grid.tolist(),
        "power": ct_power.tolist(),
        "alpha": float(ct_alpha),
        "intercept": float(ct_b),
    },
    "latent": {
        "freq": lat_grid.tolist(),
        "power_ch0": lat_power[0].tolist(),
        "power_ch1": lat_power[1].tolist(),
        "power_ch2": lat_power[2].tolist(),
        "power_ch3": lat_power[3].tolist(),
        "alpha_ch0": float(lat_alpha[0]),
        "alpha_ch1": float(lat_alpha[1]),
        "alpha_ch2": float(lat_alpha[2]),
        "alpha_ch3": float(lat_alpha[3]),
    },
}
with open(f"{OUT}/tables/rapsd_raw.json", "w") as fh:
    json.dump(raw, fh, indent=2)

# ----------------------------------------------------------------------------- FIGURE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

# left: CT RAPSD vs ideal 1/f^2
axL.loglog(
    grid,
    ct_power,
    color="tab:blue",
    lw=2.2,
    label=f"CT axial slices (alpha={ct_alpha:.2f}, n={n_ct_slices})",
)
# ideal 1/f^2 reference anchored at CT curve's low-freq end
ref = ct_power[0] * (grid / grid[0]) ** (-2.0)
axL.loglog(grid, ref, "k--", lw=1.5, label="ideal 1/f^2 (alpha=2)")
axL.set_xlabel("spatial frequency (cycles/pixel)")
axL.set_ylabel("radially-averaged power")
axL.set_title(
    "CT-RATE axial-slice RAPSD vs natural-image 1/f^2\n"
    "(30 bundle volumes, HU clip [-1000,1000], per-slice Hann window)"
)
axL.legend(fontsize=9)
axL.grid(True, which="both", alpha=0.25)

# right: 4 latent channels
cols = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
for c in range(4):
    axR.loglog(
        lat_grid,
        lat_power[c],
        color=cols[c],
        lw=2.0,
        label=f"latent ch{c} (alpha={lat_alpha[c]:.2f})",
    )
axR.set_xlabel("spatial frequency (cycles/pixel)")
axR.set_ylabel("radially-averaged power")
axR.set_title(
    f"MAISI VAE latent RAPSD per channel\n"
    f"({len(lat_files)} valid_v2 samples, 120x120 planes, z-averaged)"
)
axR.legend(fontsize=9)
axR.grid(True, which="both", alpha=0.25)

fig.suptitle(
    "Power-spectrum (RAPSD) — analysis unit: axial slice (pixel) / z-plane (latent)",
    fontsize=13,
    y=1.02,
)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/rapsd.png", dpi=220, bbox_inches="tight")
print("saved figures/rapsd.png")
print("DONE")
