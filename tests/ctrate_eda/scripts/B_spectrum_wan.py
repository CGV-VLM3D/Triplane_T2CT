"""B_spectrum_wan — RAPSD of the Wan-VAE latent space, side-by-side with MAISI.

Companion to B_spectrum.py. That script measured the radially-averaged power
spectral density (RAPSD, ~1/f^alpha) of CT pixels and of the MAISI VAE latent
(4 ch, 120x120 planes). Here we add the WanVAE latent (16 ch, 64x64 planes) and
plot both latent spaces on their own Nyquist grids so their spectral slopes
(alpha) — i.e. how "spectrally white" each latent is — can be compared directly.

Latent shapes (channel-last, from the *_emb.nii.gz sidecars):
    MAISI : (120, 120, 64, 4)
    Wan   : ( 64,  64, 64, 16)

Outputs:
  figures/rapsd_wan.png    left = MAISI latent (4 ch), right = Wan latent (16 ch)
  tables/rapsd_wan.json    per-channel alphas + metadata for both latent spaces
"""

import glob
import json
import os

import numpy as np
import nibabel as nib

OUT = "/workspace/tests/ctrate_eda"
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

MAISI_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"
WAN_DIR = "/workspace/data/report2ct_wan/latents_512x512x253"

N_SAMPLES = 150  # latent samples per space (matches B_spectrum.py latent panel)
ZPLANES = 16  # evenly spaced z-planes per sample
NBINS = 60  # common log-freq grid resolution


# --- helpers copied verbatim from B_spectrum.py:39-86 (identical RAPSD math) ---
def hann2d(ny, nx):
    """Separable 2D Hann window ``(ny, nx)`` to suppress FFT edge leakage."""
    return np.outer(np.hanning(ny), np.hanning(nx))


def rapsd(img):
    """Radially-averaged power spectral density of a 2D array ``(H, W)``.

    Returns (freq, power): freq in cycles/pixel (0..0.5 Nyquist, DC dropped).
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
    radial = tbin / np.maximum(nbin, 1)
    n = min(ny, nx)
    kmax = n // 2
    radial = radial[: kmax + 1]
    freq = np.arange(kmax + 1) / n  # cycles/pixel; index k -> k/N
    return freq[1:], radial[1:]  # drop DC


def interp_log(freq, power, grid):
    """Interpolate log10(power) onto a common log-freq grid; returns log10 power."""
    lf, lp = np.log10(freq), np.log10(np.maximum(power, 1e-30))
    return np.interp(np.log10(grid), lf, lp, left=np.nan, right=np.nan)


def fit_alpha(grid, logpow):
    """Fit log10(power) = b - alpha*log10(freq); return (alpha, intercept)."""
    m = np.isfinite(logpow)
    slope, intercept = np.polyfit(np.log10(grid)[m], logpow[m], 1)
    return -slope, intercept  # RAPSD ~ 1/f^alpha => slope = -alpha


def latent_rapsd(files, n_side, n_ch):
    """Per-channel RAPSD over a latent set.

    Args:
        files: list of *_emb.nii.gz paths; each loads to ``(n_side, n_side, Z, n_ch)``.
        n_side: in-plane size (120 MAISI / 64 Wan) -> own Nyquist grid.
        n_ch: number of latent channels (4 MAISI / 16 Wan).
    Returns:
        (grid, power[n_ch][NBINS], alpha[n_ch]).
    """
    grid = np.logspace(np.log10(1.0 / n_side), np.log10(0.5), NBINS)
    logpows = {c: [] for c in range(n_ch)}
    for f in files:
        arr = nib.load(f).get_fdata()  # (n_side, n_side, Z, n_ch)
        Zc = arr.shape[2]
        zs = np.unique(np.linspace(0, Zc - 1, ZPLANES).astype(int))
        for c in range(n_ch):
            for z in zs:
                plane = arr[:, :, z, c]  # (n_side, n_side)
                if plane.std() < 1e-8:  # skip dead/constant plane
                    continue
                fr, pw = rapsd(plane)
                logpows[c].append(interp_log(fr, pw, grid))
    power, alpha = {}, {}
    for c in range(n_ch):
        lp = np.nanmean(np.stack(logpows[c]), axis=0)  # (NBINS,) log-domain mean
        power[c] = 10**lp
        alpha[c], _ = fit_alpha(grid, lp)
    return grid, power, alpha


# ------------------------------------------------------------------- COMPUTE
maisi_files = sorted(glob.glob(f"{MAISI_DIR}/*_emb.nii.gz"))[:N_SAMPLES]
wan_files = sorted(glob.glob(f"{WAN_DIR}/valid_*_emb.nii.gz"))[:N_SAMPLES]
print(f"[MAISI] {len(maisi_files)} samples  [WAN] {len(wan_files)} samples")

m_grid, m_power, m_alpha = latent_rapsd(maisi_files, n_side=120, n_ch=4)
w_grid, w_power, w_alpha = latent_rapsd(wan_files, n_side=64, n_ch=16)
print("[MAISI] alpha ch0..3 =", [f"{m_alpha[c]:.2f}" for c in range(4)])
print("[WAN]   alpha ch0..15 =", [f"{w_alpha[c]:.2f}" for c in range(16)])

# ------------------------------------------------------------------- SAVE JSON
raw = {
    "meta": {
        "n_samples": N_SAMPLES,
        "zplanes_per_sample": ZPLANES,
        "freq_units": "cycles/pixel, Nyquist=0.5, DC excluded",
        "note": "alpha = log-log RAPSD slope; ~0 = spectrally white, ~2 = natural-image 1/f^2",
    },
    "maisi": {
        "n_ch": 4,
        "in_plane": 120,
        "alpha": [float(m_alpha[c]) for c in range(4)],
        "freq": m_grid.tolist(),
        "power": {f"ch{c}": m_power[c].tolist() for c in range(4)},
    },
    "wan": {
        "n_ch": 16,
        "in_plane": 64,
        "alpha": [float(w_alpha[c]) for c in range(16)],
        "freq": w_grid.tolist(),
        "power": {f"ch{c}": w_power[c].tolist() for c in range(16)},
    },
}
with open(f"{OUT}/tables/rapsd_wan.json", "w") as fh:
    json.dump(raw, fh, indent=2)

# ------------------------------------------------------------------- FIGURE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

# left: MAISI latent (4 ch) — same as B_spectrum.py right panel
cols = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
for c in range(4):
    axL.loglog(
        m_grid,
        m_power[c],
        color=cols[c],
        lw=2.0,
        label=f"ch{c} (alpha={m_alpha[c]:.2f})",
    )
m_a = np.array([m_alpha[c] for c in range(4)])
axL.set_xlabel("spatial frequency (cycles/pixel)")
axL.set_ylabel("radially-averaged power")
axL.set_title(
    f"MAISI VAE latent RAPSD — 4 ch, 120x120 planes\n"
    f"({N_SAMPLES} valid_v2 samples; alpha mean={m_a.mean():.2f} "
    f"[{m_a.min():.2f}..{m_a.max():.2f}])"
)
axL.legend(fontsize=9)
axL.grid(True, which="both", alpha=0.25)

# right: Wan latent (16 ch) — colormap by channel + colorbar (16 legend rows is too many)
cmap = cm.get_cmap("viridis", 16)
for c in range(16):
    axR.loglog(w_grid, w_power[c], color=cmap(c), lw=1.4, alpha=0.9)
w_a = np.array([w_alpha[c] for c in range(16)])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=15))
cbar = fig.colorbar(sm, ax=axR, ticks=[0, 5, 10, 15], pad=0.01)
cbar.set_label("Wan latent channel")
axR.set_xlabel("spatial frequency (cycles/pixel)")
axR.set_ylabel("radially-averaged power")
axR.set_title(
    f"WanVAE latent RAPSD — 16 ch, 64x64 planes\n"
    f"({N_SAMPLES} valid samples; alpha mean={w_a.mean():.2f} "
    f"[{w_a.min():.2f}..{w_a.max():.2f}])"
)
axR.grid(True, which="both", alpha=0.25)

fig.suptitle(
    "Latent power-spectrum (RAPSD): MAISI (4 ch) vs Wan (16 ch) — "
    "flatter (alpha->0) = more spectrally white",
    fontsize=13,
    y=1.02,
)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/rapsd_wan.png", dpi=220, bbox_inches="tight")
print("saved figures/rapsd_wan.png")
print("DONE")
