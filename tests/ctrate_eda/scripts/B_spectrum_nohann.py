"""B_spectrum_nohann — how much does the Hann window change the RAPSD / alpha?

Dieleman's post computes RAPSD with a plain 2D FFT (no explicit window). Our
B_spectrum.py adds a per-slice Hann window to suppress edge leakage. This script
recomputes the same three sources BOTH ways (Hann vs no window) and overlays them,
so we can see whether our measured log-log slope alpha is robust to that choice.

Sources: CT axial slices (bundle), MAISI latent (4 ch), Wan latent (16 ch).
For the latents we plot the channel-mean RAPSD (one curve per window mode) and
also report the per-channel alpha range both ways.

Output: figures/rapsd_nohann.png  +  printed alpha table.
"""

import glob
import os

import numpy as np
import nibabel as nib

OUT = "/workspace/tests/ctrate_eda"
os.makedirs(f"{OUT}/figures", exist_ok=True)

BUNDLE = "/workspace/tests/ctrate_eda_bundle/files"
MAISI_DIR = "/workspace/data/ctrate_toy_v2/valid_v2/latents"
WAN_DIR = "/workspace/data/report2ct_wan/latents_512x512x253"

HU_LO, HU_HI = -1000.0, 1000.0
CT_SLICES = 20  # axial slices per bundle volume
N_LAT = 150  # latent samples per space
ZPLANES = 16  # z-planes per latent sample
NBINS = 60


def hann2d(ny, nx):
    return np.outer(np.hanning(ny), np.hanning(nx))


def rapsd(img, window):
    """RAPSD of a 2D array ``(H, W)``; `window`=apply Hann (else plain FFT)."""
    img = img - img.mean()  # kill DC
    if window:
        img = img * hann2d(*img.shape)  # (H, W)
    F = np.fft.fftshift(np.fft.fft2(img))  # (H, W) complex
    P = np.abs(F) ** 2  # (H, W) power
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)  # (H, W) radial bin
    radial = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = min(ny, nx)
    kmax = n // 2
    freq = np.arange(kmax + 1) / n  # cycles/pixel
    return freq[1:], radial[1 : kmax + 1]  # drop DC


def interp_log(freq, power, grid):
    lf, lp = np.log10(freq), np.log10(np.maximum(power, 1e-30))
    return np.interp(np.log10(grid), lf, lp, left=np.nan, right=np.nan)


def fit_alpha(grid, logpow):
    m = np.isfinite(logpow)
    slope, _ = np.polyfit(np.log10(grid)[m], logpow[m], 1)
    return -slope


# ------------------------------------------------------------------- CT
ct_files = sorted(glob.glob(f"{BUNDLE}/*/*.nii.gz"))
ct_grid = np.logspace(np.log10(1.5e-3), np.log10(0.5), NBINS)
ct_logpow = {True: [], False: []}
for f in ct_files:
    vol = np.clip(nib.load(f).get_fdata(), HU_LO, HU_HI)
    vol = (vol - HU_LO) / (HU_HI - HU_LO)  # -> [0,1]
    Z = vol.shape[2]
    zs = np.unique(
        np.clip(np.linspace(Z * 0.1, Z * 0.9, CT_SLICES).astype(int), 0, Z - 1)
    )
    for z in zs:
        sl = vol[:, :, z]  # (X, Y)
        if sl.std() < 1e-4:
            continue
        for win in (True, False):
            fr, pw = rapsd(sl, win)
            ct_logpow[win].append(interp_log(fr, pw, ct_grid))
ct_power = {w: 10 ** np.nanmean(np.stack(ct_logpow[w]), axis=0) for w in (True, False)}
ct_alpha = {
    w: fit_alpha(ct_grid, np.nanmean(np.stack(ct_logpow[w]), axis=0))
    for w in (True, False)
}


def latent_both(files, n_side, n_ch):
    """Return (grid, mean_power[win], mean_alpha[win], per_ch_alpha[win]) for win in {True,False}."""
    grid = np.logspace(np.log10(1.0 / n_side), np.log10(0.5), NBINS)
    # per (window, channel) accumulator of interpolated log-power rows
    acc = {w: {c: [] for c in range(n_ch)} for w in (True, False)}
    for f in files:
        arr = nib.load(f).get_fdata()  # (n_side, n_side, Z, n_ch)
        zs = np.unique(np.linspace(0, arr.shape[2] - 1, ZPLANES).astype(int))
        for c in range(n_ch):
            for z in zs:
                plane = arr[:, :, z, c]  # (n_side, n_side)
                if plane.std() < 1e-8:
                    continue
                for win in (True, False):
                    fr, pw = rapsd(plane, win)
                    acc[win][c].append(interp_log(fr, pw, grid))
    mean_power, mean_alpha, per_ch_alpha = {}, {}, {}
    for w in (True, False):
        ch_logmean = np.stack(
            [np.nanmean(np.stack(acc[w][c]), axis=0) for c in range(n_ch)]
        )  # (n_ch, NBINS)
        per_ch_alpha[w] = np.array(
            [fit_alpha(grid, ch_logmean[c]) for c in range(n_ch)]
        )
        allmean = ch_logmean.mean(axis=0)  # channel-mean in log-domain
        mean_power[w] = 10**allmean
        mean_alpha[w] = fit_alpha(grid, allmean)
    return grid, mean_power, mean_alpha, per_ch_alpha


maisi_files = sorted(glob.glob(f"{MAISI_DIR}/*_emb.nii.gz"))[:N_LAT]
wan_files = sorted(glob.glob(f"{WAN_DIR}/valid_*_emb.nii.gz"))[:N_LAT]
m_grid, m_power, m_alpha, m_chalpha = latent_both(maisi_files, 120, 4)
w_grid, w_power, w_alpha, w_chalpha = latent_both(wan_files, 64, 16)

# ------------------------------------------------------------------- PRINT TABLE
print("\n=== alpha: Hann vs no-window ===")
print(
    f"CT     : Hann={ct_alpha[True]:.3f}   no-win={ct_alpha[False]:.3f}   delta={ct_alpha[False] - ct_alpha[True]:+.3f}"
)
print(
    f"MAISI  : Hann={m_alpha[True]:.3f}   no-win={m_alpha[False]:.3f}   delta={m_alpha[False] - m_alpha[True]:+.3f}   (channel-mean)"
)
print(
    f"         per-ch Hann  range [{m_chalpha[True].min():.2f}..{m_chalpha[True].max():.2f}] mean {m_chalpha[True].mean():.2f}"
)
print(
    f"         per-ch no-win range [{m_chalpha[False].min():.2f}..{m_chalpha[False].max():.2f}] mean {m_chalpha[False].mean():.2f}"
)
print(
    f"WAN    : Hann={w_alpha[True]:.3f}   no-win={w_alpha[False]:.3f}   delta={w_alpha[False] - w_alpha[True]:+.3f}   (channel-mean)"
)
print(
    f"         per-ch Hann  range [{w_chalpha[True].min():.2f}..{w_chalpha[True].max():.2f}] mean {w_chalpha[True].mean():.2f}"
)
print(
    f"         per-ch no-win range [{w_chalpha[False].min():.2f}..{w_chalpha[False].max():.2f}] mean {w_chalpha[False].mean():.2f}"
)

# ------------------------------------------------------------------- FIGURE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))


def draw(ax, grid, power, alpha, title, ref=False):
    ax.loglog(
        grid,
        power[True],
        color="tab:blue",
        lw=2.2,
        label=f"Hann (alpha={alpha[True]:.2f})",
    )
    ax.loglog(
        grid,
        power[False],
        color="tab:red",
        lw=2.0,
        ls="--",
        label=f"no window (alpha={alpha[False]:.2f})",
    )
    if ref:
        r = power[True][0] * (grid / grid[0]) ** (-2.0)
        ax.loglog(grid, r, "k:", lw=1.3, label="ideal 1/f^2")
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("radially-averaged power")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)


draw(axes[0], ct_grid, ct_power, ct_alpha, "CT axial slices\n(30 vols)", ref=True)
draw(
    axes[1],
    m_grid,
    m_power,
    m_alpha,
    f"MAISI latent (channel-mean)\nper-ch alpha [{m_chalpha[True].min():.2f}..{m_chalpha[True].max():.2f}] Hann",
)
draw(
    axes[2],
    w_grid,
    w_power,
    w_alpha,
    f"Wan latent (channel-mean)\nper-ch alpha [{w_chalpha[True].min():.2f}..{w_chalpha[True].max():.2f}] Hann",
)

fig.suptitle(
    "RAPSD with Hann window vs no window — does the window change alpha?",
    fontsize=14,
    y=1.03,
)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/rapsd_nohann.png", dpi=200, bbox_inches="tight")
print("\nsaved figures/rapsd_nohann.png")
print("DONE")
