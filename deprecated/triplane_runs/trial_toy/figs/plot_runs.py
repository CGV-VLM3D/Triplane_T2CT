"""Produce training-curve figures for the Tier-1 3-way comparison."""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── config ──────────────────────────────────────────────────────────────────
OUT_DIR = "/workspace/runs/trial_toy/figs"
DATA_FILE = os.path.join(OUT_DIR, "run_data.json")

MAISI_PSNR = 26.63  # 2mm image-domain ceiling
MAISI_SSIM = 0.686  # not in latent data but kept for reference

BATCH_SIZES = {"trial_toy": 8, "trial_toy_baseline": 4, "trial_toy_d3t": 4}
TRAIN_SIZE = 5000

LABELS = {
    "trial_toy": "TriplaneAE std v1.0 w½ (ours)",
    "trial_toy_baseline": "TriVQAEConv (baseline)",
    "trial_toy_d3t": "TriVAE_D3T Swin (d3t)",
}
COLORS = {
    "trial_toy": "#1f77b4",  # blue
    "trial_toy_baseline": "#d62728",  # red
    "trial_toy_d3t": "#2ca02c",  # green
}
LINESTYLES = {
    "trial_toy": "-",
    "trial_toy_baseline": "--",
    "trial_toy_d3t": "-.",
}

# ── load data ────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    data = json.load(f)


def steps_to_epoch(steps, name):
    spe = math.ceil(TRAIN_SIZE / BATCH_SIZES[name])
    return steps / spe


# ── helper: val epochs and metric arrays ────────────────────────────────────
def get_val_series(name, metric):
    rows = data[name]["val_rows"]
    xs, ys = [], []
    for r in rows:
        if r.get(metric) is not None:
            xs.append(steps_to_epoch(r["step"], name))
            ys.append(r[metric])
    return xs, ys


def get_train_series(name):
    rows = data[name]["train_rows"]
    # downsample to at most 2000 points for readability
    step = max(1, len(rows) // 2000)
    sampled = rows[::step]
    xs = [steps_to_epoch(r["step"], name) for r in sampled]
    ys = [r["train_loss"] for r in sampled]
    return xs, ys


# ── figure 1: val latent_psnr ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in ["trial_toy", "trial_toy_baseline", "trial_toy_d3t"]:
    xs, ys = get_val_series(name, "val_psnr")
    ax.plot(
        xs,
        ys,
        label=LABELS[name],
        color=COLORS[name],
        linestyle=LINESTYLES[name],
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

ax.axhline(
    MAISI_PSNR,
    color="black",
    linestyle=":",
    linewidth=1.4,
    label=f"2mm image ceiling (informational) {MAISI_PSNR:.2f} dB",
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val Latent PSNR (dB)", fontsize=12)
ax.set_title("Tier-1 3-Way: Val Latent PSNR vs Epoch", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
# ensure ceiling line is visible
ymin = min(min(get_val_series(n, "val_psnr")[1], default=20) for n in LABELS)
ax.set_ylim(bottom=min(ymin - 0.5, MAISI_PSNR - 1.5), top=MAISI_PSNR + 1.5)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_psnr.png"), dpi=150)
plt.close(fig)
print("Saved val_psnr.png")

# ── figure 2: val latent_l1 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in ["trial_toy", "trial_toy_baseline", "trial_toy_d3t"]:
    xs, ys = get_val_series(name, "val_l1")
    ax.plot(
        xs,
        ys,
        label=LABELS[name],
        color=COLORS[name],
        linestyle=LINESTYLES[name],
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val Latent L1", fontsize=12)
ax.set_title("Tier-1 3-Way: Val Latent L1 vs Epoch", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_l1.png"), dpi=150)
plt.close(fig)
print("Saved val_l1.png")

# ── figure 3: train loss ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in ["trial_toy", "trial_toy_baseline", "trial_toy_d3t"]:
    xs, ys = get_train_series(name)
    ax.plot(
        xs,
        ys,
        label=LABELS[name],
        color=COLORS[name],
        linestyle=LINESTYLES[name],
        linewidth=1.0,
        alpha=0.8,
    )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Train Loss (L1 total)", fontsize=12)
ax.set_title("Tier-1 3-Way: Train Loss vs Epoch", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "train_loss.png"), dpi=150)
plt.close(fig)
print("Saved train_loss.png")

print("\nAll figures saved to", OUT_DIR)
PYEOF
