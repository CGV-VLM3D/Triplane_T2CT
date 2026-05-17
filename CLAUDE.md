# Triplane Autoencoder for MAISI Latents

## Project goal
Encode MAISI VAE latent `[B, 4, 120, 120, 64]` → 3 axis-aligned 2D triplanes (XY, YZ, XZ) → decode back to 3D latent → pass through the frozen MAISI decoder to recover the CT volume. Evaluate PSNR/SSIM against MAISI's own round-trip reconstruction (upper bound); minimize the gap.

## Environment
DeepCGV-Mk7, up to 3× A6000 Blackwell. Docker dev container. Python 3.11, PyTorch 2.x, MONAI, wandb. Hydra configs.

## Dataset reference
Pipeline: raw CT NIfTI → resample to target spacing (default 1mm³) → MAISI preproc `[1, 480, 480, 256]` → MAISI VAE encode `[4, 120, 120, 64]` (fp16) → **our triplane AE** encodes/decodes this latent. MAISI VAE compression ratio is 4× per spatial axis. MAISI VAE is fully convolutional and used with sliding-window inference (default ROI 80³, see `maisi_bundle/configs/inference.json`), so input spatial dimensions are flexible — any size that is a multiple of 4 works. The VAE itself was trained on 64³/128³ random patches across diverse CT spacings (~0.5–2mm), so changing FOV or spacing at encode time is supported in principle; it has only mild robustness limits, not a hard fixed input size.

- **Raw CT (CT-RATE)** at [/workspace/datasets/datasets/CT-RATE/dataset/](datasets/datasets/CT-RATE/dataset/)
  - `train_fixed/`: 20,000 patient dirs, 47,148 scans (NIfTI, 512×512×~300, XYSpacing ~0.82mm, ZSpacing 1.0mm)
  - `valid_fixed/`: 1,304 patient dirs, 3,038 scans
  - `metadata/{train,validation}_metadata.csv`: 44 cols (manufacturer, age, spacing, kernel, ...)
  - `multi_abnormality_labels/`: 18 binary abnormality labels per scan
  - `radiology_text_reports/`: free-text radiology reports
  - `ts_seg/`, `anatomy_segmentation_labels/`: segmentation maps

- **Precomputed MAISI latents** at [/workspace/datasets/datasets/latents/](datasets/datasets/latents/)
  - 5,000 train / 1,000 valid (toy subset prepared by a collaborator)
  - Per sample: `mu.pt` `[4, 120, 120, 64]` fp16, `sigma.pt` (same shape), `src.txt` (pointer to source NIfTI)
  - `stats.json`: channel-wise mean/std for normalization. Loader: [src/data/maisi_latent_dataset.py](src/data/maisi_latent_dataset.py)
  - This is our main training source. **Read-only — do not write new artifacts here.**

- **Storage convention**: `/workspace/datasets/` is the collaborator's read-only data area. All new artifacts (toy-spacing latents, recon caches, intermediate tensors) go under [/workspace/data/](data/). Example: Path-B toy latents at `/workspace/data/latents_2mm/{train,valid}/<sample>/{mu.pt, sigma.pt, src.txt}` + `stats.json`.

- **GPU convention**: prefix every script and training command with `CUDA_VISIBLE_DEVICES=0` so only GPU 0 is used. Scripts pin `device = "cuda:0"` for the same reason.

## Conventions
- All experiment artifacts under `runs/<exp_name>/{checkpoints,figs,logs,hydra}/`.
- Project-wide baseline (`results/upper_bound.json`) stays at the top-level `results/`.
- Image metrics on `[0, 1]` domain; SSIM `win_size=11` (matches the upper-bound measurement).
- `sw_batch_size`: 64 on A6000 Pro (default), 8–16 on RTX 4090 — override with `eval.sw_batch_size=8`.

## Upper bound (measured 2026-05-12)
MAISI VAE encode→decode round-trip on the 1000 CT-RATE valid volumes: **PSNR 30.94 ± 2.97 dB, SSIM 0.7195 ± 0.1084**. Intensity: HU clipped to `[-1000, 1000]`, scaled to `[0, 1]`; spatial 480×480×256. Full details in `results/upper_bound.json`.

## Experiment iteration tiers
Goal: shorten the feedback loop from ~1 day to ~30–60 min per Tier-1 experiment. **Latent crop is the worst option** — it breaks triplane's global axis-aligned structure. Two viable paths:

- **Path A (safe, no precompute)**: keep `[4, 120, 120, 64]` latent. Reduce model width, data count, and wall-clock budget (μP / Chinchilla / Karpathy autoresearch convention). Keep model **depth unchanged** (depth-transfer is the most fragile under μP).
- **Path B (faster per step, needs MAISI re-encode)**: resample raw CT to coarser spacing (e.g. 2mm³) → MAISI encode → `[4, 60, 60, 32]` toy latent. Per-step compute drops ~8×. Validate first by measuring MAISI round-trip PSNR vs 1mm baseline (29 dB) on ~50 volumes; if drop is ≤1–2 dB the toy latent is usable.

Tier plan (using Path A by default; substitute Path B latents if validated):
- **Tier 0 sanity (<5 min)**: 8–16 samples, overfit single batch to ~0 loss. Wired into pytest.
- **Tier 1 architecture sweep (30–60 min, wall-clock capped)**: 500 latents (~10% of 5k), model width ½ (encoder `emb_dim` 512→256, decoder `hidden` 32→16), depth unchanged. Compare 5–10 candidates by **latent-domain reconstruction loss**.
- **Tier 2 mid validation (4–6 h)**: all 5,000 latents, full model width, ~15 epochs. Promote 2–3 candidates.
- **Tier 3 full run (1 day+)**: full data, full model, full epochs. 1–2 finalists only.

## Non-goals
Text conditioning, diffusion training, new dataset preprocessing.

## Post-eval workflow
After the `result-analyzer` subagent finishes for an experiment, invoke `research-summarizer` with the same `exp_name` to refresh `research_summary/summary.md`. A `SubagentStop` hook also surfaces this reminder.
