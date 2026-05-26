# Tier 1 architecture sweep — 3-way comparison

**Setup**: 2mm toy latents `[4, 60, 60, 32]`, 5000 train / 1000 valid, 60-min wall-clock cap each, GPU 0, patch_size=4 across all three. Per-arch batch sizes (Karpathy "compute-equal per-arch best"): ours=8, baseline=4, d3t=4. AMP bf16 on d3t only.

**Tier-1 ceiling** (informational, image-domain): MAISI VAE round-trip at 2mm, n=1000 → PSNR 26.63 ± 1.45 dB, SSIM 0.686 ± 0.085. Tier-1 metric below is LATENT-domain so the absolute number is not directly comparable to this ceiling.

## Final metrics

| arch | params | epochs | wall-clock (min) | final val PSNR | final val L1 | last-3-ep ΔPSNR | trajectory |
|------|--------|--------|------------------|----------------|--------------|-----------------|------------|
| **baseline** (TriVQAEConv) | 16M | 9 | 60.1 (cap hit) | **27.388** dB | 0.4239 | **+0.253** | still climbing |
| d3t (TriVAE_D3T width-½) | 2.94M | 99 | 56.1 | 26.012 dB | 0.4969 | +0.003 | plateau |
| ours (TriplaneAE std v1.0 width-½) | 5.5M | 50 | 41.6 (finished) | 25.660 dB | 0.5223 | +0.004 | plateau (very early) |

## Verdict

**baseline (TriVQAEConv) wins this Tier-1 sweep by a large margin** — +1.73 dB over ours, +1.38 dB over d3t — **and it is still climbing at the cap**. With more compute it would extend its lead.

Reading the absolute numbers without the trajectory column is misleading:
- baseline only reached 9 epochs in 60 min because its conv-heavy stack runs at 335 ms/step. Its progression slope is the steepest of the three (+0.18-0.30 dB/epoch sustained in the last few epochs).
- d3t reached 99 epochs (25 ms/step) and is fully plateaued. Its capacity ceiling at this width is ~26.0 dB.
- ours reached 50 epochs and finished BEFORE the cap (41.6 min) because validation overhead per epoch is small. Plateaued by epoch 5; remaining 45 epochs added only +0.4 dB. Capacity ceiling appears to be ~25.7 dB.

## Architectural lesson

| arch | type | params | per-compute efficiency | plateau height |
|---|---|---|---|---|
| baseline | 3D conv (ResBlock3D + axis-only down/up + cross-plane mixer) | 16M | low (slow per-step) | high (>27.3, not plateaued) |
| ours | patchified transformer (`_f_psi` over each plane) | 5.5M | medium | low (~25.7, very fast plateau) |
| d3t | Swin transformer (windowed self-attn) | 2.94M | high (fast per-step) | medium (~26.0) |

The transformer-style architectures (ours, d3t) plateau much earlier than the conv baseline despite having access to more Adam updates within the same wall-clock budget. This is the opposite of what the "transformer scales better" intuition would suggest — it likely points at a recipe issue (no LR schedule / warmup, L1-only loss) more than a model-capacity issue, since trial2 documented in `research_summary/summary.md` showed similar plateau behavior with much smaller transformer.

## Recommended next steps (not part of this report)

1. Re-run **ours** and **d3t** with: lr warmup, cosine decay, possibly L2 + perceptual loss. Check if the transformer plateau lifts.
2. Promote **baseline** to Tier 2 as the current frontrunner (would need longer cap to see where its plateau actually lies).
3. Treat this Tier-1 number for transformer archs as a "with vanilla L1+Adam" lower bound, not as a capacity ceiling.

## Artifacts

- Curves: `val_psnr.png`, `val_l1.png`, `train_loss.png` (this directory)
- Raw run data: `run_data.json` (full per-step history pulled from wandb)
- wandb runs:
  - ours: https://wandb.ai/jasonna24-/triplane-ae/runs/shucjhx5
  - baseline: https://wandb.ai/jasonna24-/triplane-ae/runs/fs4st1kd
  - d3t: https://wandb.ai/jasonna24-/triplane-ae/runs/kp4260wc
