---
paths:
  - "src/baselines/**"
  - "src/eval/**"
---

# Baseline / model-clone audit checklist

When cloning a baseline or wiring a third-party model into our inference path, eyeball ALL of these — they're the silent bugs that don't crash but produce wrong outputs:
1. **Activation-range handoff between stages.** A→B chain (e.g. low-res→super-res, VAE-encode→diffuse→VAE-decode): does B's input range match A's output? Missing affine (`(x+1)/2` etc.) yields plausible-looking but biased HU.
2. **Sampler loop boundary.** Diffusion / RFlow `for t, next_t in zip(timesteps, timesteps[1:] + [0])` — many ports drop the final `→0` step and leave residual noise.
3. **Config vs. hardcode drift.** If the upstream reference script hardcodes a literal (e.g. `layer_cross_attns=[F,F,T]`), trust the hardcode over your YAML — OmegaConf `ListConfig` may silently coerce through `cast_tuple` and feed a different shape to the checkpoint.
4. **Axis order at I/O boundaries.** PyTorch `(B,C,D,H,W)` vs SimpleITK `(Z,Y,X)` vs NIfTI `(X,Y,Z)`. Track shape comments through every `.squeeze` / `.permute`.
5. **Tiled decoders.** Large-volume VAE decode: if upstream uses `SlidingWindowInferer(roi, overlap, mode="gaussian")`, replicate it — direct decode can OOM or produce different boundary statistics.
6. **HU save format.** `clip → rescale → clip` order, dtype (`int16` vs `float32`), `SetSpacing`/`SetOrigin`. Copy upstream's save helper byte-for-byte.
7. **Mixed precision.** Match the training `precision` (e.g. `bf16-mixed`) at inference with `torch.amp.autocast(...)`. Many MONAI/MAISI layers (`MaisiGroupNorm3D` with `norm_float16: true`) emit fp16 internally and crash without autocast.
8. **Modality / class-label conditioning.** A class-conditional UNet trained with `class_labels=ones()` must see `class_labels=ones()` at inference; one-off label drift is silent.
