# Wan2.1-latent runbook (report2ct_wan)

`report2ct_wan` = the **real report2ct** (2560-d findings+impression conditioning + RFlow + varied
per-scan spacing) with **only the VAE latent swapped MAISI → Wan2.1** (diffusers `AutoencoderKLWan`,
8×8×4 compression, latent ch=16). It is NOT the CLIP3D-768 text2ct twin — the only variable vs the
`report2ct`(MAISI) baseline is the latent space. Plan: `/root/.claude/plans/temporal-shimmying-axolotl.md`.

## Geometry (locked)
CT resampled to **512×512×253** → Wan latent **(16, 64, 64, 64)** = (C, D_lat, H, W), stored as
`<id>_emb.nii.gz` in HWDC **(64, 64, 64, 16)** float32.
- in-plane **512** (not 480): Wan 8× + MAISI-UNet ÷8 ⇒ input must be ÷64 (480 → latent 60, breaks UNet).
- depth **253** (`≡1 mod 4`): Wan causal ×4 temporal lossless (253→64→253); latent depth 64 (÷8 ✓);
  ≈ CT-RATE native z (~260) ⇒ best reconstruction fidelity + matches MAISI's 256-slice z-density.
- **Reconstruction ceiling (verified, N=30 valid): PSNR 33.45 ± 4.70 dB / SSIM 0.814** — beats the
  MAISI VAE ceiling (30.94 / 0.720). Even the worst volume (26.4 dB) is anatomically clean.

## Environment (isolated) + GPU
- Wan VAE needs `diffusers ≥ 0.34` → dedicated **`wan` conda env** (`/opt/conda/envs/wan/bin/python`,
  diffusers 0.39). **Only encode (precompute) + decode run in the `wan` env.** Training + RFlow
  latent generation run in the **main env** (never imports diffusers/Wan — trains on precomputed latents).
- `HF_HOME=/workspace/data/checkpoints/hf_cache` so the Wan VAE loads from the project cache.
- **GPU**: train on **GPU 1**. GPU 0 is usually another job; GPU 2 is free-ish (use for precompute
  parallelism + eval while training runs on GPU 1). Simultaneous-init race → **stagger** worker starts.

---

## Step 1 — precompute Wan latents (wan env, user-owned)
Single output dir for both splits (`train_*`/`valid_*` filenames distinguish them); `meta.json`
records the geometry and is verified on reruns (cache-misuse guard). Resumable (`--skip-existing`).
```bash
# per shard (interleaved, disjoint): GPU1 gets shard 0-5, GPU2 gets 6+, staggered ~5s apart.
CUDA_VISIBLE_DEVICES=<1|2> HF_HOME=/workspace/data/checkpoints/hf_cache \
  /opt/conda/envs/wan/bin/python scripts/precompute_wan_image_embeddings.py \
    --ids-file /workspace/data/ctrate_toy_v2/{train,valid_v2}/ids.json \
    --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/{train,valid}_fixed \
    --out-dir /workspace/data/report2ct_wan/latents_512x512x253 \
    --depth 253 --num-shards <N> --shard <s> --device cuda:0
```
Driver that shards across GPU1+GPU2 with stagger: `scratchpad/run_wan_precompute.sh` pattern.
~7 s/vol encode; 2-GPU ≈ 17 latents/min → 6304 vols (5000 train + 1304 valid_v2) in ~6-7 h. ~15 MB/vol (~93 GB).
**Status: DONE — 6304/6304 on disk.**

## Step 2 — build datalist (main env; reuses the report2ct 2560 datalist builder)
The 2560 findings+impression text context is latent-agnostic, so the SAME `text_embeddings/*multi_2560.json`
used by report2ct(MAISI) pair with the Wan latents. `build_report2ct_datalist.py` reads the Wan
`_emb.nii.gz` header for the **varied per-scan spacing** and merges the text JSON — no change needed.
```bash
python scripts/build_report2ct_datalist.py \
  --image-dir /workspace/data/report2ct_wan/latents_512x512x253 \
  --text-dir  /workspace/data/report2ct_work_dir/text_embeddings \
  --ids-train /workspace/data/ctrate_toy_v2/train/ids.json \
  --ids-valid /workspace/data/ctrate_toy_v2/valid_v2/ids.json \
  --out /workspace/data/report2ct_wan/datalist_wan_2560.json
```
**Verified**: 5000 train + 1304 valid, 0 skipped. Spacing distribution in-plane med 0.72, z med 1.33
(≈ MAISI 1.31); eval 0.73/0.73/1.34 is in-distribution. **Status: DONE** (`datalist_wan_2560.json` at the
config's path `/workspace/data/report2ct_wan/`).

## Step 3 — train (main env, GPU 1) — USER-RUN
`experiment=report2ct_wan`: **300 epochs**, batch 8, lr 2e-4 (+500-step linear warmup), cfg_drop 0.15
**per-sample**, **no grad-clip** (grad-norm is logged as `train/grad_norm` — observed ~1–4, stable, so
clipping unwarranted). ~7 min/epoch. Checkpoints every 10 ep to `outputs/report2ct_wan/<KST-date>/checkpoints/`.
```bash
cd /workspace
CUDA_VISIBLE_DEVICES=1 nohup python src/train.py experiment=report2ct_wan \
  > logs/report2ct_wan_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# resume after a crash (restores weights+optimizer+LR+epoch):
CUDA_VISIBLE_DEVICES=1 python src/train.py experiment=report2ct_wan \
  ckpt_path=outputs/report2ct_wan/<ts>/checkpoints/last.ckpt
```
Why 300 (not 100): batch 8 ⇒ only 62k gradient steps at 100 ep (MAISI's 100 ep = 250k steps). 300 ep ≈
187k steps. Horizon set upfront so PolynomialLR spans it. **Judge by checkpoint FID, not loss** —
diffusion loss plateaus early (noise floor) while sample quality keeps improving; stop when FID plateaus.

## Step 4 — eval (3 sub-steps: generate → decode → score) — USER-RUN
Wan decode must run in the `wan` env, so generation (main env, RFlow → latents) and decode (wan env,
latents → .mha) are split; then `run_eval` scores the pre-decoded `.mha`. `run_eval` internally uses
`out_dir/predictions` — so decode writes there and `run_eval` gets `out_dir=$OUT` (verified `run_eval.py:55-56,88`).
```bash
CKPT=outputs/report2ct_wan/<ts>/checkpoints/epoch_099.ckpt
# Eval dir name MUST encode ep / sp / cfg (convention — see CLAUDE.md):
OUT=/workspace/outputs/report2ct_wan/eval_ep099_sp0.73_1.34_cfg5
# 4a. main env: RFlow-denoise → <OUT>/latents/*.npy   (--spacing + --cfg-scale REQUIRED, no defaults; n-steps default 100)
CUDA_VISIBLE_DEVICES=2 python scripts/generate_wan_latents.py --ckpt $CKPT --out $OUT --n 1304 \
  --spacing 0.73 0.73 1.34 --cfg-scale 5
# 4b. wan env: decode latents → <OUT>/predictions/*.mha
CUDA_VISIBLE_DEVICES=2 HF_HOME=/workspace/data/checkpoints/hf_cache \
  /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \
    --latent-dir $OUT/latents --out $OUT/predictions --spacing 0.73 0.73 1.34
# 4c. main env: score FID/CLIP (sampler is a pass-through over the pre-decoded .mha)
python scripts/run_eval.py task=ctgen model=report2ct_wan out_dir=$OUT task.n_samples=1304
```
Quick check: `--n 100` + `task.n_samples=100`. Spacing sweep: pass `--spacing 0.70 0.70 1.34` (and 0.80)
to 4a+4b — eval spacing is decoupled from training (varied-spacing model), so pick the FID optimum post-hoc.
Why 0.73 in-plane: 512×0.73 ≈ 374 mm FOV = the Wan FID optimum, matching the data median (0.72).
(report2ct/MAISI's own optimum was 384 mm at 480×0.8 — this 0.73/1.34 optimum is Wan-specific.)

## Step 5 — MAISI baseline comparison (clean baseline already exists — NO retrain)
`outputs/report2ct/2026-06-20_toy_v2` is the **clean** report2ct(MAISI) toy_v2 run (datalist_v2, no
`CONTAMINATED.md`, 2560 cond, batch 2, 100 ep, checkpoints epoch 9–99 + last).
⚠ The old eval `outputs/report2ct/eval_2026-06-22` used **cfg 1.0 + old spacing** (CLIPScore 16.7 = floor,
FID_Avg 8.12) — **NOT reusable**. Re-run at the FID-optimal spacing 0.8 / cfg 5 (now REQUIRED on the CLI —
report2ct-family configs set `spacing_mm`/`cfg_scale` = `???`, no default):
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_eval.py task=ctgen model=report2ct \
  model.ckpt_path=/workspace/outputs/report2ct/2026-06-20_toy_v2/checkpoints/epoch_099.ckpt \
  model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0 \
  task.n_samples=1304 out_dir=/workspace/outputs/report2ct/eval_ep099_sp0.8_cfg5
```
Comparison axes (both models have multi-epoch checkpoints → sliceable post-hoc): **best-vs-best**
(challenge), **same wall-clock / compute** (efficiency story — Wan is ~faster/epoch), **same gradient-steps**
(removes the batch-2-vs-8 confound: MAISI 100 ep = 250k steps ≈ Wan 400 ep; MAISI ep75 ≈ Wan 300 ep in steps).

## report2ct_wan_mask_v2 — dual-condition CFG (learned null mask + IP2P 5/5/5) — USER-RUN
Mask-conditioned Wan variant where the 4-organ Wan mask latent is **classifier-free-guided** (not just
concatenated). vs `report2ct_wan_mask`: (a) IP2P 5/5/5 dropout of TEXT + MASK during training, (b) a
**learned null mask embedding** (`no_mask_embed`, SAM pattern — small-random init, NOT zeros/air) swapped
in for dropped masks, (c) text-inner / mask-outer 3-pass CFG at eval:
`pred = e(∅m,∅t) + s_t·[e(∅m,t)−e(∅m,∅t)] + s_m·[e(m,t)−e(∅m,t)]`. **s_m is a clean mask-effect dial**:
`s_m=0` ignores the mask (pure text→CT), `s_m=1` natural, `s_m>1` amplifies organ geometry.
Reuses the SAME mask latents as `report2ct_wan_mask` (`scripts/precompute_wan_mask_latents.py`) — no
extra precompute (the null is created + learned in-module, no all-background encode).
```bash
cd /workspace
# Train (300 ep, batch 8; dropout is dual-condition, model.cfg_drop_prob/cfg_per_sample are IGNORED):
CUDA_VISIBLE_DEVICES=1 nohup python src/train.py experiment=report2ct_wan_mask_v2 \
  > logs/report2ct_wan_mask_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Eval — sweep s_m at s_t=5. Dir name encodes ep / sp / BOTH scales: eval_ep<NNN>_sp<in>_<z>_cfgt<T>_m<M>
CKPT=outputs/report2ct_wan_mask_v2/<ts>/checkpoints/epoch_099.ckpt
MASKS=/workspace/data/report2ct_wan/mask_latents_512x512x253
for SM in 0 1 1.5 2; do
  OUT=/workspace/outputs/report2ct_wan_mask_v2/eval_ep099_sp0.73_1.34_cfgt5_m${SM}
  # 1. main env: 3-pass dual-CFG RFlow → <OUT>/latents/*.npy  (--cfg-scale-text/-mask + --spacing REQUIRED)
  CUDA_VISIBLE_DEVICES=2 python scripts/generate_wan_mask_v2_latents.py --ckpt $CKPT --out $OUT \
    --mask-dir $MASKS --n 1304 --cfg-scale-text 5 --cfg-scale-mask $SM --spacing 0.73 0.73 1.34
  # 2. wan env: decode → <OUT>/predictions/*.mha
  CUDA_VISIBLE_DEVICES=2 HF_HOME=/workspace/data/checkpoints/hf_cache \
    /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \
      --latent-dir $OUT/latents --out $OUT/predictions --spacing 0.73 0.73 1.34
  # 3. main env: score (pass-through over the pre-decoded .mha)
  python scripts/run_eval.py task=ctgen model=report2ct_wan_mask_v2 out_dir=$OUT task.n_samples=1304
done
```
`s_m=0 vs 1 vs >1` on {2.5D-FID, CLIPScore-T2I, FVD} = the mask's measured effect; organ adherence via
`tests/abnormality_fidelity/` Dice should climb with s_m. (The ckpt MUST have `no_mask_embed` — the
generate step raises otherwise; a plain `report2ct_wan_mask` ckpt is incompatible.)

## Notes / gotchas
- **Latent normalization**: the adapter applies Wan's per-channel `latents_mean/std` at encode (diffusers
  returns raw μ; the pipeline normally does this). Report2CTModule's scalar `scale_factor=1/std` then lands
  ≈1 — different layers, compose without conflict, no double-normalization. (`src/baselines/wan_vae.py`.)
- **HU on decode**: Wan maps [-1,1] → *1000 (not MAISI's *2000-1000).
- **Spacing**: trained on TRUE per-scan resampled spacing (from the `_emb.nii.gz` header); eval stamps a
  single value (`--spacing`, now REQUIRED — no default) for UNet conditioning AND the .mha affine — they must match.
- **Precompute stagger**: launch shard workers ~5 s apart — a simultaneous CUDA/VAE-init race stalls workers at 0.
- **kill pattern trap**: never `pkill -f run_wan_precompute` / `-f src/train.py` in a command whose own argv
  contains that string (self-kill). Use a bracket pattern `[r]un_...` and keep launch strings out of the kill command.
