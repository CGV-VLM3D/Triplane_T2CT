# mask_wan_roundtrip — input mask vs WanVAE recon (report2ct_wan_mask)

Sanity check for the **report2ct_wan_mask** conditioning: does the frozen Wan VAE preserve the
painted **4-organ** mask geometry that the UNet is conditioned on?

Pipeline (per scan): ts_seg → `apply_grouping` (117 → {0=bg,1=lung,2=heart,3=aorta,4=esophagus}) →
paint `hu=-1000+500·g` → **WanVAE.encode** → `<id>_mask_emb.nii.gz` (this is what training concats).
This check **decodes** that latent back and compares to the painted input.

## Montage (`png/<id>.png`)
Each row = one axial level; columns:
- **INPUT mask (4-organ)** — the painted mask fed into the pipeline (lung=orange, heart=green,
  aorta=red, esophagus=blue, bg=black).
- **WanVAE RECON (HU)** — `WanVAE.decode(mask_latent)`, the 5 painted levels come back as gray levels.
- **RECON label** — recon quantized to the nearest painted level → essentially identical to INPUT.

## Result (per-organ Dice, INPUT vs RECON, n=4)
| id | lung | heart | aorta | esophagus |
|---|---|---|---|---|
| valid_1000 | 0.994 | 0.984 | 0.969 | 0.931 |
| valid_1001 | 0.995 | 0.983 | 0.975 | 0.938 |
| valid_1002 | 0.989 | 0.984 | 0.974 | 0.909 |
| valid_1005 | 0.993 | 0.986 | 0.965 | 0.923 |

→ The frozen Wan VAE reconstructs the 4-organ geometry at **Dice 0.91–0.99**, so the 16-ch mask
latent carries usable organ geometry for conditioning. (Esophagus is lowest — thinnest/smallest
organ, most lossy under 8× spatial compression — but still ~0.92.)

## Reproduce (`wan` conda env, GPU)
```bash
CUDA_VISIBLE_DEVICES=2 HF_HOME=/workspace/data/checkpoints/hf_cache \
  /opt/conda/envs/wan/bin/python tests/mask_wan_roundtrip/run.py
```
Reads the trained mask latents from `data/report2ct_wan/mask_latents_512x512x253/`; edit `IDS` in
`run.py` to visualize other scans.

> Not the same as the stale `data/report2ct_wan/mask_roundtrip_png/` (an earlier **multi-label**
> exploration, not the current 4-organ scheme).
