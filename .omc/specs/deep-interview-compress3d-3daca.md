# Deep Interview Spec: Compress3D 3D-aware Cross Attention for MAISI Triplane AE

## Metadata
- Interview ID: compress3d-3daca-2026-05-21
- Rounds: 5 (early-exit due to user deadline)
- Final Ambiguity Score: 39%
- Type: brownfield
- Generated: 2026-05-21
- Threshold: 20%
- Status: **BELOW_THRESHOLD_EARLY_EXIT** — user has a presentation deadline tomorrow morning; remaining design micro-decisions locked with sensible defaults.
- Initial Context Summarized: yes (user clarification rolled into state)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.2975 |
| Constraint Clarity | 0.60 | 0.25 | 0.15 |
| Success Criteria | 0.55 | 0.25 | 0.1375 |
| Context Clarity (brownfield) | 0.85 | 0.15 | 0.1275 |
| **Total Clarity** | | | **0.6125** |
| **Ambiguity** | | | **0.3875 (~39%)** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| 3D Feature Volume (K/V source) | active | What 3D volume feeds K/V into cross-attention. | K/V = MAISI latent directly. If triplane branch is downsampled, align via 3DConv(stride=o). |
| 3D-aware Cross-Attention Block | active | Core new module. | Single fire after ResBlocks, before μ/σ heads. m=1 (column-only). TriConv for Q. Learnable axial 3D PE on K/V. 4 heads, d_kv=16. |
| Pipeline Integration | active | Encoder backbone, μ/σ, decoder. | Encoder: 3-axis 3D conv (paper Eq 3-5) → ResBlocks ×2/×2/×4. Downsampling OFF default (option ON later). μ/σ: VAE with KL=1e-6 (paper). Decoder: A (existing broadcast+sum+conv, kept as baseline) + D (new TriConv+broadcast, main candidate). |
| Experiment & Evaluation Plan | active | Tier 1, 500 latents, 60min cap. | First sweep cells: {A baseline (existing), D no-attn, D w/ attn (m=1)}. Optional 4th: {D w/ attn (m=3 cube)} if time allows. Headline metric: latent PSNR delta between D no-attn and D w/ attn. Pass if delta > +0.5 dB. Promotion to Tier-2 = best with-attn cell. |

## Goal
Build a Compress3D-style triplane autoencoder over MAISI 3D latents `[B,4,120,120,64]`, where the encoder is a **3-axis 3D conv → ResBlocks ×2/×2/×4 → 3D-aware Cross-Attention (Q=triplane, K/V=MAISI latent with learnable axial 3D PE) → μ/σ heads (VAE, KL=1e-6)**, and the decoder is the paper-flavored variant **D** (TriConv ResBlocks + Upsample on 2D triplanes → broadcast+fuse → Conv3d back to MAISI latent shape). Validate by training on 500 MAISI latents in a Tier-1 sweep (60min cap) and reporting the latent-PSNR delta from cross-attention.

## Constraints
- Downsampling OFF by default; OPTIONAL flag for future runs (paper has ½ → ½ stages with ResBlock×4 each).
- Cross-Attn fires ONCE per forward, between ResBlock stack and μ/σ heads.
- TriConv (Rodin) must be ported from `https://github.com/RodinHD/RodinHD/blob/main/pretrained_diffusion/nn.py` — used both in Q construction and in Decoder D ResBlocks.
- Triplane latent must remain diffusion-friendly (downstream future): VAE with light KL, per-channel mean/std/KL-to-N(0,1) tracked as secondary metrics.
- Decoder A (existing `src/models/triplane_decoder.py`) is preserved as a comparison baseline; do not modify it.
- All experiment artifacts under `runs/<exp_name>/{checkpoints,figs,logs,hydra}/` per project convention.
- `CUDA_VISIBLE_DEVICES=0` for every training command (single GPU per CLAUDE.md convention).
- Hard wall-clock cap per Tier-1 run: 60 min.

## Non-Goals
- Training a latent diffusion model (downstream, separate phase).
- Aggressive compression / downsampling sweeps (later phase).
- Multi-scale cross-attention (paper uses single-scale; we follow).
- TriConv vs Conv2d ablation (TriConv is the canonical choice; ablation deferred).
- m>1 cube attention rigorous study (optional 4th cell only if time allows).
- Distribution-metric formal validation for diffusion (informally logged).

## Acceptance Criteria
- [ ] Pytest tier-0 passes for `TriplaneAE` with the new encoder + decoder D (single-batch overfit, relative loss < 0.3× initial within 250 steps).
- [ ] Pytest verifies cross-attention is differentiable and respects axial PE precedent (permutation-non-equivariance under axis-collapsed test).
- [ ] At least 3 Tier-1 runs complete within 60min each: {A existing baseline, D no-attn, D w/ attn (m=1)}.
- [ ] wandb logs latent PSNR, latent L1, per-channel mean/std, step time for every run.
- [ ] Headline result figure compares the 3 runs’ latent PSNR over training steps, with horizontal reference lines at 25.66 dB (current TriplaneAE plateau) and 27.39 dB (Conv baseline).
- [ ] Cross-attention pass criterion: `D w/ attn` latent PSNR exceeds `D no-attn` by ≥ +0.5 dB at the same step budget.
- [ ] Architecture diagram, training curves, and headline delta number ready for presentation slides.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Encoder follows paper §3.1 backbone | User confirmed 3-axis 3D conv + ResBlocks ×2/×2/×4; downsampling OFF default | Locked. Downsampling kept as optional flag. |
| K/V volume source = a separately-encoded 3D feature volume | User said: K/V source = MAISI latent directly (we already have a 3D feature volume as input) | Locked. 3DConv(stride=o) alignment only when triplane branch is downsampled. |
| Triplane latent is deterministic | User pointed out MAISI is itself a VAE output AND triplane latent will feed a future diffusion model → regularization needed | Locked: VAE with KL=1e-6 (paper). |
| Decoder: pick one family | User wanted A+D parallel after reviewing 4 reference iterations | Locked: D as main candidate, A (existing code) as comparison baseline. |
| Tier-1 sweep should cover all design axes | Time pressure (presentation tomorrow); user picked cross-attn ablation focus | Locked: 3–4 cells, cross-attn isolated as the test variable. |
| PE for cross-attn K/V volume | Paper says "PE" but doesn't specify form; user codebase precedent is 1D axial learnable PE | Locked: learnable axial 3D PE (X 1D + Y 1D + Z 1D sum). |
| Success criterion is absolute PSNR > 27.39 | User picked Δ-centric framing: cross-attn is the test variable, beating Conv baseline is secondary | Locked: headline = Δ PSNR between with-/without-attn; pass if > +0.5 dB. |

## Technical Context (brownfield)
- Encoder to replace: `src/models/triplane_encoder.py:TriplaneEncoder` (currently patchify Conv3d + z_init token + per-plane TransformerEncoder).
- Decoder A (keep as baseline, no edits): `src/models/triplane_decoder.py:TriplaneDecoder` (broadcast + sum + ResBlock3D + ConvTranspose3d).
- AE wrapper to update: `src/models/triplane_ae.py:TriplaneAE`.
- Reference implementations consulted: `reference/models/trivae.py` (per-axis 3D ladder), `reference/models/trivae2.py:CrossPlaneMixer` (cross-plane mixing pattern), `reference/models/trivae3.py:TriPlaneImplicitDecoder` (implicit MLP — not adopted), `reference/models/trivae4.py:PlaneExpandSum,RefineDecoder3D` (decoder A pattern).
- TriConv (Rodin) port target: new module, candidate path `src/models/tri_conv.py`.
- Training entry: `scripts/train.py` with Hydra config under `src/configs/`.
- Pytest: `tests/test_tier0_overfit.py` and shape/grad sanity tests under `tests/`.
- Baselines on record: current TriplaneAE plateau 25.66 dB; TriVQAEConv 27.39 dB; MAISI VAE upper bound (CT recon) 30.94 dB ± 2.97 dB (from `results/upper_bound.json`).

## Ontology (Key Entities — final round)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| MAISI Latent | input | [B,4,120,120,64], fp16, μ from MAISI VAE | input to Encoder; K/V source for Cross-Attn |
| 3-axis 3D Conv | encoder stem | k=(.,.,r),(r,.,.),( .,r,.) per axis | produces high-res triplane features |
| Triplane Feature | intermediate | 3 planes (XY,YZ,XZ), plane_channels=16, spatial = (120,120)/(120,64)/(120,64) | consumed by ResBlocks then Cross-Attn |
| Cross-Attn Block | core new module | Q (TriConv), K/V (3DConv on MAISI latent + axial 3D PE), m=1 local window, 4 heads, d=16 | enhances Triplane Feature; reads MAISI Latent |
| Triplane Latent | encoder output | (μ, log σ) per plane, plane_channels=16 | consumed by Decoder D; future diffusion target |
| Decoder D | new decoder | TriConv ResBlocks (×N) on 2D triplanes → broadcast+fuse → Conv3d | outputs MAISI Latent recon |
| Decoder A | baseline | existing broadcast+sum+conv | kept untouched as comparison cell |
| MAISI Decoder (frozen) | external | takes MAISI latent → CT volume | downstream eval only |
| Latent Diffusion Model (future) | downstream | trained later on triplane latent | drives the VAE design choice |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|-----------------|
| 0 | 6 | 6 | - | - | N/A |
| 2 | 7 | 1 (Latent Diffusion future) | 0 | 6 | 86% |
| 5 (final) | 9 | 2 (Decoder D, Decoder A split; 3-axis 3D Conv split out) | 1 (Cross-Attn fields expanded) | 6 | 78% |

## Interview Transcript
<details>
<summary>Full Q&A (5 rounds)</summary>

### Round 0 — Topology
**Q:** Confirm 4-component topology (Volume / Cross-Attn / Integration / Experiment plan).
**A:** User reframed scope: input = MAISI latent; encoder = 3-axis 3D conv + ResBlocks ×2/×2/×4; downsampling OFF default; Cross-Attn K/V source = MAISI latent. μ/σ extraction and decoder design are open; collaborator reference implementations should be reviewed. TriConv to be ported from Rodin.
**Outcome:** Topology locked, scope sharpened.

### Round 1 — Decoder Family
**Q:** Which decoder family — A (broadcast+3D refine), B (3D ladder + CrossPlaneMixer), C (implicit MLP), D (TriConv+broadcast)?
**A:** A+D parallel (recommended).
**Score after:** Ambiguity 62% → 54%.

### Round 2 — μ/σ Extraction
**Q:** Deterministic AE / VAE(KL=1e-6) / hybrid?
**A:** User clarified: MAISI is itself a VAE, AND triplane latent will feed a future diffusion model — so VAE with light KL.
**Score after:** Ambiguity 54% → 46%. Project-memory updated.

### Round 3 — First Experiment Matrix
**Q:** Tier-1 first sweep — Cross-Attn ablation / Decoder family compare / Regularization sweep / Downsampling sweep?
**A:** Cross-Attn ablation focused: {D no-attn, D w/ attn (m=1), D w/ attn (m=3)}, A as 4th comparison cell.
**Score after:** Ambiguity 46% → 43%.

### Round 4 — Position Embedding for K/V Volume
**Q:** Learnable axial 3D PE / Learnable dense 3D / Sinusoidal 3D / no PE?
**A:** Learnable axial 3D PE (matches commit 067622b precedent).
**Score after:** Ambiguity 43% → 39%.

### Round 5 — Success / Promotion Criteria
**Q:** Cross-attn Δ-centric / Conv baseline beat / Multi-metric Pareto / Rank-then-decide?
**A:** Cross-attn Δ-centric (pass if Δ > +0.5 dB).
**Early exit:** User declared presentation deadline tomorrow morning; further detail rounds skipped, remaining micro-decisions locked with sensible defaults.

</details>

## Locked Defaults (post-early-exit)
- `plane_channels` = 16
- Cross-Attn `heads` = 4, `d_kv` = 16
- `m` = 1 (column-only window); m=3 cube as optional 4th cell only if compute time permits
- KL weight = 1e-6
- Loss = L1 + KL(1e-6)
- Cross-Attn fires once, after ResBlocks, before μ/σ heads
- TriConv used in Q construction and in Decoder D ResBlocks
- Downsampling OFF, optional flag for future
- Tier-1 cap: 60 min wall-clock, 500 latents, `CUDA_VISIBLE_DEVICES=0`
- wandb project: existing project, run names follow `tier1_compress3d_{cell_id}`
