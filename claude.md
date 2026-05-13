## Project goal
- Input: MAISI VAE latent (shape: [B, 4, 120, 120, 64])
- Encode: 3D latent → triplane latent (3 axis-aligned 2D planes: XY, YZ, XZ)
- Decode: triplane latent → 3D latent
- Reconstruction을 MAISI decoder로 통과시켜 최종 CT volume 복원
- 평가: PSNR / SSIM을 **MAISI 자체 reconstruction (upper bound)** 와 비교
- 목표: upper bound와의 gap을 최소화

## Environment
- DeepCGV-Mk7 (max 3× A6000 blackwell)
- Docker dev container
- Python 3.11, PyTorch 2.x, monai, wandb
- Dataset: CT-RATE

## Conventions
- Config system: Hydra
- Logging: wandb
- Checkpoint 위치: checkpoints/<exp_name>/

## Baselines
- Upper bound: MAISI VAE 자체 reconstruction PSNR/SSIM (이게 우리가 따라잡을 ceiling)
- Lower bound: identity (encode=decode=I) — 정상이라면 abysmal

### Measured upper bound (2026-05-12)
MAISI VAE encode→decode round-trip on all 1000 CT-RATE validation volumes: **PSNR = 30.94 ± 2.97 dB, SSIM = 0.7195 ± 0.1084**. Intensity convention: HU clipped to [-1000, 1000] then scaled to [0, 1]; spatial size 480×480×256; decoder run via SlidingWindowInferer (latent_roi=(20,20,20), overlap=0.4, sw_batch_size=16, GPU stitching, torch.compile reduce-overhead). Full per-sample results and methodology notes in `/workspace/results/upper_bound.json`.

## Non-goals (이 프로젝트에서 안 함)
- Text conditioning
- Diffusion training
- 새 데이터셋 전처리 (CT-RATE preprocessed latent 사용)


# Behavioral guidelines
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.