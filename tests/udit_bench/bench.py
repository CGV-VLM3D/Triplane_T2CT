#!/usr/bin/env python3
"""Phase 2a dummy benchmark — measure real activation memory + step time for the
hierarchical U-DiT backbone (`model.py`), to replace the analytical estimates in
docs/vlm3d_research_roadmap.md (~3.2 GB activations, MFU 25-30%) with measurements.

No real data, no text conditioning, no wiring into src/ or configs/ — see the
isolation principle in tests/repa_probe/README.md. Trains against a dummy MSE
target so the backward pass (and grad-checkpoint recomputation) is exercised
exactly as it would be by a real diffusion loss.

Usage:
  CUDA_VISIBLE_DEVICES=3 python tests/udit_bench/bench.py
  CUDA_VISIBLE_DEVICES=3 python tests/udit_bench/bench.py --batch-sizes 4,8,16 --checkpoint both
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from tests.udit_bench.model import HierarchicalUDiT3D

# Wan2.1 latent geometry this backbone is sized for — see
# scripts/precompute_wan_image_embeddings.py:11-12 and docs/wan_latent_runbook.md.
LATENT_SHAPE = (16, 64, 64, 64)  # (C, D, H, W)


def build_model(args: argparse.Namespace, use_checkpoint: bool) -> HierarchicalUDiT3D:
    return HierarchicalUDiT3D(
        in_channels=LATENT_SHAPE[0],
        dims=tuple(args.dims),
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window=args.window,
        use_checkpoint=use_checkpoint,
    )


def analytical_forward_flops(
    dims, depths, window, n0_tokens=32**3, n1_tokens=16**3, n2_tokens=8**3
):
    """Rough forward FLOPs estimate (matmuls only: attn qkv/proj + mlp + attn score/out).

    Same accounting convention used in the roadmap doc's hand-computed table, kept
    here only as a cross-check against the measured wall-clock throughput — NOT
    used for anything the benchmark reports as ground truth.
    """
    d0, d1, d2 = dims
    n0, n1, n2 = depths

    def block_flops(dim, n_tok, mlp_ratio=4.0, attn_tok=None):
        attn_tok = attn_tok or n_tok
        qkvo = 8 * n_tok * dim * dim  # qkv (3x) + proj, 4 * 2*N*d*d
        attn_score_out = (
            4 * n_tok * attn_tok * dim
        )  # scores + weighted sum, both directions
        mlp = (
            2 * 2 * n_tok * dim * int(dim * mlp_ratio)
        )  # 2 linears, fwd only factor 2*N*d*h
        return qkvo + attn_score_out + mlp

    l0 = (
        2 * n0 * block_flops(d0, n0_tokens, attn_tok=window**3)
    )  # x2: encoder+decoder mirror
    l1 = 2 * n1 * block_flops(d1, n1_tokens)
    l2 = n2 * block_flops(d2, n2_tokens)  # bottleneck runs once
    return l0 + l1 + l2, {
        "L0": 2 * n0 * block_flops(d0, n0_tokens, attn_tok=window**3),
        "L1": 2 * n1 * block_flops(d1, n1_tokens),
        "L2": l2,
    }


def run_one(
    device: str,
    batch_size: int,
    use_checkpoint: bool,
    args: argparse.Namespace,
    n_warmup: int,
    n_timed: int,
) -> dict:
    # set_device must precede the first cuda call in a process — otherwise
    # reset_peak_memory_stats("cuda:0") raises "Invalid device argument" (reproduced
    # directly against torch 2.7.0+cu128; empty_cache() alone doesn't establish
    # the context reset_peak_memory_stats needs).
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(args, use_checkpoint).to(device)
    model.train()
    n_params = model.param_count()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(batch_size, *LATENT_SHAPE, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    target = torch.randn_like(x)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x, t)
            loss = nn.functional.mse_loss(pred.float(), target)
        loss.backward()
        opt.step()
        return loss

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize(device)

    mem_after_warmup = torch.cuda.memory_allocated(
        device
    )  # ~params + optimizer state, steady state
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated(device)
    sec_per_step = elapsed / n_timed
    samples_per_sec = batch_size / sec_per_step
    activation_mem_per_sample = (peak_mem - mem_after_warmup) / batch_size

    total_flops, per_level = analytical_forward_flops(
        tuple(args.dims), tuple(args.depths), args.window
    )
    train_flops_per_sample = 3 * total_flops  # fwd + 2x bwd, standard convention
    achieved_tflops = (train_flops_per_sample * samples_per_sec) / 1e12

    return {
        "batch_size": batch_size,
        "use_checkpoint": use_checkpoint,
        "n_params": n_params,
        "steady_state_mem_gb": mem_after_warmup / 2**30,
        "peak_mem_gb": peak_mem / 2**30,
        "activation_mem_per_sample_gb": activation_mem_per_sample / 2**30,
        "sec_per_step": sec_per_step,
        "samples_per_sec": samples_per_sec,
        "analytical_fwd_tflop_per_sample": total_flops / 1e12,
        "achieved_tflops": achieved_tflops,
        "per_level_fwd_gflop": {k: v / 1e9 for k, v in per_level.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", default="4,8,16", help="comma-separated")
    p.add_argument("--checkpoint", choices=["on", "off", "both"], default="both")
    p.add_argument("--dims", default="384,768,1152")
    p.add_argument(
        "--depths", default="2,4,12", help="encoder-side counts; L0/L1 decoder mirrors"
    )
    p.add_argument("--num-heads", default="6,12,18")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-timed", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = p.parse_args()
    args.dims = [int(v) for v in args.dims.split(",")]
    args.depths = [int(v) for v in args.depths.split(",")]
    args.num_heads = [int(v) for v in args.num_heads.split(",")]

    batch_sizes = [int(v) for v in args.batch_sizes.split(",")]
    ckpt_modes = (
        [True, False] if args.checkpoint == "both" else [args.checkpoint == "on"]
    )

    print(
        f"device={args.device} | dims={args.dims} depths={args.depths} window={args.window}"
    )
    results = []
    for use_ckpt in ckpt_modes:
        for bs in batch_sizes:
            label = f"batch={bs:>3d} checkpoint={'on ' if use_ckpt else 'off'}"
            try:
                r = run_one(
                    args.device, bs, use_ckpt, args, args.n_warmup, args.n_timed
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{label}  OOM")
                results.append(
                    {"batch_size": bs, "use_checkpoint": use_ckpt, "oom": True}
                )
                torch.cuda.empty_cache()
                continue
            print(
                f"{label}  peak={r['peak_mem_gb']:.2f}GB "
                f"(act/sample={r['activation_mem_per_sample_gb'] * 1000:.1f}MB)  "
                f"{r['samples_per_sec']:.2f} samples/s  "
                f"{r['achieved_tflops']:.1f} TFLOPS"
            )
            results.append(r)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
