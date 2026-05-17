"""Probe which SDPA backends actually work on this GPU.

Tries each of {flash, mem_efficient, math, cudnn} individually on a tensor
shaped like the TriplaneEncoder's XY plane attention.  Reports per-backend
success, latency, and peak memory.
"""

from __future__ import annotations

import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


def _info() -> None:
    print(f"torch           : {torch.__version__}")
    print(f"cuda runtime    : {torch.version.cuda}")
    print(f"cudnn           : {torch.backends.cudnn.version()}")
    print(f"device          : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute cap     : sm_{cap[0]}{cap[1]}")
    print(f"flash available : {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"mem_eff avail   : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"cudnn sdp avail : {torch.backends.cuda.cudnn_sdp_enabled()}")
    print()


def _run_one(name: str, backend: SDPBackend, q, k, v) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with sdpa_kernel(backend):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) * 1000

            # backward path
            loss = out.float().sum()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            bwd_ms = (time.perf_counter() - t1) * 1000

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"  [{name:12s}] OK   fwd={fwd_ms:7.1f} ms  bwd={bwd_ms:7.1f} ms  "
            f"peak={peak_gb:5.2f} GiB"
        )
    except Exception as e:
        msg = str(e).splitlines()[0][:120]
        print(f"  [{name:12s}] FAIL {msg}")


def main() -> None:
    _info()

    B, H, D = 1, 8, 64
    cases = [
        ("XY  N=14400 fp16", 14400, torch.float16),
        ("XY  N=14400 bf16", 14400, torch.bfloat16),
        ("XY  N=14400 fp32", 14400, torch.float32),
        ("YZ  N= 7680 fp16", 7680, torch.float16),
    ]

    for label, N, dtype in cases:
        print(f"\n== {label}  (B={B}, H={H}, N={N}, D={D}) ==")
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

        for name, backend in [
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("mem_eff", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
            ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ]:
            _run_one(name, backend, q, k, v)


if __name__ == "__main__":
    main()
