"""teacher forward 정밀도 선택 — 속도 vs feature 보존 실측.

precompute가 **GPU-bound**로 나왔다(4 shard로 24 s/it, GPU util 100 %). ViT-L의 matmul이
기본 fp32라 TF32/bf16 tensor core를 전혀 못 쓰고 있어서다. 다만 teacher feature는 REPA의
정렬 타깃이므로 마음대로 낮출 수 없다 — **상대 구조(cosine·Gram)가 보존되는지**를 먼저 재고 고른다.

판정 기준 (모두 fp32 기준선 대비):
  · token-wise cosine 평균 ≥ 0.9999  AND 최솟값 ≥ 0.999   → 정렬 타깃으로 안전
  · SRSS(폐 vs 그 외) 변화 ≤ 0.001                        → 공간 구조 지표 불변
  · 저장은 어차피 fp16이므로 fp16 저장 왕복 오차도 같은 표에 함께 보고한다.

실행:
    CUDA_VISIBLE_DEVICES=2 python -m tests.repa_probe.u2b_io.precision
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch

from tests.repa_probe._spectre import (
    CKPT_SSL,
    build_backbone,
    load_volume,
    lung_occupancy,
)

SCAN_ID = "valid_1000_a_1"
RESULT_DIR = Path(__file__).parent / "results"


def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


@contextlib.contextmanager
def _mode(name: str):
    """정밀도 모드를 세팅하고, 끝나면 기본값(fp32, TF32 off)으로 되돌린다."""
    if name == "fp32":
        _set_tf32(False)
        yield contextlib.nullcontext()
    elif name == "tf32":
        _set_tf32(True)
        yield contextlib.nullcontext()
    elif name in ("bf16", "fp16"):
        _set_tf32(True)
        dtype = torch.bfloat16 if name == "bf16" else torch.float16
        yield torch.autocast("cuda", dtype=dtype)
    else:
        raise ValueError(name)
    _set_tf32(False)


def srss(dense: torch.Tensor, lung_occ: torch.Tensor) -> float:
    """폐 내부 anchor 기준 (폐 평균 cos − 비폐 평균 cos). U0 스모크와 같은 정의."""
    feat = torch.nn.functional.normalize(
        dense.reshape(-1, dense.shape[-1]).float(), dim=-1
    )
    lung = lung_occ.reshape(-1) > 0.5
    anchor = int(torch.nonzero(lung_occ.reshape(-1) >= 0.999).flatten()[0])
    sim = feat @ feat[anchor]
    other = ~lung
    other[anchor] = False
    return float(sim[lung].mean() - sim[other].mean())


def main() -> None:
    backbone = build_backbone(CKPT_SSL, device="cuda", with_combiner=False)
    vol = load_volume(SCAN_ID)
    crops, grid = backbone.window(vol)
    crops = crops.cuda()
    lung = lung_occupancy(SCAN_ID)

    rows: list[dict] = []
    ref: torch.Tensor | None = None
    for name in ("fp32", "tf32", "bf16", "fp16"):
        with _mode(name) as amp:
            with amp:  # warmup — 첫 호출은 커널 오토튠 때문에 느리다
                backbone.encode_crops(crops[:16], (1, 1, 16))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with amp:
                dense, _ = backbone.encode_crops(crops, grid)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

        dense = dense.float().cpu()
        row = {
            "mode": name,
            "forward_s": round(dt, 3),
            "srss": round(srss(dense, lung), 6),
        }
        if ref is None:
            ref = dense
            row |= {"cos_mean": 1.0, "cos_min": 1.0, "max_abs_diff": 0.0}
        else:
            a = torch.nn.functional.normalize(ref.reshape(-1, ref.shape[-1]), dim=-1)
            b = torch.nn.functional.normalize(
                dense.reshape(-1, dense.shape[-1]), dim=-1
            )
            cos = (a * b).sum(-1)
            row |= {
                "cos_mean": round(float(cos.mean()), 8),
                "cos_min": round(float(cos.min()), 8),
                "max_abs_diff": round(float((ref - dense).abs().max()), 5),
            }
        rows.append(row)
        print(row, flush=True)

    # 저장 dtype(fp16) 왕복 오차를 같은 축에서 보고 — 어차피 이만큼은 감수하고 있다.
    a = torch.nn.functional.normalize(ref.reshape(-1, ref.shape[-1]), dim=-1)
    b = torch.nn.functional.normalize(
        ref.to(torch.float16).float().reshape(-1, ref.shape[-1]), dim=-1
    )
    cos = (a * b).sum(-1)
    store = {
        "mode": "fp32 → fp16 저장 왕복",
        "forward_s": None,
        "srss": round(srss(ref.to(torch.float16).float(), lung), 6),
        "cos_mean": round(float(cos.mean()), 8),
        "cos_min": round(float(cos.min()), 8),
        "max_abs_diff": round(
            float((ref - ref.to(torch.float16).float()).abs().max()), 5
        ),
    }
    rows.append(store)
    print(store)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "precision.json"
    out.write_text(json.dumps({"scan_id": SCAN_ID, "rows": rows}, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
