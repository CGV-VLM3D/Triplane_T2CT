"""U0 스모크 — SPECTRE dense token을 wan latent와 같은 그리드에서 뽑을 수 있는지 실측 검증.

계획(U0)의 6개 검증 항목을 순서대로 실행하고 `results/smoke_spectre.json` + `figs/`에 남긴다.
판정은 [REPORT.md](REPORT.md).

  S0  `_crops_to_grid`가 `grid_patch`를 정확히 되돌리는지 — **모델 없이** 인덱스 왕복으로 증명
  S1  z end-pad 후 crop grid == (4,4,4) & center-crop으로 잘린 voxel == 0
  S2  crop당 token == 1 + 512, `num_prefix_tokens == 1`
  S3  CLS 제거 후 (32,32,32,1080) 재조립이 **해부학적으로** 맞는지 (ts_seg 폐 마스크 + cos-sim map)
  S4  `forward_intermediates`로 중간 layer 선택
  S5  combiner scan-level global shape
  S6  GPU 메모리 / 볼륨당 시간

실행:
    CUDA_VISIBLE_DEVICES=2 python -m tests.repa_probe.u0_smoke.run
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.spectre_adapter import SpectreBackbone, import_spectre
from tests.repa_probe._spectre import (
    AIR_HU,
    CKPT_SSL,
    DEPTH_PAD,
    EMBED_DIM,
    PATCH_SIZE,
    TOKEN_GRID,
    build_backbone,
    load_volume,
    lung_occupancy,
)

SCAN_ID = "valid_1000_a_1"
OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figs"
RESULT_DIR = OUT_DIR / "results"


def s0_index_roundtrip(window_scan) -> dict:
    """모델 없이 재조립 순서를 증명한다: 각 patch에 자기 인덱스를 심고 왕복시킨다.

    틀린 permutation은 여기서 즉시 걸린다 — crop 순서·patch flatten 순서·CLS drop 규칙을
    한꺼번에 고정하는, 이 스모크에서 가장 결정적인 검사.
    """
    idx = torch.arange(int(np.prod(TOKEN_GRID)), dtype=torch.float32).reshape(
        TOKEN_GRID
    )
    vol = idx
    for axis, rep in enumerate(PATCH_SIZE):
        vol = vol.repeat_interleave(rep, dim=axis)
    vol = vol.unsqueeze(0)  # (1, 512, 512, 256)

    crops, grid = window_scan(vol, SpectreBackbone.crop_size, scale_intensity=False)
    p = tuple(
        c // q for c, q in zip(SpectreBackbone.crop_size, PATCH_SIZE)
    )  # (8, 8, 8)
    n = crops.shape[0]
    per_patch = crops[:, 0].reshape(
        n, p[0], PATCH_SIZE[0], p[1], PATCH_SIZE[1], p[2], PATCH_SIZE[2]
    )
    per_patch = per_patch.mean(dim=(2, 4, 6)).reshape(n, int(np.prod(p)), 1)

    got = SpectreBackbone._crops_to_grid(per_patch, grid)[..., 0]  # (32, 32, 32)
    return {
        "crop_grid": list(grid),
        "n_crops": int(n),
        "roundtrip_exact": bool(torch.equal(got, idx)),
        "max_abs_err": float((got - idx).abs().max()),
    }


def s3_anatomy(grid_feat: torch.Tensor, lung_occ: torch.Tensor) -> dict:
    """폐 마스크로 SRSS-lite를 계산하고 anchor cos-sim map을 그린다.

    Args:
        grid_feat: `(32, 32, 32, 1080)` teacher token grid (fp32, CPU).
        lung_occ: `(32, 32, 32)` token별 폐 voxel 점유율 [0, 1].

    Returns:
        srss = mean(cos(anchor, lung)) − mean(cos(anchor, non-lung)). 재조립이 틀렸으면 0 근처.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    feat = torch.nn.functional.normalize(
        grid_feat.reshape(-1, grid_feat.shape[-1]), dim=-1
    )
    lung = lung_occ.reshape(-1) > 0.5
    if int(lung.sum()) < 32:
        raise RuntimeError(
            f"폐 token이 너무 적다 ({int(lung.sum())}) — 마스크/그리드 확인"
        )

    lung_idx = torch.nonzero(lung_occ.reshape(-1) >= 0.999).flatten()
    anchor = int(lung_idx[len(lung_idx) // 2])  # 폐 내부 깊숙한 token
    sim = feat @ feat[anchor]  # (32768,)

    other = ~lung
    other[anchor] = False
    srss = float(sim[lung].mean() - sim[other].mean())

    sim_grid = sim.reshape(*TOKEN_GRID)
    ai, aj, ak = (int(v) for v in np.unravel_index(anchor, TOKEN_GRID))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    planes = [
        (f"axial  (D={ak})", sim_grid[:, :, ak], lung_occ[:, :, ak]),
        (f"coronal (W={aj})", sim_grid[:, aj, :], lung_occ[:, aj, :]),
        (f"sagittal (H={ai})", sim_grid[ai, :, :], lung_occ[ai, :, :]),
    ]
    for col, (title, s, m) in enumerate(planes):
        im = axes[0, col].imshow(s.T.numpy(), cmap="turbo", vmin=-0.2, vmax=1.0)
        axes[0, col].set_title(f"cos-sim | {title}", fontsize=9)
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        axes[1, col].imshow(m.T.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"lung occupancy | {title}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"SPECTRE-SSL dense tokens, anchor in lung at (h,w,d)=({ai},{aj},{ak})  |  SRSS={srss:+.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "S3_cossim_map_ssl.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return {
        "n_lung_tokens": int(lung.sum()),
        "n_other_tokens": int(other.sum()),
        "anchor_hwd": [ai, aj, ak],
        "srss_lung_vs_other": srss,
        "cos_lung_mean": float(sim[lung].mean()),
        "cos_other_mean": float(sim[other].mean()),
        "figure": str(out.relative_to(Path("/workspace"))),
    }


def main() -> None:
    spectre = import_spectre()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {
        "scan_id": SCAN_ID,
        "device": device,
        "spectre_version": spectre.__version__,
    }

    print("[S0] index roundtrip (model-free) ...")
    report["S0_index_roundtrip"] = s0_index_roundtrip(spectre.window_scan)
    print("    ", report["S0_index_roundtrip"])
    assert report["S0_index_roundtrip"]["roundtrip_exact"], "재조립 순서가 틀렸다"

    print("[load] CT → wan grid (z end-pad 253→256) ...")
    t0 = time.perf_counter()
    vol = load_volume(SCAN_ID)  # (1, 512, 512, 256), raw HU
    t_load = time.perf_counter() - t0
    print(
        f"     {tuple(vol.shape)} HU[{float(vol.min()):.0f}, {float(vol.max()):.0f}] in {t_load:.1f}s"
    )

    print("[build] SPECTRE-SSL backbone (dense, no combiner) ...")
    t0 = time.perf_counter()
    backbone = build_backbone(CKPT_SSL, device=device, with_combiner=False)
    crops, grid = backbone.window(vol)
    t_build = time.perf_counter() - t0
    n_params = sum(p.numel() for p in backbone.model.backbone.parameters())

    print("[S1] window ...")
    kept = spectre.windowing.largest_multiple_crop_size(
        tuple(vol.shape[1:]), SpectreBackbone.crop_size
    )
    report["S1_window"] = {
        "shape_padded": list(vol.shape),
        "depth_pad": DEPTH_PAD,
        "kept_after_center_crop": list(kept),
        "dropped_voxels": int(np.prod(vol.shape[1:]) - np.prod(kept)),
        "crop_grid": list(grid),
        "crops_shape": list(crops.shape),
        "crops_min": float(crops.min()),
        "crops_max": float(crops.max()),
        "pad_value_hu": AIR_HU,
    }
    print("    ", report["S1_window"])
    assert report["S1_window"]["dropped_voxels"] == 0, "center-crop으로 voxel이 잘렸다"
    assert tuple(grid) == (4, 4, 4), f"crop grid {grid} != (4,4,4)"

    print("[S2/S3] backbone forward → CLS drop → 32³ 재조립 ...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    dense, _ = backbone.encode_crops(crops, grid)  # (32, 32, 32, 1080)
    if device == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    p_tok = int(
        np.prod([c // q for c, q in zip(SpectreBackbone.crop_size, PATCH_SIZE)])
    )
    report["S2_tokens"] = {
        "dense_shape": list(dense.shape),
        "num_prefix_tokens": int(backbone.model.backbone.num_prefix_tokens),
        "patch_tokens_per_crop": p_tok,
        "embed_dim": int(dense.shape[-1]),
        "backbone_params_M": round(n_params / 1e6, 1),
    }
    print("    ", report["S2_tokens"])
    assert backbone.model.backbone.num_prefix_tokens == 1, "prefix token이 1개가 아니다"
    assert tuple(dense.shape) == (*TOKEN_GRID, EMBED_DIM), tuple(dense.shape)

    report["S3_anatomy"] = s3_anatomy(dense.float().cpu(), lung_occupancy(SCAN_ID))
    print("    ", report["S3_anatomy"])

    print("[S4] forward_intermediates ...")
    inter = [
        backbone.encode_crops(crops[:4], (1, 1, 4), layer=layer)[0]
        for layer in (11, 17, 23)
    ]
    report["S4_intermediates"] = {
        "layers": [11, 17, 23],
        "shapes": [list(t.shape) for t in inter],
        "depth": len(backbone.model.backbone.blocks),
    }
    print("    ", report["S4_intermediates"])

    print("[S5] combiner scan-level global ...")
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()
    backbone_g = build_backbone(CKPT_SSL, device=device, with_combiner=True)
    _, glob = backbone_g.encode_crops(crops, grid, want_global=True)
    report["S5_global"] = {"global_shape": list(glob.shape)}
    print("    ", report["S5_global"])
    del backbone_g
    if device == "cuda":
        torch.cuda.empty_cache()

    report["S6_cost"] = {
        "load_resample_s": round(t_load, 2),
        "backbone_build_and_window_s": round(t_build, 2),
        "backbone_forward_s": round(t_fwd, 2),
        "peak_gpu_GB": round(peak_gb, 2),
        "dense_32_MB_fp16": round(float(np.prod(TOKEN_GRID)) * EMBED_DIM * 2 / 1e6, 1),
        "dense_16_MB_fp16": round(
            float(np.prod(TOKEN_GRID)) * EMBED_DIM * 2 / 8 / 1e6, 1
        ),
    }
    print("    ", report["S6_cost"])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "smoke_spectre.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] {RESULT_DIR / 'smoke_spectre.json'}")


if __name__ == "__main__":
    main()
