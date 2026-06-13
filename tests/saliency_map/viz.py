"""Saliency 결과 저장 + 오버레이 시각화.

SaliencyResult → 디스크:
  maps/<scan_id>/<name>.npy            (cam, raw float)
  maps/<scan_id>/<name>_native.npy     (native map, 있으면)
  overlays/<scan_id>/<name>.png        (CT + cam, axial+coronal 패널)
  overlays/<scan_id>/<name>_native.png (CT + native, 있으면)

cam/native 는 [0,1] 정규화 후 jet 으로 CT(gray) 위에 alpha 합성. 최대-saliency
슬라이스를 축별로 골라 그린다 (axial=D축, coronal=H축).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tests.saliency_map.runners.base import SaliencyResult  # noqa: E402


def _norm01(a: np.ndarray) -> np.ndarray:
    """[min,max] → [0,1] (상수면 0)."""
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=np.float32)


def _overlay_panels(display: np.ndarray, heat: np.ndarray, title: str, png_path: Path):
    """CT(display) + heat 를 최대-saliency axial/coronal 슬라이스로 그려 저장.

    display, heat: (D, H, W) 동일 그리드. heat 는 [0,1] 가정.
    """
    zi = int(heat.sum(axis=(1, 2)).argmax())  # axial: D 축 최대 saliency 슬라이스
    yi = int(heat.sum(axis=(0, 2)).argmax())  # coronal: H 축 최대 saliency 슬라이스
    ct = _norm01(display)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axial: (H, W) — D 고정
    axes[0].imshow(ct[zi], cmap="gray")
    axes[0].imshow(heat[zi], cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[0].set_title(f"axial z={zi}")
    # coronal: (D, W) — H 고정
    axes[1].imshow(ct[:, yi], cmap="gray", aspect="auto")
    axes[1].imshow(heat[:, yi], cmap="jet", alpha=0.4, vmin=0, vmax=1, aspect="auto")
    axes[1].set_title(f"coronal y={yi}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_result(result: SaliencyResult, scan_id: str, out_dir: Path) -> None:
    """SaliencyResult → npy(cam[, native]) + overlay png(s)."""
    maps_dir = out_dir / "maps" / scan_id
    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / f"{result.item_name}.npy", result.cam)  # (D, H, W)

    arm = f"{result.organ or 'global'} | prob={result.pred_prob:.2f} score={result.score:+.3f}"
    _overlay_panels(
        result.display,
        _norm01(result.cam),
        f"{scan_id} · {result.item_name} (cam) · {arm}",
        out_dir / "overlays" / scan_id / f"{result.item_name}.png",
    )

    if result.native_map is not None:
        np.save(maps_dir / f"{result.item_name}_native.npy", result.native_map)
        _overlay_panels(
            result.display,
            _norm01(result.native_map),
            f"{scan_id} · {result.item_name} (native) · {arm}",
            out_dir / "overlays" / scan_id / f"{result.item_name}_native.png",
        )
