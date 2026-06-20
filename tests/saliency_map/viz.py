"""Saliency 결과 저장 + 오버레이 시각화.

SaliencyResult → 디스크:
  maps/<scan_id>/<name>.npy            (cam, raw float)
  maps/<scan_id>/<name>_native.npy     (native map, 있으면)
  overlays/<scan_id>/<name>.png        (CT + cam, axial+coronal 패널)
  overlays/<scan_id>/<name>_native.png (CT + native, 있으면)

방향: display 의 (D,H,W) 축 방향코드(orient)로 **canonical 뷰**로 정규화 —
axial(anterior 위, 환자-left 오른쪽) / coronal(머리 위, 환자-left 오른쪽). cam·display 를
함께 변환해 정렬 유지. heat 는 **값-비례 alpha**(낮으면 투명)로 얹어 CT 배경이 비치게 함.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tests.saliency_map.runners.base import SaliencyResult  # noqa: E402

_PLANE = {"S": "SI", "I": "SI", "A": "AP", "P": "AP", "L": "LR", "R": "LR"}
_MAX_ALPHA = 0.8  # heat 최대 불투명도 (값-비례)


def _norm01(a: np.ndarray) -> np.ndarray:
    """[min,max] → [0,1] (상수면 0)."""
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=np.float32)


def _to_canonical(arr: np.ndarray, orient: tuple[str, str, str]) -> np.ndarray:
    """(D,H,W) → canonical (SI, AP, LR), 증가방향 = (Inferior, Posterior, Left).

    그러면 canonical[si] = axial(행=AP: anterior 위, 열=LR: 환자-left 오른쪽),
    canonical[:,ap] = coronal(행=SI: superior 위, 열=LR). orient 은 각 축의 *증가* 방향코드.
    전처리에 부호반전이 없어 단순 transpose + 필요한 축 뒤집기로 충분.
    """
    order = [
        next(i for i, c in enumerate(orient) if _PLANE[c] == p)
        for p in ("SI", "AP", "LR")
    ]
    c = np.transpose(arr, order)  # (SI, AP, LR)
    codes = [orient[a] for a in order]
    if codes[0] == "S":  # 증가=Superior → 목표 Inferior 증가로 뒤집기
        c = c[::-1, :, :]
    if codes[1] == "A":  # 증가=Anterior → 목표 Posterior
        c = c[:, ::-1, :]
    if codes[2] == "R":  # 증가=Right → 목표 Left
        c = c[:, :, ::-1]
    return c


def _panel(ax, ct2d: np.ndarray, heat2d: np.ndarray, title: str):
    """CT(gray) + heat(jet, 값-비례 alpha) 한 패널."""
    ax.imshow(_norm01(ct2d), cmap="gray", aspect="auto")
    alpha = np.clip(heat2d, 0, 1) * _MAX_ALPHA  # 낮은 heat = 투명 → CT 비침
    ax.imshow(heat2d, cmap="jet", vmin=0, vmax=1, alpha=alpha, aspect="auto")
    ax.set_title(title)
    ax.axis("off")


def _overlay_panels(
    display: np.ndarray,
    heat: np.ndarray,
    orient: tuple[str, str, str],
    title: str,
    png_path: Path,
):
    """canonical axial/coronal 최대-saliency 슬라이스에 CT+heat 오버레이 저장.

    display, heat: (D, H, W) 동일 그리드. heat 는 [0,1] 가정.
    """
    dc = _to_canonical(display, orient)  # (SI, AP, LR)
    hc = _to_canonical(heat, orient)
    si = int(hc.sum(axis=(1, 2)).argmax())  # axial: SI 축 최대 saliency
    ap = int(hc.sum(axis=(0, 2)).argmax())  # coronal: AP 축 최대 saliency

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _panel(axes[0], dc[si], hc[si], f"axial (sup z={si})")  # (AP, LR)
    _panel(axes[1], dc[:, ap], hc[:, ap], f"coronal (ant y={ap})")  # (SI, LR)
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
        result.orient,
        f"{scan_id} · {result.item_name} (cam) · {arm}",
        out_dir / "overlays" / scan_id / f"{result.item_name}.png",
    )

    if result.native_map is not None:
        np.save(maps_dir / f"{result.item_name}_native.npy", result.native_map)
        _overlay_panels(
            result.display,
            _norm01(result.native_map),
            result.orient,
            f"{scan_id} · {result.item_name} (native) · {arm}",
            out_dir / "overlays" / scan_id / f"{result.item_name}_native.png",
        )
