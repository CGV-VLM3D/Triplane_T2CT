"""마스크 VAE 재구성 시각화 — GT 라벨맵 vs 재구성 라벨맵 슬라이스 비교.

각 샘플의 3개 axial 슬라이스를 GT(위)/recon(아래)로 나란히 저장한다. 라벨은 discrete colormap
(`nipy_spectral`)로 색칠. seed·샘플 수는 호출부(train config)에서 지정.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.amp import autocast  # noqa: E402


@torch.no_grad()
def save_recon_vis(
    ae: torch.nn.Module,
    prep,
    masks: torch.Tensor,
    out_dir: Path,
    n_samples: int,
    seed: int,
    num_classes: int = 118,
) -> int:
    """마스크 배치에서 n_samples개를 골라 GT vs 재구성 비교 이미지를 저장한다.

    Args:
        ae: 학습된 마스크 VAE.
        prep: 정수 라벨 → VAE 입력 변환 함수.
        masks: ``(N,1,D,H,W)`` 정수 라벨(디바이스 위).
        out_dir: 저장 폴더(`<exp>/vis`).
        n_samples: 시각화할 샘플 수.
        seed: 랜덤 샘플 선택 시드(재현성).
        num_classes: 라벨 클래스 수(컬러맵 범위).

    Returns:
        저장한 이미지 수.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, masks.shape[0])
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(masks.shape[0], generator=g)[:n].tolist()

    ae.eval()
    for i, si in enumerate(sel):
        m = masks[si : si + 1]  # (1,1,D,H,W)
        with autocast("cuda", dtype=torch.bfloat16):
            recon, _, _ = ae(prep(m))
        pred = recon.argmax(1)[0].cpu().numpy()  # (D,H,W)
        gt = m[0, 0].cpu().numpy()  # (D,H,W)

        depth = gt.shape[0]
        zs = [depth // 4, depth // 2, (3 * depth) // 4]
        fig, axes = plt.subplots(2, len(zs), figsize=(4 * len(zs), 8))
        for j, z in enumerate(zs):
            axes[0, j].imshow(gt[z], cmap="nipy_spectral", vmin=0, vmax=num_classes - 1)
            axes[0, j].set_title(f"GT z={z}")
            axes[0, j].axis("off")
            axes[1, j].imshow(
                pred[z], cmap="nipy_spectral", vmin=0, vmax=num_classes - 1
            )
            axes[1, j].set_title(f"recon z={z}")
            axes[1, j].axis("off")
        fig.suptitle(f"sample idx={si}")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:02d}.png", dpi=80)
        plt.close(fig)
    ae.train()
    return n


__all__ = ["save_recon_vis"]
