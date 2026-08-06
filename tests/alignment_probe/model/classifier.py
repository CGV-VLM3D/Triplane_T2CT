"""LatentClassifier — MAISI latent (4,120,120,64) → 18-label 작은 3D CNN + latent Dataset.

penultimate(256-d) = Probe B의 semantic image feature ``img_feat``. 5개 text 인코더와 독립
(random init, image+label만 사용) → Probe B 순환논증 방지.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from tests.alignment_probe.utils.cases import ABNORMALITY_LABELS, AlignmentCase
from tests.alignment_probe.utils.latents import has_latent, load_latent

_FEAT_DIM = 256


class LatentClassifier(nn.Module):
    """MAISI latent (4,120,120,64) → 18-label. penultimate = 256-d pre-ReLU signed feature.

    img_feat(=``features()``)는 마지막 ReLU **이전** pooled feature라 음수 성분이 살아있다. ReLU를
    통과하면 모든 차원 ≥ 0 → 벡터가 양수 사분면에 갇혀 cosine ∈ [0, 1]로 압축된다. pre-ReLU는
    cosine ∈ [-1, 1] 전 범위를 사용해 변별력을 회복한다.
    """

    def __init__(self, in_ch: int = 4, feat_dim: int = _FEAT_DIM, n_labels: int = 18):
        """3D CNN body, pool, head를 초기화한다."""
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            """Conv3d(3×3×3, padding=1) + BN + ReLU + MaxPool3d(2) 블록."""
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            )

        self.body = nn.Sequential(
            block(in_ch, 32),  # (32, 60, 60, 32)
            block(32, 64),  # (64, 30, 30, 16)
            block(64, 128),  # (128, 15, 15, 8)
            nn.Conv3d(128, feat_dim, 3, padding=1),
            nn.BatchNorm3d(feat_dim),  # 마지막 ReLU 제거
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(feat_dim, n_labels) 

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, 120, 120, 64) → (B, feat_dim) pre-ReLU signed penultimate."""
        return self.pool(self.body(x)).flatten(
            1
        )  # (B, feat_dim, 1,1,1) → (B, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, 120, 120, 64) → (B, 18) logits."""
        return self.head(self.act(self.features(x)))


class LatentDataset(Dataset):
    """latent이 디스크에 있는 케이스만. __getitem__ → (latent (4,120,120,64), labels (18,), scan_id)."""

    def __init__(self, cases: list[AlignmentCase], split: str):
        """디스크에 latent가 존재하는 케이스만 필터링해 Dataset을 초기화한다."""
        self.split = split
        self.items = [c for c in cases if has_latent(c.scan_id, split)]

    def __len__(self) -> int:
        """유효 케이스 수를 반환한다."""
        return len(self.items)

    def __getitem__(self, i: int):
        """인덱스 i에 해당하는 (latent, labels, scan_id) 튜플을 반환한다.

        Returns:
            latent: MAISI VAE latent 텐서. ``(4, 120, 120, 64)``
            labels: 이진 abnormality 레이블. ``(18,)``
            scan_id: 해당 scan의 식별자 문자열.
        """
        c = self.items[i]
        y = torch.tensor([c.labels[a] for a in ABNORMALITY_LABELS], dtype=torch.float32)
        return load_latent(c.scan_id, self.split), y, c.scan_id
