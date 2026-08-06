"""MaskConditioner — LDM-style concat conditioner for 3D semantic (organ) masks.

Turns a **soft, area-downsampled** organ label map (already reduced to the diffusion latent grid)
into a small channel-embedding that is concatenated onto the noisy latent before the UNet. This
is the 3D, memory-feasible realization of LDM's ``SpatialRescaler`` concat conditioning
(https://github.com/CompVis/latent-diffusion, ``models/ldm/semantic_synthesis512``), whose
``forward`` is ``interpolate`` (spatial downsample of a one-hot seg map) → ``channel_mapper`` (a
``1x1`` conv, ``bias=False``).

Why *soft* (area) and not nearest: downsampling a categorical map ~4x per axis with nearest
point-samples one label per output cell, so thin/small structures vanish. The correct downsample
is the **partial-volume fraction** — the area each class covers in the cell. LDM approximates this
with ``bilinear`` on the one-hot map; we use exact block area-fractions (the principled version).

Memory: a full ``num_classes``-channel one-hot at 3D resolution is infeasible
(``118*128*128*32*4B ≈ 0.5 GB`` even at the latent grid; ``>26 GB`` at full res). So the precompute
keeps, per latent voxel, only the **top-K** ``(class, area_fraction)`` pairs (``fracs`` renormalized
to sum to 1). The conditioner then computes ``e_mask = Σ_k frac_k · embedding(class_k)``, which is
**exactly** ``channel_mapper`` applied to the soft one-hot (an ``nn.Embedding`` row is a
``channel_mapper`` column), truncated to top-K — and reduces to a plain per-voxel lookup at ``K=1``.

Init properties (empirically validated — plan A4 / Unit-0):
  * **Zero-init gamma** (per-channel LayerScale) ⇒ ``e_mask == 0`` at start ⇒ bit-identical to the
    mask-free baseline; ``gamma`` learns promptly (one-step warmup), doubling as a "mask strength" knob.
  * **Unit-scale content** (``nn.Embedding`` default init ``N(0,1)``): a single-class voxel has
    content std ≈ 1 (matches the ``scale_factor``-normalized latent, std ≈ 1); soft mixing of classes
    only lowers it toward the convex-combination range (still O(1)), and ``gamma`` adapts the scale.

Latent-shape agnostic: the per-voxel weighted lookup has no spatial-size assumption — same module for
the text2ct grid ``(128,128,32)`` or the MAISI grid ``(120,120,64)``.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskConditioner(nn.Module):
    """Top-K soft organ-label embedding concatenated to the latent (LDM concat conditioning).

    Args:
        num_classes: number of segmentation labels including background (TotalSegmentator v2 = 118).
        embed_dim: mask embedding channels appended to the latent; the UNet's ``in_channels`` must
            be ``latent_channels + embed_dim`` (e.g. ``4 + 4 = 8``).
        class_dropout: per-sample, per-organ-class probability of dropping that class to background
            during training (partial-mask augmentation, since a downstream text→mask model emits
            imperfect masks and ts_seg class sets vary per scan). Background (0) is never dropped.
            NOTE: there is no whole-mask / CFG dropout — classifier-free guidance is applied to the
            TEXT only (text2ct convention); the mask is always present, including at inference.
    """

    def __init__(
        self,
        num_classes: int = 118,
        embed_dim: int = 4,
        class_dropout: float = 0.1,
        # mask_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_dropout = float(class_dropout)
        # self.mask_dropout = float(mask_dropout)
        # Rows = per-class channel vectors (== channel_mapper columns). Default init N(0,1)
        # -> single-class content std ~1 (matches scale_factor-normalized z).
        self.embedding = nn.Embedding(num_classes, embed_dim)
        # LayerScale, zero-initialized -> e_mask == 0 at start (== mask-free baseline).
        self.gamma = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, classes: torch.Tensor, fracs: torch.Tensor) -> torch.Tensor:
        """Embed a top-K soft label map into concat channels.

        Args:
            classes: top-K class ids per latent voxel, ``(B, K, H, W, D)`` int. Padding slots
                (cells with < K classes) use class 0 with ``fracs`` 0 (contribute nothing).
            fracs: matching area fractions, ``(B, K, H, W, D)`` float, summing to 1 over ``K``.

        Returns:
            Mask embedding ``(B, embed_dim, H, W, D)``. All-zeros at init (``gamma=0``).
        """
        classes = classes.long()

        if self.training and self.class_dropout > 0.0:
            # Per-organ dropout -> the class becomes background (0), so its area fraction now
            # contributes the background embedding (i.e. "this organ was not segmented").
            b = classes.shape[0]
            drop_tbl = (
                torch.rand(b, self.num_classes, device=classes.device)
                < self.class_dropout
            )  # (b, c=118) 랜덤
            # drop_tbl[i, c] == True 면 샘플 i에서 클래스 c를 drop하겠다는 의미
            drop_tbl[:, 0] = False  # never drop background, 배경은 drop 안함
            voxel_drop = drop_tbl.gather(
                1, classes.reshape(b, -1)
            ).reshape(
                classes.shape
            )  # tbl을 voxel map으로 변환, classes.shape = (B, K, H, W, D), classes.reshape(b, -1) = (B, K*H*W*D)
            # drop_tbl.gather(1, (B, K*H*W*D)) -> out[i, j] = drop_tbl[i, index[i, j]]
            classes = classes.masked_fill(voxel_drop, 0)

        emb = self.embedding(classes)  # (B, K, H, W, D, embed_dim)
        e = (emb * fracs.unsqueeze(-1)).sum(
            dim=1
        )  # (B, H, W, D, embed_dim) — soft weighted sum
        e = e * self.gamma  # per-channel LayerScale (broadcast over last dim)
        # 총 emded_dim 개수만큼의 gamma가 존재함, 하나의 gamma는 B, H, W, D 공통으로
        e = e.permute(0, 4, 1, 2, 3).contiguous()  # (B, embed_dim, H, W, D)

        # if self.training and self.mask_dropout > 0.0:
        #     # Whole-mask dropout -> null (zeros); matches the CFG null convention (null_like).
        #     keep = torch.rand(e.shape[0], device=e.device) >= self.mask_dropout
        #     e = e * keep.view(-1, 1, 1, 1, 1)  # (B, embed_dim, H, W, D)
        return e

    # @staticmethod
    # def null_like(e_mask: torch.Tensor) -> torch.Tensor:
    #     """Return the classifier-free-guidance null conditioning (all zeros).

    #     Args:
    #         e_mask: a reference mask embedding ``(B, embed_dim, H, W, D)``.

    #     Returns:
    #         Zeros of the same shape/dtype/device — the mask-unconditional branch, matching the
    #         all-zeros convention used by ``mask_dropout`` at train time.
    #     """
    #     return torch.zeros_like(e_mask)


__all__ = ["MaskConditioner"]
