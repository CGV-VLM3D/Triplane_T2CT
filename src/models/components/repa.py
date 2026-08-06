"""Representation-alignment head for the report2ct U-Net (U-REPA / iREPA / VideoREPA).

Aligns a frozen CT foundation model's dense token grid (SPECTRE, `src/baselines/spectre_adapter.py`)
with a middle-stage feature of the diffusion U-Net. The whole thing is opt-in: when the training
module is built without a `RepaAligner` the loss path is bit-identical to the baseline.

The objective follows U-REPA Eq.5 with the outer coefficient λ applied by the caller::

    L = L_velocity + λ · ( hard_weight · L_cos + rel_weight · L_rel )

`L_cos` is REPA's token-wise cosine (Eq.1); `L_rel` is a *relational* term that only constrains the
pairwise similarity structure. Which of the two carries the work depends on the regime, and the
weights make that an experiment rather than an assumption:

* **from scratch** — U-REPA Table 15: λ=0.5 → FID 5.72, λ=0.25 → 6.42, **λ=0 (no `L_cos`) → 10.91**.
  The hard term is load-bearing when the network has no representation yet.
* **finetuning a trained model** — VideoREPA §4.5: hard cosine on a pretrained video DiT destroys
  its established features; their TRD loss is relational-only (`hard_weight=0`).

Three design choices here are measurements, not defaults (see `tests/repa_probe/`):

* `spatial_norm=True` on the **teacher only** (iREPA Algorithm 1). U3 measured it winning on all
  four spatial-structure metrics in 12/12 arms; it drops the teacher's mean off-diagonal cosine
  from 0.24 to 0.05 by removing the global component.
* `projector="conv"` (iREPA) vs `"mlp"` (U-REPA), and projection **before** upsampling either way
  — U-REPA Table 7: MLP-then-upscale 5.72 vs upscale-then-MLP 5.84 vs learned upscale-in-MLP 6.36.
* `rel_scope="within_crop"` exists because SPECTRE attends only inside a 128×128×64 crop, so the
  reassembled grid has a seam: U3 measured neighbour cosine dropping 22.9 % across a crop boundary
  at the final layer, and 63/64 of random token pairs are cross-crop to begin with. The token-wise
  cosine term is immune (each token matches its own target); only the relational term is polluted.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

REL_LOSSES: Final[tuple[str, ...]] = ("urepa_l2", "videorepa_l1_st", "cos_ce")
PROJECTORS: Final[tuple[str, ...]] = ("conv", "mlp")
UPSAMPLES: Final[tuple[str, ...]] = ("trilinear", "deconv")
REL_SCOPES: Final[tuple[str, ...]] = ("global", "within_crop")


def spatial_norm(
    tokens: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """iREPA spatial normalization — remove the global component across the token axis.

    ``y = (x - γ·E[x]) / sqrt(Var[x] + ε)`` with mean/variance over the spatial (token) axis.
    Applied to the target representation only.

    Args:
        tokens: ``(B, T, D)``.

    Returns:
        ``(B, T, D)``.
    """
    mean = tokens.mean(dim=-2, keepdim=True)
    var = tokens.var(dim=-2, unbiased=False, keepdim=True)
    return (tokens - gamma * mean) / torch.sqrt(var + eps)


def st_split(
    diff: torch.Tensor, slab: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a pairwise error matrix into VideoREPA's spatial and temporal components.

    VideoREPA Eq.5 sums two averages: one over token pairs inside the same latent frame
    (spatial relations) and one over pairs in different frames (temporal relations). Our
    "frames" are the Wan latent's z-slabs, so this becomes an in-plane / out-of-plane split —
    which is interesting here because out-of-plane FID is this model family's known weak axis.
    The diagonal is excluded: a token's similarity to itself is 1 by construction.

    Args:
        diff: ``(B, M, M)`` non-negative pairwise error.
        slab: ``(M,)`` slab (z) index of each sampled token.

    Returns:
        ``(spatial, temporal)`` scalars. `temporal` is 0 when every token is in one slab.
    """
    same = (slab.unsqueeze(1) == slab.unsqueeze(0)).unsqueeze(0).expand_as(diff)
    eye = torch.eye(len(slab), dtype=torch.bool, device=slab.device).expand_as(diff)
    spatial = diff[same & ~eye].mean() if (same & ~eye).any() else diff.new_zeros(())
    temporal = diff[~same].mean() if (~same).any() else diff.new_zeros(())
    return spatial, temporal


def _build_projector(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    """Channel projection applied at the student's native (low) resolution."""
    if kind == "conv":
        # iREPA Algorithm 1, 3D: nn.Conv2d(D_in, D_out, kernel_size=3, padding=1)
        return nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
    if kind == "mlp":
        # U-REPA build_mlp: Linear → SiLU → Linear → SiLU → Linear, as 1×1×1 convs
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 1),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 1),
        )
    raise ValueError(f"projector must be one of {PROJECTORS}, got {kind!r}")


class RepaAligner(nn.Module):
    """Projects a U-Net feature onto the teacher grid and scores the alignment.

    Args:
        student_channels: Channels of the tapped U-Net feature (512 for `middle_block`).
        teacher_dim: Teacher token dim (1080 for SPECTRE ViT-L).
        teacher_grid: Token grid the teacher features live on, ``(H, W, D)`` — 32³ native or
            16³ pooled. The student is upsampled to this, never the other way round (U-REPA
            Table 6: downscaling the ViT loses fine-grained information, 5.99 vs 5.72).
        projector: `"conv"` (iREPA, k=3) or `"mlp"` (U-REPA, 3-layer).
        upsample: `"trilinear"` (default) or `"deconv"`. U-REPA already measured a learned
            upsampler as the worst of three placements (6.36 vs 5.72), so `"deconv"` is kept
            only as an ablation knob.
        spatial_norm_teacher: Apply iREPA spatial norm to the teacher tokens.
        gamma: Spatial-norm mean coefficient; 0 keeps the global component.
        hard_weight: Weight on the token-wise cosine term. 0 gives a VideoREPA-style
            relational-only objective.
        rel_weight: Weight on the relational term (U-REPA's `w`; their optimum is 3).
        rel_loss: `"urepa_l2"` (Frobenius² of the Gram difference, U-REPA Eq.3-4),
            `"videorepa_l1_st"` (L1 split into in-slab / cross-slab, VideoREPA Eq.5), or
            `"cos_ce"` (row-softmax cross-entropy — the relational stand-in for HASTE's ATTA).
        rel_tokens: Tokens sampled per step for the Gram; the full 32768² matrix is not
            computable. `None` uses every token.
        rel_scope: `"global"` samples anywhere; `"within_crop"` samples inside one random
            teacher crop so no pair straddles a crop boundary.
        crop_tokens: Teacher tokens per crop along each axis (8³ at the native 32³ grid).
        global_weight: Weight on an optional cosine between the pooled student feature and the
            teacher's scan-level embedding. Off by default — it injects semantics without any
            spatial structure, which is the opposite of what iREPA says matters.
        teacher_global_dim: Dim of that scan-level embedding.
    """

    def __init__(
        self,
        student_channels: int = 512,
        teacher_dim: int = 1080,
        teacher_grid: tuple[int, int, int] = (16, 16, 16),
        projector: str = "conv",
        upsample: str = "trilinear",
        student_grid: tuple[int, int, int] = (8, 8, 8),
        spatial_norm_teacher: bool = True,
        gamma: float = 1.0,
        hard_weight: float = 1.0,
        rel_weight: float = 3.0,
        rel_loss: str = "urepa_l2",
        rel_tokens: int | None = 1024,
        rel_scope: str = "global",
        crop_tokens: tuple[int, int, int] = (8, 8, 8),
        global_weight: float = 0.0,
        teacher_global_dim: int = 1080,
    ) -> None:
        super().__init__()
        if rel_loss not in REL_LOSSES:
            raise ValueError(f"rel_loss must be one of {REL_LOSSES}, got {rel_loss!r}")
        if upsample not in UPSAMPLES:
            raise ValueError(f"upsample must be one of {UPSAMPLES}, got {upsample!r}")
        if rel_scope not in REL_SCOPES:
            raise ValueError(
                f"rel_scope must be one of {REL_SCOPES}, got {rel_scope!r}"
            )
        if hard_weight == 0.0 and rel_weight == 0.0 and global_weight == 0.0:
            raise ValueError("all three weights are 0 — the aligner would be a no-op")

        self.teacher_grid = tuple(teacher_grid)
        self.student_grid = tuple(student_grid)
        self.crop_tokens = tuple(crop_tokens)
        self.spatial_norm_teacher = spatial_norm_teacher
        self.gamma = gamma
        self.hard_weight = hard_weight
        self.rel_weight = rel_weight
        self.rel_loss = rel_loss
        self.rel_tokens = rel_tokens
        self.rel_scope = rel_scope
        self.global_weight = global_weight

        self.projector = _build_projector(projector, student_channels, teacher_dim)
        self.up: nn.Module | None = None
        if upsample == "deconv":
            factors = [t // s for t, s in zip(self.teacher_grid, self.student_grid)]
            if any(
                f * s != t
                for f, s, t in zip(factors, self.student_grid, self.teacher_grid)
            ):
                raise ValueError(
                    f"deconv needs an integer upsample factor, got {self.student_grid} → "
                    f"{self.teacher_grid}"
                )
            self.up = nn.ConvTranspose3d(
                teacher_dim,
                teacher_dim,
                kernel_size=factors,
                stride=factors,
                groups=teacher_dim,
            )
        self.global_head = (
            nn.Linear(teacher_dim, teacher_global_dim) if global_weight else None
        )

    # --- projection ----------------------------------------------------------
    def project(self, student_feat: torch.Tensor) -> torch.Tensor:
        """Project then upsample — U-REPA's "MLP first and then upscale".

        Args:
            student_feat: ``(B, C, h, w, d)`` tapped U-Net feature.

        Returns:
            ``(B, T, teacher_dim)`` with ``T = prod(teacher_grid)``.
        """
        z = self.projector(student_feat)  # (B, teacher_dim, h, w, d)
        if tuple(z.shape[2:]) != self.teacher_grid:
            z = (
                self.up(z)
                if self.up is not None
                else F.interpolate(
                    z, size=self.teacher_grid, mode="trilinear", align_corners=False
                )
            )  # (B, teacher_dim, H, W, D)
        return z.flatten(2).transpose(1, 2)  # (B, T, teacher_dim)

    # --- token sampling ------------------------------------------------------
    def _sample_tokens(
        self, device: torch.device, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Indices of the tokens the relational term is computed on this step.

        `within_crop` draws one random crop and takes all of its tokens, so every sampled pair
        shares the teacher's attention context (see the module docstring on crop seams).
        """
        grid = self.teacher_grid
        n_tokens = grid[0] * grid[1] * grid[2]
        if self.rel_scope == "within_crop":
            n_crops = [g // c for g, c in zip(grid, self.crop_tokens)]
            pick = [
                int(torch.randint(n, (1,), generator=generator, device=device))
                for n in n_crops
            ]
            axes = [
                torch.arange(p * c, (p + 1) * c, device=device)
                for p, c in zip(pick, self.crop_tokens)
            ]
            ii, jj, kk = torch.meshgrid(*axes, indexing="ij")
            idx = ((ii * grid[1] + jj) * grid[2] + kk).reshape(-1)
        elif self.rel_loss == "videorepa_l1_st":
            # 슬랩 구조를 살려야 하므로 (h, w) 위치를 뽑고 z는 통째로 가져간다.
            n_hw = grid[0] * grid[1]
            keep = max(2, (self.rel_tokens or n_tokens) // grid[2])
            hw = torch.randperm(n_hw, generator=generator, device=device)[:keep]
            idx = (
                hw.unsqueeze(1) * grid[2] + torch.arange(grid[2], device=device)
            ).reshape(-1)
            return idx
        else:
            idx = torch.arange(n_tokens, device=device)
        if self.rel_tokens is not None and len(idx) > self.rel_tokens:
            idx = idx[
                torch.randperm(len(idx), generator=generator, device=device)[
                    : self.rel_tokens
                ]
            ]
        return idx

    # --- losses --------------------------------------------------------------
    def _relational(
        self, student: torch.Tensor, teacher: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Relational term on the sampled tokens; all variants are 0 for identical inputs."""
        s = F.normalize(student[:, idx].float(), dim=-1)
        t = F.normalize(teacher[:, idx].float(), dim=-1)
        gram_s = s @ s.transpose(1, 2)  # (B, M, M)
        gram_t = t @ t.transpose(1, 2)

        if self.rel_loss == "urepa_l2":
            # U-REPA Eq.4 — ‖sim(y*) − sim(h(h_t))‖²_F, mean-reduced
            return (gram_s - gram_t).pow(2).mean()
        if self.rel_loss == "cos_ce":
            # row-softmax cross-entropy: ATTA의 분포 매칭 성격을 attention 배관 없이 흉내낸다
            log_p = F.log_softmax(gram_s, dim=-1)
            q = F.softmax(gram_t, dim=-1)
            return -(q * log_p).sum(-1).mean()
        # videorepa_l1_st — VideoREPA Eq.5: L1, in-slab(spatial) / cross-slab(temporal) 분리 합산
        spatial, temporal = st_split(
            (gram_s - gram_t).abs(), idx % self.teacher_grid[2]
        )
        # 두 항을 따로 로깅할 수 있게 남긴다. 합만 보면 이 손실의 요점(두 관계 유형에 **동등**
        # 가중치를 준다)이 실제로 작동하는지 알 수 없다 — 무작위 표본에서 cross-slab 쌍이
        # 93.8 %라 균일 평균이면 z 관계가 사실상 전부를 차지한다(측정치, 16³ / 1024 토큰).
        # temporal 이 0이면 표본이 한 slab에 몰린 것이므로 즉시 드러나야 한다.
        self._last_st = (spatial.detach(), temporal.detach())
        return spatial + temporal

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_tokens: torch.Tensor,
        teacher_global: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score how well the U-Net feature aligns with the teacher grid.

        Args:
            student_feat: ``(B, C, h, w, d)`` — the tapped U-Net feature. **Must still be in the
                autograd graph**: detaching it here silently reduces the aligner to training its
                own projector while the U-Net learns nothing.
            teacher_tokens: ``(B, T, teacher_dim)`` precomputed frozen features.
            teacher_global: ``(B, teacher_global_dim)``, required iff `global_weight > 0`.
            generator: RNG for token sampling. Pass a dedicated generator so enabling the
                aligner does not shift the global RNG stream (keeps a `repa_weight=0` run
                bit-identical to the baseline).

        Returns:
            ``(loss, logs)`` — the weighted sum, and detached scalars for logging
            (`repa_cos` is the mean cosine, i.e. higher is better, not the loss).
        """
        if (
            teacher_tokens.shape[1]
            != self.teacher_grid[0] * self.teacher_grid[1] * self.teacher_grid[2]
        ):
            raise ValueError(
                f"teacher has {teacher_tokens.shape[1]} tokens but teacher_grid "
                f"{self.teacher_grid} implies "
                f"{self.teacher_grid[0] * self.teacher_grid[1] * self.teacher_grid[2]}"
            )
        teacher = teacher_tokens.float()
        if self.spatial_norm_teacher:
            teacher = spatial_norm(teacher, gamma=self.gamma)

        student = self.project(student_feat)  # (B, T, teacher_dim)
        loss = student.new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        if self.hard_weight:
            student_n = F.normalize(student.float(), dim=-1)
            teacher_n = F.normalize(teacher, dim=-1)
            cos = (student_n * teacher_n).sum(-1).mean()  # REPA Eq.1
            loss = loss + self.hard_weight * (-cos)
            logs["repa_cos"] = cos.detach()
            # Diagnostic, not a loss term: the same cosine against a teacher belonging to a
            # DIFFERENT volume in the batch. Chest CTs are anatomically stereotyped, so a pure
            # position prior already scores high — an 8-volume overfit reaches 0.45 with the
            # teacher reshuffled every step vs 0.52 with the real one
            # (tests/repa_probe/u5_overfit/). The GAP is the volume-specific part and is what
            # should be watched during training; raw `repa_cos` overstates the alignment.
            if teacher_n.shape[0] > 1:
                shuffled = (student_n * torch.roll(teacher_n, 1, dims=0)).sum(-1).mean()
                logs["repa_cos_shuffled"] = shuffled.detach()
                logs["repa_cos_gap"] = (cos - shuffled).detach()

        if self.rel_weight:
            idx = self._sample_tokens(student.device, generator)
            rel = self._relational(student, teacher, idx)
            loss = loss + self.rel_weight * rel
            logs["repa_rel"] = rel.detach()
            logs["repa_rel_tokens"] = torch.tensor(
                float(len(idx)), device=student.device
            )
            st = getattr(self, "_last_st", None)  # videorepa_l1_st 에서만 채워진다
            if st is not None:
                logs["repa_rel_spatial"], logs["repa_rel_temporal"] = st

        if self.global_weight:
            if teacher_global is None:
                raise ValueError("global_weight > 0 requires teacher_global")
            pooled = self.global_head(student.mean(dim=1))  # (B, teacher_global_dim)
            gcos = (
                (
                    F.normalize(pooled.float(), dim=-1)
                    * F.normalize(teacher_global.float(), dim=-1)
                )
                .sum(-1)
                .mean()
            )
            loss = loss + self.global_weight * (-gcos)
            logs["repa_global_cos"] = gcos.detach()

        return loss, logs


__all__ = [
    "PROJECTORS",
    "REL_LOSSES",
    "REL_SCOPES",
    "UPSAMPLES",
    "RepaAligner",
    "spatial_norm",
    "st_split",
]
