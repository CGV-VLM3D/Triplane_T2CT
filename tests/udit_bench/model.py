"""Dummy hierarchical U-DiT backbone for cost/memory benchmarking (Phase 2a).

Self-attention + AdaLN-Zero timestep conditioning only — no text cross-attention.
Cross-attn / MM-DiT cost was already computed analytically with high confidence
(see docs/vlm3d_research_roadmap.md Phase 2c); the part that needs empirical
validation is the 3D windowed-attention hierarchy itself (real activation memory,
real achieved TFLOPS on this hardware).

Isolation principle (tests/repa_probe/README.md convention): this file does not
touch src/ or configs/. Nothing here is wired into training.

Grid convention: tokens are stored as (B, N, C) with N = Dg*Hg*Wg, flattened from
a (B, Dg, Hg, Wg, C) volume in standard row-major order (W fastest). Every module
that needs spatial structure (windowing, merge, expand) takes the grid shape
explicitly rather than inferring it from N.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from monai.networks.nets.swin_unetr import window_partition, window_reverse
from torch import nn

Grid = tuple[int, int, int]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulate: `x * (1 + scale) + shift`, broadcasting over the token dim.

    Args:
        x: `(B, N, C)`.
        shift: `(B, C)`.
        scale: `(B, C)`.

    Returns:
        `(B, N, C)`.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP (DiT / U-REPA convention)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t: `(B,)`. Returns: `(B, hidden_size)`."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Attention3D(nn.Module):
    """Multi-head self-attention over a `(B, N, C)` token sequence via SDPA.

    Optionally windowed: partitions the `(Dg, Hg, Wg)` grid into non-overlapping
    `window^3` blocks (MONAI's `window_partition`/`window_reverse`, the same
    functions `SwinTransformer` uses), attending within each block only. Every
    other windowed block shifts the grid by `window // 2` before partitioning
    (Swin-style shifted window) so information crosses block boundaries.
    """

    def __init__(self, dim: int, num_heads: int, window: int | None = None) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _sdpa(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_windows_or_1, tokens, C)
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def forward(self, x: torch.Tensor, grid: Grid, shift: bool) -> torch.Tensor:
        """Args: x: `(B, N, C)`. grid: `(Dg, Hg, Wg)`. shift: use the shifted-window variant.

        Returns: `(B, N, C)`.
        """
        if self.window is None:
            return self._sdpa(x)

        B, N, C = x.shape
        Dg, Hg, Wg = grid
        w = self.window
        x = x.view(B, Dg, Hg, Wg, C)
        shift_size = w // 2 if shift else 0
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size,) * 3, dims=(1, 2, 3))

        windows = window_partition(x, (w, w, w))  # (B*n_win, w^3, C)
        windows = self._sdpa(windows)
        x = window_reverse(windows, (w, w, w), (B, Dg, Hg, Wg))

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size,) * 3, dims=(1, 2, 3))
        return x.view(B, N, C)


class DiTBlock(nn.Module):
    """Pre-norm self-attn + MLP block with AdaLN-Zero conditioning (DiT / U-REPA)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention3D(dim, num_heads, window=window)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, grid: Grid, shift: bool
    ) -> torch.Tensor:
        """Args: x: `(B, N, C)`. c: AdaLN conditioning vector, `(B, C)`. grid, shift: see `Attention3D`."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), grid, shift
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class PatchMerge3D(nn.Module):
    """Downsample a grid 2x2x2 -> 1: concat 8 neighbors, project to `out_dim`."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim * 8, out_dim)

    def forward(self, x: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, Grid]:
        B, N, C = x.shape
        Dg, Hg, Wg = grid
        x = x.view(B, Dg // 2, 2, Hg // 2, 2, Wg // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            B, (Dg // 2) * (Hg // 2) * (Wg // 2), 8 * C
        )
        return self.proj(x), (Dg // 2, Hg // 2, Wg // 2)


class PatchExpand3D(nn.Module):
    """Upsample a grid 1 -> 2x2x2: project to `8*out_dim`, scatter into the finer grid."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, 8 * out_dim)

    def forward(self, x: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, Grid]:
        B, N, _ = x.shape
        Dg, Hg, Wg = grid
        x = self.proj(x).view(B, Dg, Hg, Wg, 2, 2, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
            B, 2 * Dg, 2 * Hg, 2 * Wg, self.out_dim
        )
        new_grid = (2 * Dg, 2 * Hg, 2 * Wg)
        return x.reshape(
            B, new_grid[0] * new_grid[1] * new_grid[2], self.out_dim
        ), new_grid


class SkipFuse(nn.Module):
    """Concat an upsampled feature with its encoder skip, project back to `dim` (U-REPA `skip_linear`)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x, skip], dim=-1))


class HierarchicalUDiT3D(nn.Module):
    """3-level U-shaped 3D DiT: L0 (windowed) -> L1 -> L2 (bottleneck) -> L1 -> L0.

    Matches the roadmap's Phase 2a/2b sizing: L0 32^3/d384/window=8 (2 enc + 2 dec
    blocks = 4 total), L1 16^3/d768 (4 enc + 4 dec = 8 total), L2 8^3/d1152 bottleneck
    (12 blocks, run once — not split encoder/decoder). Patchify uses a stride-2 conv
    (patch=2), i.e. the model consumes the Wan latent directly at its native (16, 64,
    64, 64) resolution.
    """

    def __init__(
        self,
        in_channels: int = 16,
        dims: tuple[int, int, int] = (384, 768, 1152),
        depths: tuple[int, int, int] = (
            2,
            4,
            12,
        ),  # encoder-side counts; L0/L1 decoder mirrors
        num_heads: tuple[int, int, int] = (6, 12, 18),
        window: int = 8,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        d0, d1, d2 = dims
        self.dims = dims
        self.patchify = nn.Conv3d(in_channels, d0, kernel_size=2, stride=2)
        self.t_embedder = TimestepEmbedder(d0)
        # separate AdaLN conditioning width per level via a linear re-projection
        self.c_proj1 = nn.Linear(d0, d1)
        self.c_proj2 = nn.Linear(d0, d2)

        n0, n1, n2 = depths
        self.enc0 = nn.ModuleList(
            [DiTBlock(d0, num_heads[0], mlp_ratio, window=window) for _ in range(n0)]
        )
        self.down0 = PatchMerge3D(d0, d1)
        self.enc1 = nn.ModuleList(
            [DiTBlock(d1, num_heads[1], mlp_ratio) for _ in range(n1)]
        )
        self.down1 = PatchMerge3D(d1, d2)

        self.bottleneck = nn.ModuleList(
            [DiTBlock(d2, num_heads[2], mlp_ratio) for _ in range(n2)]
        )

        self.up1 = PatchExpand3D(d2, d1)
        self.skip1 = SkipFuse(d1)
        self.dec1 = nn.ModuleList(
            [DiTBlock(d1, num_heads[1], mlp_ratio) for _ in range(n1)]
        )
        self.up0 = PatchExpand3D(d1, d0)
        self.skip0 = SkipFuse(d0)
        self.dec0 = nn.ModuleList(
            [DiTBlock(d0, num_heads[0], mlp_ratio, window=window) for _ in range(n0)]
        )

        self.final_norm = nn.LayerNorm(d0, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(d0, 2 * 2 * 2 * in_channels)
        self.in_channels = in_channels

    def _run_block(
        self, blk: DiTBlock, x: torch.Tensor, c: torch.Tensor, grid: Grid, shift: bool
    ) -> torch.Tensor:
        """Call `blk`, optionally recomputing activations in the backward pass.

        Gated on `self.training` too: checkpointing only trades compute for memory
        when a backward pass will follow, and `torch.utils.checkpoint` requires at
        least one input to require grad.
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                blk, x, c, grid, shift, use_reentrant=False
            )
        return blk(x, c, grid, shift)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Args: x: `(B, in_channels, 64, 64, 64)`. t: `(B,)` diffusion timestep.

        Returns: `(B, in_channels, 64, 64, 64)`, same shape as `x` (velocity/noise pred).
        """
        B = x.shape[0]
        x = self.patchify(x)  # (B, d0, 32, 32, 32)
        grid0 = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N0, d0)

        c0 = self.t_embedder(t)
        c1, c2 = self.c_proj1(c0), self.c_proj2(c0)

        for i, blk in enumerate(self.enc0):
            x = self._run_block(blk, x, c0, grid0, shift=(i % 2 == 1))
        skip0 = x

        x, grid1 = self.down0(x, grid0)
        for blk in self.enc1:
            x = self._run_block(blk, x, c1, grid1, shift=False)
        skip1 = x

        x, grid2 = self.down1(x, grid1)
        for blk in self.bottleneck:
            x = self._run_block(blk, x, c2, grid2, shift=False)

        x, grid1b = self.up1(x, grid2)
        assert grid1b == grid1
        x = self.skip1(x, skip1)
        for blk in self.dec1:
            x = self._run_block(blk, x, c1, grid1, shift=False)

        x, grid0b = self.up0(x, grid1)
        assert grid0b == grid0
        x = self.skip0(x, skip0)
        for i, blk in enumerate(self.dec0):
            x = self._run_block(blk, x, c0, grid0, shift=(i % 2 == 1))

        x = self.final_proj(self.final_norm(x))  # (B, N0, 8*in_channels)
        Dg, Hg, Wg = grid0
        x = x.view(B, Dg, Hg, Wg, 2, 2, 2, self.in_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            B, self.in_channels, 2 * Dg, 2 * Hg, 2 * Wg
        )
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
