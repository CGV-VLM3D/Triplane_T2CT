"""DiffusionModelUNetMaisiTextPooled — pooled-text conditioning via the UNet's `emb` path.

Report2CT's `DiffusionModelUNetMaisi` already threads an optional embedding vector, `emb`,
into every ResBlock (`h = h + time_emb_proj(emb)[...,None,None,None]` — a per-channel additive
bias, broadcast spatially; NOT a full AdaLN scale+shift+gate). `emb` starts as the timestep +
class embedding, then upstream's own `_get_input_embeddings` optionally *concatenates* on a
top/bottom-region-index and/or spacing embedding, each `time_embed_dim`-wide
(`third_party` note: this file wraps the pip-installed MONAI package, not a vendored tree, but
the same "reuse, don't rewrite" duplication-citation convention applies —
`monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py`).

⚠ Why this class ADDS the pooled-text projection instead of concatenating like the existing
optional signals do: every ResBlock's own `time_emb_proj = nn.Linear(temb_channels, out_channels)`
is sized at `__init__` time from `new_time_embed_dim` — `time_embed_dim` plus one more
`time_embed_dim` per enabled optional flag (verified by reading
`DiffusionModelUNetMaisi.__init__` directly). Concatenating one more segment onto `emb` at
*forward* time would grow it past the width every ResBlock's `time_emb_proj` was already built
for — a hard shape-mismatch crash, not a subtle bug. Projecting the pooled text vector to that
SAME final width and adding it (mirroring how each ResBlock itself injects `emb` into `h`) sidesteps
this entirely: `emb`'s width never changes, so nothing downstream needs to know this class exists.

⚠ Why the projection is `time_embed_dim`-wide (the leading segment) rather than `emb`'s full
width: after `_get_input_embeddings`, `emb` is `[timestep+class | spacing | region...]` — the
first segment is a genuine *sum* (upstream does `emb += class_emb`, exactly DiT's `c = t + y`),
while the optional geometric signals are *concatenated* into their own reserved slots. Writing
across the whole width would mean no downstream layer can ever read a clean spacing signal
(every column carrying spacing would also carry text). That matters here for two reasons:
spacing/FOV measurably moves this project's headline metric, and the intended use is finetuning
from a checkpoint that already learned to read those columns as pure spacing. Zero-padding the
projection's output keeps the geometric segments byte-identical and puts text where DiT/SD3 put
their non-timestep conditioning — added into the timestep space
(`facebookresearch/DiT` `c = t + y`; diffusers `CombinedTimestepTextProjEmbeddings`,
`conditioning = timesteps_emb + pooled_projections`).

Zero-initialized (matches this codebase's established convention for new conditioning pathways —
`MaskConditioner`'s `gamma`, `DiffusionUNetResnetBlock`'s `zero_module(conv2)`): at init this
projection outputs exactly 0 for any input, so a run with `text_pooled_f/i=None` and one with real
pooled vectors start bit-identical, and the pathway's influence grows in from training rather than
injecting random-init noise into `emb` from step 0.

⚠ Known ceiling: MONAI's `DiffusionUNetResnetBlock` injects `emb` as a plain additive per-channel
bias applied *before* `norm2` (the `"default"` variant in diffusers' `ResnetBlock2D`), so this
conditioning is both weaker than true AdaLN (no multiplicative scale, no gate — DiT/SD3 derive 6
modulation params per block) and partially attenuated by the GroupNorm immediately following it.
Switching to the `"scale_shift"`/AdaGN variant would double every ResBlock's `time_emb_proj` output
width and break checkpoint compatibility, so it is out of scope while finetuning is the goal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.utils.type_conversion import convert_to_tensor
from torch import nn

from src.models.components.masked_cross_attention import enable_context_masking


class DiffusionModelUNetMaisiTextPooled(DiffusionModelUNetMaisi):
    """`DiffusionModelUNetMaisi` + a zero-init additive pooled-text embedding path.

    Args:
        *args, **kwargs: forwarded to `DiffusionModelUNetMaisi.__init__` unchanged.
        text_pooled_dim: width of the pooled findings/impression embedding (2560 for
            Report2CT's 3-encoder concatenation).
    """

    def __init__(
        self, *args: object, text_pooled_dim: int = 2560, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        # Projects to the timestep+class segment's width ONLY, not `emb`'s full width — see
        # the module docstring for why the spacing/region segments are deliberately left out.
        # DUPLICATION INTENTIONAL — mirrors upstream's `time_embed_dim = num_channels[0] * 4`
        # (DiffusionModelUNetMaisi.__init__).
        self.time_embed_dim = self.block_out_channels[0] * 4
        self.text_pooled_proj = nn.Linear(text_pooled_dim, self.time_embed_dim)
        nn.init.zeros_(self.text_pooled_proj.weight)
        nn.init.zeros_(self.text_pooled_proj.bias)
        # Convert the cross-attention blocks in place and keep the handles, so `forward` can
        # publish a key-padding mask without walking the module tree every step. Adds no
        # parameters — see `enable_context_masking`.
        self._xattn_blocks = enable_context_masking(self)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
        text_pooled_f: torch.Tensor | None = None,
        text_pooled_i: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — identical to the base class, with pooled text added into `emb`.

        Args:
            x, timesteps, context, class_labels, down_block_additional_residuals,
                mid_block_additional_residual, top_region_index_tensor,
                bottom_region_index_tensor, spacing_tensor: see
                `DiffusionModelUNetMaisi.forward`, unchanged.
            text_pooled_f: pooled findings embedding, `(B, text_pooled_dim)` or
                `(B, 1, text_pooled_dim)`. `None` (default) ⇒ no contribution.
            text_pooled_i: pooled impression embedding, same shape convention. Findings and
                impression share one projection (same encoder-output space either way) and are
                added as two separate terms rather than pre-combined, so neither is diluted by
                averaging.
            context_mask: key-padding mask over `context`, `(B, T_context)` bool with `True` =
                real token. `None` (default) ⇒ every position is attended, i.e. stock MONAI
                behaviour. Required whenever `context` is a right-padded sequence — see
                `src/models/components/masked_cross_attention.py` for the measured cost of
                leaving padding unmasked.

        Returns:
            Same as the base class: `(B, out_channels, *spatial)`.
        """
        # Publish the mask to the cross-attention blocks, which cannot receive it as an argument
        # (four MONAI layers in between forward only `context`). Set unconditionally — including
        # to None — so no stale mask can survive into a later call; deliberately NOT cleared
        # afterwards, because gradient-checkpoint recomputation replays this forward during
        # backward and would then silently attend to padding.
        for block in self._xattn_blocks:
            block.context_mask = context_mask

        # DUPLICATION INTENTIONAL — copy of upstream forward()
        # (monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py,
        # DiffusionModelUNetMaisi.forward), extended only by the text_pooled_f/i block below.
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(
            emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor
        )
        text_terms = [
            self.text_pooled_proj(t.flatten(1))  # (B, time_embed_dim)
            for t in (text_pooled_f, text_pooled_i)
            if t is not None
        ]
        if text_terms:
            # Right-pad with zeros to `emb`'s full width so the sum lands ONLY on the leading
            # timestep+class segment; any concatenated spacing/region segment receives exactly 0.
            # Zero-width pad (a no-op) when no optional segment is enabled.
            text_emb = sum(text_terms)  # (B, time_embed_dim)
            emb = emb + F.pad(text_emb, (0, emb.shape[1] - text_emb.shape[1]))
        h = self.conv_in(x)
        h, _updated_down_block_res_samples = self._apply_down_blocks(
            h, emb, context, down_block_additional_residuals
        )
        h = self.middle_block(h, emb, context)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, _updated_down_block_res_samples)
        h = self.out(h)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor


__all__ = ["DiffusionModelUNetMaisiTextPooled"]
