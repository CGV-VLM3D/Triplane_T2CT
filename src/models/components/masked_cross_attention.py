"""Padding-aware cross-attention for MONAI diffusion UNets.

`monai.networks.blocks.crossattention.CrossAttentionBlock.forward(x, context)` takes no
attention mask, and neither does any of the four layers above it
(`DiffusionModelUNetMaisi` -> `CrossAttnDownBlock`/`CrossAttnMidBlock`/`CrossAttnUpBlock`
-> `SpatialTransformer` -> `DiffusionUNetTransformerBlock` -> `CrossAttentionBlock`), so a
right-padded text context is attended to as if the padding were real tokens.

Why that matters here (measured on 64 real precomputed reports, `TextSequenceProjector`
output at `SEQ_LEN_FINDINGS=384` / `SEQ_LEN_IMPRESSION=128`, 3 encoders => 1536 positions):
padding absorbs **40.1% of the cross-attention probability mass on average, up to 82.2%** for
short reports, attenuating the attention output to **0.602x on average and 0.194x** in the worst
case. Because the attenuation tracks each report's own length it varies per sample (0.19x-0.99x),
so the network cannot learn a single compensating scale. Note the padded positions are not even
inert: `TextSequenceProjector`'s `nn.Linear` has a bias, so a zero-padded row projects to that
bias vector and contributes real (spurious) content through `to_v`, not just dilution.

Precedent is genuinely split — SD1.x/SDXL/SD3/Flux ship unmasked cross-attention, while
PixArt-alpha passes `encoder_attention_mask` — but the unmasked systems all condition on short,
near-fully-populated sequences (CLIP's fixed 77, or T5 at a few hundred), a regime where the
padded fraction is small. At a 1536-slot context whose real occupancy ranges from 18% to 100%,
the measurements above put us far outside that regime, so we mask.

Masking is applied by swapping the *class* of already-constructed `CrossAttentionBlock`
instances (see `enable_context_masking`) rather than by threading a `mask` argument through
four MONAI signatures we would then have to keep in sync with upstream.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.networks.blocks.crossattention import CrossAttentionBlock


class MaskedCrossAttentionBlock(CrossAttentionBlock):
    """`CrossAttentionBlock` that honours a key-padding mask stored on the instance.

    Adds no parameters, buffers, or `__init__` of its own — existing instances are converted
    in place by reassigning `__class__` (`enable_context_masking`), so weights and
    `state_dict()` keys are untouched and a converted UNet still loads a stock checkpoint.

    The mask is read from `self.context_mask` (set by the owning UNet before its forward)
    instead of being passed as an argument, because the four MONAI layers between the UNet
    and this block do not forward extra kwargs.
    """

    #: Key-padding mask, ``(B, T_context)`` bool with ``True`` = real token. ``None`` disables
    #: masking, making this class behave exactly like `CrossAttentionBlock`.
    context_mask: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Cross-attend `x` to `context`, ignoring positions marked padding in `context_mask`.

        Args:
            x: query sequence, ``(B, N_query, C)``.
            context: key/value sequence, ``(B, T_context, context_dim)``. ``None`` falls back to
                self-attention (upstream behaviour), in which case no mask is applied.

        Returns:
            ``(B, N_query, C)``.
        """
        mask = self.context_mask
        if mask is None or context is None:
            return super().forward(x, context)

        # DUPLICATION INTENTIONAL — reproduces CrossAttentionBlock.forward
        # (monai/networks/blocks/crossattention.py), adding only the two mask applications.
        # Kept minimal: the `rel_positional_embedding` / `save_attn` / `causal` branches are
        # unreachable in this UNet (verified: cross-attn blocks are built with defaults), and
        # `causal` is asserted below rather than silently combined with a padding mask.
        if self.causal:
            raise NotImplementedError(
                "MaskedCrossAttentionBlock does not support causal=True together with a "
                "context mask; this UNet builds cross-attention with causal=False."
            )

        q = self.input_rearrange(self.to_q(x))  # (B, heads, N_query, head_dim)
        k = self.input_rearrange(self.to_k(context))  # (B, heads, T_context, head_dim)
        v = self.input_rearrange(self.to_v(context))  # (B, heads, T_context, head_dim)
        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        # (B, T) -> (B, 1, 1, T) so it broadcasts over heads and query positions.
        attn_mask = mask.to(device=q.device)[:, None, None, :]

        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,  # bool: True = attend
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=False,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
            # finfo.min rather than -inf: if a query's keys were ever all padding, an all--inf
            # row softmaxes to NaN, whereas this degrades to a uniform (harmless) distribution.
            # Real reports always carry >= 3 tokens per section, so that row cannot occur today.
            att_mat = att_mat.masked_fill(~attn_mask, torch.finfo(att_mat.dtype).min)
            att_mat = att_mat.softmax(dim=-1)
            att_mat = self.drop_weights(att_mat)
            out = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        out = self.out_rearrange(out)
        out = self.out_proj(out)
        out = self.drop_output(out)
        return out


def enable_context_masking(module: torch.nn.Module) -> list[MaskedCrossAttentionBlock]:
    """Convert every `CrossAttentionBlock` under `module` into a `MaskedCrossAttentionBlock`.

    Reassigns `__class__` in place: `MaskedCrossAttentionBlock` declares no `__init__` and no
    new parameters or buffers, so every weight and `state_dict()` key is preserved and the
    conversion is invisible to checkpointing. Rebuilding the blocks instead would mean
    re-deriving each one's constructor arguments from the built module — fragile, and pointless
    when the only difference is the forward.

    Self-attention is untouched: MONAI's `SABlock` is a separate class, not a
    `CrossAttentionBlock` subclass (verified), so `isinstance` cannot catch it by accident.

    Args:
        module: any module containing cross-attention blocks (e.g. the diffusion UNet).

    Returns:
        The converted blocks, in `named_modules()` order, so the caller can set
        `context_mask` on them without re-walking the tree.
    """
    converted: list[MaskedCrossAttentionBlock] = []
    for submodule in module.modules():
        if isinstance(submodule, CrossAttentionBlock) and not isinstance(
            submodule, MaskedCrossAttentionBlock
        ):
            submodule.__class__ = MaskedCrossAttentionBlock
            submodule.context_mask = None
            converted.append(submodule)
    return converted


__all__ = ["MaskedCrossAttentionBlock", "enable_context_masking"]
