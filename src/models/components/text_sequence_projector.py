"""TextSequenceProjector — per-encoder token-sequence projection for cross-attention.

Report2CT's 3 frozen biomedical text encoders (`Report2CTTextEncoder`, hidden dims
1024/768/768) each tokenize findings/impression independently and disagree on both
hidden width and real token count for the same text (measured median: findings 285 /
305 / 224 tokens across the 3 encoders — see docs/vlm3d_research_roadmap.md Phase 1b).
Upstream's own `c_ctx` construction (`third_party/report2ct/vlm3d_inference.ipynb`
cell 0) reconciles this by zero-padding every encoder's hidden dim up to the largest
(1024) before concatenating along the sequence axis — for the two 768-d encoders that
stores 256 zero columns on every one of their rows.

This module reconciles it instead with a **learned linear projection per encoder** to
a shared width (`out_dim`, `cross_attention_dim`=2560 by convention — see roadmap 1d:
keeping this width lets the projected sequence reuse the pretrained cross-attention
K/V weights from an existing checkpoint, so training can fine-tune rather than start
from scratch). Native dims flow into real learned weights instead of padding zeros.

One projection per encoder is **shared across findings and impression** — the encoder
itself defines a fixed semantic space regardless of which section it tokenized, so
there is no reason to learn separate maps for the two.

Padding/truncation to a fixed sequence length and the real-vs-pad boolean mask are the
caller's responsibility (the datamodule, which knows each encoder's true token count
before padding) — this module only projects and concatenates whatever shape it is
given, so it stays a plain, stateless-shape nn.Module usable identically at both
training (batched, from precomputed sidecars) and inference (`src/eval/samplers/
report2ct.py`, single-sample, encoded on the fly).
"""

from __future__ import annotations

import torch
from torch import nn


class TextSequenceProjector(nn.Module):
    """Projects each text encoder's token sequence to a shared width and concatenates.

    Args:
        encoder_dims: native hidden width of each encoder, in `Report2CTTextEncoder.
            model_ids` order (e.g. `(1024, 768, 768)`).
        out_dim: shared output width (`cross_attention_dim` of the diffusion UNet).
    """

    def __init__(self, encoder_dims: tuple[int, ...], out_dim: int) -> None:
        super().__init__()
        self.encoder_dims = tuple(encoder_dims)
        self.out_dim = out_dim
        self.projections = nn.ModuleList([nn.Linear(d, out_dim) for d in encoder_dims])

    def forward(
        self,
        findings_seqs: list[torch.Tensor],
        findings_masks: list[torch.Tensor],
        impression_seqs: list[torch.Tensor],
        impression_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every encoder's findings + impression sequence, concat into one context.

        Args:
            findings_seqs: per-encoder findings token sequences, each
                ``(B, n_findings, encoder_dims[i])``, already padded to `n_findings`.
            findings_masks: per-encoder findings padding masks, each ``(B, n_findings)``
                bool (`True` = real token, `False` = pad).
            impression_seqs: per-encoder impression token sequences, each
                ``(B, n_impression, encoder_dims[i])``, already padded to `n_impression`.
            impression_masks: per-encoder impression padding masks, each
                ``(B, n_impression)`` bool.

        Returns:
            Tuple ``(context, context_mask)``:
                context: ``(B, n_encoders*(n_findings+n_impression), out_dim)`` —
                    concatenated in encoder order, findings before impression per
                    encoder (matches `Report2CTTextEncoder.model_ids` order).
                context_mask: ``(B, n_encoders*(n_findings+n_impression))`` bool, same
                    concatenation order as `context`.
        """
        assert len(findings_seqs) == len(impression_seqs) == len(self.projections)
        parts = []
        masks = []
        for proj, f_seq, f_mask, i_seq, i_mask in zip(
            self.projections,
            findings_seqs,
            findings_masks,
            impression_seqs,
            impression_masks,
        ):
            parts.append(
                proj(f_seq)
            )  # (B, n_findings, encoder_dims[k]) -> (B, n_findings, out_dim)
            parts.append(
                proj(i_seq)
            )  # (B, n_impression, encoder_dims[k]) -> (B, n_impression, out_dim)
            masks.append(f_mask)
            masks.append(i_mask)
        context = torch.cat(parts, dim=1)  # (B, sum of all n_*, out_dim)
        context_mask = torch.cat(masks, dim=1)  # (B, sum of all n_*)
        return context, context_mask


__all__ = ["TextSequenceProjector"]
