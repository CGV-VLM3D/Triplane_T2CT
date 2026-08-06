"""Report2CT-Wan + token-sequence text conditioning LightningDataModule.

Thin subclass of :class:`~src.data.report2ct_datamodule.Report2CTDataModule`, following the
same pattern as :class:`~src.data.report2ct_wan_mask_datamodule.Report2CTWanMaskDataModule`
(base transforms + one extra conditioning signal). It emits everything the plain-wan module
does — ``image``, ``spacing``, ``context_f``/``context_i`` (2560-d pooled findings+impression,
unchanged, still feeds the pooled-conditioning path) — PLUS the per-encoder token sequences that
``Report2CTTextEncoder.encode_tokens`` produces and
``scripts/precompute_report2ct_text_embeddings.py --save-tokens`` writes to the
``<id>_emb.nii.gzmulti_2560_ctx.npz`` sidecar (see docs/vlm3d_research_roadmap.md Phase 1a/1b).

The npz path is **derived from the existing ``context_f`` datalist entry** (same stem, ``.json``
swapped for ``_ctx.npz``) rather than added as a new datalist key — the datalist JSON does not
need to be touched or regenerated. This only works because the sequence-loading transform is
inserted *before* the base pipeline's ``context_f``/``context_i`` ``Lambdad``s, which overwrite
those keys with the already-pooled tensors; see ``_load_token_sequences``.

Padded to a fixed length per section (not per encoder — the 3 encoders disagree on real token
count for the same text, so each gets its own real-vs-pad mask): ``SEQ_LEN_FINDINGS`` covers the
~p90 of findings length, ``SEQ_LEN_IMPRESSION`` the ~p95+ of impression length, measured over 500
precomputed reports (roadmap Phase 1b). Emits 4 new keys per encoder (``context_f_seq_i``,
``context_f_mask_i``, ``context_i_seq_i``, ``context_i_mask_i`` for ``i`` in ``0..2``) rather than
one stacked tensor, because the 3 encoders' hidden widths differ (1024/768/768) and cannot share
a single tensor without re-introducing the zero-padding upstream's ``c_ctx`` used (see
``src/models/components/text_sequence_projector.py``, which consumes exactly these 12 keys as
per-encoder lists).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import torch
from monai.transforms import Compose

from src.data.report2ct_datamodule import Report2CTDataModule, _build_transforms

# Must match Report2CTTextEncoder.MODEL_IDS order (src/baselines/report2ct_text_encoder.py):
# 0=MedEmbed-large-v0.1 (1024d), 1=ClinicalBERT (768d), 2=BiomedVLP-CXR-BERT-specialized (768d).
ENCODER_HIDDEN_DIMS: Final[tuple[int, ...]] = (1024, 768, 768)

# Adopted lengths (docs/vlm3d_research_roadmap.md Phase 1b): findings p50=285/p90=439-468 tokens,
# impression p50=33-47/p90=88-125 tokens, measured per-encoder over 500 precomputed reports.
SEQ_LEN_FINDINGS: Final[int] = 384
SEQ_LEN_IMPRESSION: Final[int] = 128


def _pad_or_truncate(
    arr: np.ndarray | torch.Tensor, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads/truncates one encoder's token sequence to a fixed length, with a real-vs-pad mask.

    Shared by the training-time datamodule (loading a precomputed npz sidecar) and the eval
    sampler (``Report2CTWanSeqLatentSampler``, padding ``Report2CTTextEncoder.encode_tokens``'s
    live output) — accepting either input type keeps both call sites byte-for-byte identical
    instead of maintaining two paddings that could quietly drift apart.

    Args:
        arr: token embeddings for one encoder, ``(n_tokens, hidden)`` — a numpy array (from the
            npz sidecar, float16) or a torch tensor (from a live ``encode_tokens`` call).
        max_len: target sequence length.

    Returns:
        Tuple ``(seq, mask)``: ``seq`` is ``(max_len, hidden)`` float32 (right-truncated if
        ``n_tokens > max_len``, zero-padded on the right otherwise); ``mask`` is ``(max_len,)``
        bool, ``True`` for real tokens and ``False`` for padding.
    """
    n_tokens, hidden = arr.shape
    t = (
        arr.float()
        if torch.is_tensor(arr)
        else torch.from_numpy(arr.astype(np.float32))
    )
    if n_tokens >= max_len:
        return t[:max_len], torch.ones(max_len, dtype=torch.bool)
    seq = torch.cat([t, torch.zeros(max_len - n_tokens, hidden)], dim=0)
    mask = torch.cat(
        [
            torch.ones(n_tokens, dtype=torch.bool),
            torch.zeros(max_len - n_tokens, dtype=torch.bool),
        ]
    )
    return seq, mask


def _load_token_sequences(data: dict[str, Any]) -> dict[str, Any]:
    """Compose step: loads the per-encoder token-sequence sidecar into 12 new dict keys.

    Must run before the base pipeline's ``context_f``/``context_i`` ``Lambdad``s (which
    overwrite those keys with pooled tensors) — this function only reads ``context_f`` to
    derive the sidecar path, matching the naming convention
    ``scripts/precompute_report2ct_text_embeddings.py`` writes:
    ``<id>_emb.nii.gzmulti_2560.json`` -> ``<id>_emb.nii.gzmulti_2560_ctx.npz``.

    Args:
        data: the per-sample dict; must still have ``context_f`` as a raw path string.

    Returns:
        `data`, with ``context_f_seq_{i}``/``context_f_mask_{i}``/``context_i_seq_{i}``/
        ``context_i_mask_{i}`` added for ``i`` in ``range(len(ENCODER_HIDDEN_DIMS))``.
        ``context_f``/``context_i`` themselves are left untouched.
    """
    npz_path = data["context_f"].replace("multi_2560.json", "multi_2560_ctx.npz")
    npz = np.load(npz_path)
    for i in range(len(ENCODER_HIDDEN_DIMS)):
        f_seq, f_mask = _pad_or_truncate(npz[f"findings_{i}"], SEQ_LEN_FINDINGS)
        i_seq, i_mask = _pad_or_truncate(npz[f"impression_{i}"], SEQ_LEN_IMPRESSION)
        data[f"context_f_seq_{i}"] = f_seq
        data[f"context_f_mask_{i}"] = f_mask
        data[f"context_i_seq_{i}"] = i_seq
        data[f"context_i_mask_{i}"] = i_mask
    return data


class Report2CTSeqDataModule(Report2CTDataModule):
    """Report2CT-Wan datamodule that additionally loads per-encoder text token sequences."""

    def _transforms(self) -> Compose:
        """``_load_token_sequences`` (needs the raw path) + base report2ct transforms."""
        base = _build_transforms(self.spacing_multiplier)
        return Compose([_load_token_sequences, *base.transforms])


__all__ = [
    "Report2CTSeqDataModule",
    "ENCODER_HIDDEN_DIMS",
    "SEQ_LEN_FINDINGS",
    "SEQ_LEN_IMPRESSION",
]
