"""Thin wrapper around the collaborator's TriVQAEConv (reference/models/trivae2.py).

Adds a from_config classmethod so it can participate in MODEL_REGISTRY without
modifying the collaborator's file.  The forward() signature is unchanged:
returns (reconstruction, zero_loss) — the train loop discards the second
element, so this is compatible with the (recon, triplane) convention.

The collaborator's module lives under reference/ which must be on sys.path
(train.py adds it).  The import is deferred to from_config so that importing
src.models at test/import time does not fail when reference/ is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TriVQAEConv(nn.Module):
    """Registry-compatible wrapper for the collaborator's TriVQAEConv.

    Delegates all work to the underlying implementation; adds from_config.
    Constructed only via from_config, which performs the deferred import.
    """

    def __init__(self, _impl: nn.Module) -> None:
        super().__init__()
        self._impl = _impl

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TriVQAEConv":
        # Deferred import: reference/ must be on sys.path (train.py sets this).
        from models.trivae2 import TriVQAEConv as _Impl  # noqa: PLC0415

        enc = cfg.model.encoder
        dec = cfg.model.decoder
        impl = _Impl(
            in_channels=int(enc.in_channels),
            out_channels=int(getattr(dec, "out_channels", enc.in_channels)),
            latent_shape=tuple(enc.latent_shape),
            encoder_channels=list(enc.encoder_channels),
            blocks_per_stage=int(getattr(enc, "blocks_per_stage", 2)),
            plane_channels=int(enc.plane_channels),
            decoder_channels=(
                list(dec.decoder_channels)
                if getattr(dec, "decoder_channels", None) is not None
                else None
            ),
            final_hidden_channels=int(getattr(dec, "final_hidden_channels", 64)),
            final_depth=int(getattr(dec, "final_depth", 2)),
        )
        return cls(impl)

    def forward(self, z: torch.Tensor, **kwargs):
        return self._impl(z, **kwargs)

    def encode(self, z: torch.Tensor):
        return self._impl.encode(z)

    def decode_planes(self, planes: dict):
        return self._impl.decode_planes(planes)
