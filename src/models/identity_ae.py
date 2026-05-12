from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAE(nn.Module):
    """No-op autoencoder for sanity-checking the training loop.

    encode and decode are both identity, so mu_hat == mu and the
    reconstruction loss should immediately fall to ~0.  Not a real model —
    use it to verify data flow, loss computation, and logging are correct
    before running TriplaneAE.

    Constructor accepts the same kwargs as TriplaneAE so configs are
    interchangeable; all arguments are ignored.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__()
        # A single dummy parameter so that optimizer construction doesn't
        # crash when the training loop calls model.parameters().
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def encode(self, mu: torch.Tensor) -> torch.Tensor:
        return mu

    def decode(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        # For identity, the "reconstructed" volume is just the first argument.
        return args[0]

    def forward(self, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # triplane_dict is empty: identity has no real plane representations.
        # The training loop must tolerate an empty dict here.
        return mu, {}
