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
        # Zero-valued parameter that keeps model.parameters() non-empty and
        # anchors forward() into the autograd graph so loss.backward() works
        # even though the reconstruction is exact.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def encode(self, mu: torch.Tensor) -> torch.Tensor:
        return mu

    def decode(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return args[0]

    def forward(self, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # `+ 0 * _dummy` is a no-op numerically but threads the graph through
        # the parameter, so a downstream loss always has a grad_fn.
        mu_hat = mu + 0.0 * self._dummy
        return mu_hat, {}
