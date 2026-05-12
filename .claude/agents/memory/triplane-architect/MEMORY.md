# Triplane Architect Memory

- **Trial 1 implemented** (`src/models/`, `src/losses/`, `src/data/`, `src/configs/trial1.yaml`): shared-weight transformer encoder (one `TransformerEncoderLayer` called `n_layers` times in a loop), expand+sum+Conv3d decoder. Tests use `emb_dim=64, n_heads=4` to stay CPU-feasible; production config uses 512/8.
- **Expand + contiguous()**: `z_yz.unsqueeze(2).expand(...).contiguous()` is required before Conv3d — `expand()` gives a non-contiguous stride-0 view that Conv3d rejects without it.
- **IdentityAE** (`src/models/identity_ae.py`): no-op baseline; `forward` returns `(mu, {})` — empty triplane dict is intentional, training loop must tolerate it. Carries a single `_dummy` nn.Parameter so `model.parameters()` is never empty and optimizer construction doesn't fail.
