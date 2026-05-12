"""Minimal Hydra training script for trial1 (IdentityAE sanity check)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Make src importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra

from src.losses.recon_loss import ReconLoss
from src.models.identity_ae import IdentityAE
from src.models.triplane_ae import TriplaneAE


# ---------------------------------------------------------------------------
# Dummy dataset
# ---------------------------------------------------------------------------


class DummyLatentDataset(Dataset):
    """Returns random [4, 120, 120, 64] tensors — no disk I/O."""

    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Fresh random tensor each call — fine for sanity checking.
        return torch.randn(4, 120, 120, 64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in latent space (data_range = max-min of target)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0.0:
        return float("inf")
    data_range = (target.max() - target.min()).item()
    if data_range == 0.0:
        data_range = 1.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def _ssim_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simple SSIM estimate over the full batch tensor treated as a 1-D signal."""
    try:
        from torchmetrics.functional import (
            structural_similarity_index_measure as ssim_fn,
        )

        # torchmetrics expects (B, C, *spatial) but our tensors are [B, 4, 120, 120, 64].
        # Flatten spatial to 2D: treat Z-slices as "width", H*W as "height".
        B, C, H, W, D = pred.shape
        # Reshape to (B, C, H*W, D) — a pseudo-2D image
        p2 = pred.reshape(B, C, H * W, D)
        t2 = target.reshape(B, C, H * W, D)
        data_range = float((target.max() - target.min()).item()) or 1.0
        val = ssim_fn(p2, t2, data_range=data_range)
        return float(val)
    except Exception:
        # Fallback: closed-form global SSIM approximation.
        mu1 = pred.mean()
        mu2 = target.mean()
        s1 = pred.std()
        s2 = target.std()
        cov = ((pred - mu1) * (target - mu2)).mean()
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        ssim = (
            (2 * mu1 * mu2 + c1)
            * (2 * cov + c2)
            / ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))
        )
        return float(ssim)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(cfg: DictConfig) -> torch.nn.Module:
    kind = cfg.model.kind
    if kind == "IdentityAE":
        return IdentityAE()
    if kind == "TriplaneAE":
        enc = cfg.model.encoder
        dec = cfg.model.decoder
        return TriplaneAE(
            in_channels=int(enc.in_channels),
            emb_dim=int(enc.emb_dim),
            n_layers=int(enc.n_layers),
            n_heads=int(enc.n_heads),
            out_channels=int(enc.out_channels),
            decoder_hidden=int(dec.hidden),
            latent_shape=tuple(enc.latent_shape),
        )
    raise ValueError(f"Unknown model.kind: {kind!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(ROOT / "src" / "configs"),
    config_name="trial1_identity",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    exp_name = cfg.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint dir
    ckpt_dir = ROOT / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        use_wandb = True
    except Exception as e:
        print(f"[warn] wandb init failed: {e} — continuing without wandb")
        run = None
        use_wandb = False

    # Dataset / dataloader
    n_batches = int(cfg.train.n_batches)
    batch_size = int(cfg.train.batch_size)
    dataset = DummyLatentDataset(n_samples=n_batches * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = build_model(cfg).to(device)
    loss_fn = ReconLoss(
        l1_weight=float(cfg.loss.l1_weight),
        lpips_weight=float(cfg.loss.lpips_weight),
        gan_weight=float(cfg.loss.gan_weight),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    print(f"[train] exp={exp_name}  model={cfg.model.kind}  device={device}")
    print(
        f"[train] batches={n_batches}  batch_size={batch_size}  epochs={cfg.train.epochs}"
    )

    global_step = 0
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        for step, mu in enumerate(loader):
            mu = mu.to(device)

            opt.zero_grad(set_to_none=True)
            mu_hat, _triplane = model(mu)  # tolerates empty dict
            losses = loss_fn(mu_hat, mu)
            # Anchor to model params so backward() always has a grad_fn
            # (needed when loss is exactly 0, e.g. IdentityAE).
            total = losses["total"] + 0.0 * sum(p.sum() for p in model.parameters())
            total.backward()
            opt.step()

            # Metrics (detached, no grad)
            with torch.no_grad():
                psnr = _psnr(mu_hat, mu)
                ssim = _ssim_3d(mu_hat, mu)

            log = {
                "loss/total": losses["total"].item(),
                "loss/l1": losses["l1"].item(),
                "metrics/psnr": psnr,
                "metrics/ssim": ssim,
                "step": global_step,
            }

            if use_wandb:
                wandb.log(log, step=global_step)

            if step % 10 == 0:
                print(
                    f"[epoch {epoch} step {step:03d}]  "
                    f"loss={log['loss/total']:.2e}  "
                    f"l1={log['loss/l1']:.2e}  "
                    f"psnr={psnr:.1f}  "
                    f"ssim={ssim:.6f}"
                )

            global_step += 1

        # End-of-epoch print with last-step values
        print(
            f"[epoch {epoch} DONE]  "
            f"final_loss={log['loss/total']:.2e}  "
            f"psnr={log['metrics/psnr']:.2f}  "
            f"ssim={log['metrics/ssim']:.6f}"
        )

        # Save checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        print(f"[checkpoint] {ckpt_path}")

    if use_wandb:
        wandb.finish()
        print(f"[wandb] run URL: {run.url}")
    else:
        print("[wandb] offline — no URL available")


if __name__ == "__main__":
    main()
