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
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "reference" / "scripts"))

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

    def __getitem__(self, idx: int) -> dict:
        return {"mu": torch.randn(4, 120, 120, 64)}


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
# MAISI decoder loader (used for image-domain validation)
# ---------------------------------------------------------------------------


def _load_maisi_decoder(device: torch.device):
    """Load the frozen MAISI autoencoder and return net.decode_stage_2_outputs."""
    BUNDLE_DIR = ROOT / "maisi_bundle"
    sys.path.insert(0, str(BUNDLE_DIR))
    from extract_maisi_latent import (
        add_bundle_to_syspath,
        build_autoencoder,
        find_ae_ckpt,
        load_state,
    )

    add_bundle_to_syspath(BUNDLE_DIR)
    net = build_autoencoder(BUNDLE_DIR, "autoencoder_def")
    ckpt_path = find_ae_ckpt(BUNDLE_DIR, None)
    load_state(net, ckpt_path)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _warmup_maisi_decoder(decoder, device: torch.device) -> None:
    """Run 3 sliding-window passes through `decoder` so torch.compile's kernel
    cache is hot before the first timed validation. Mirrors measure_upper_bound.py."""
    from monai.inferers import SlidingWindowInferer

    inferer = SlidingWindowInferer(
        roi_size=(20, 20, 20),
        sw_batch_size=16,
        mode="gaussian",
        overlap=0.4,
        device=device,
        sw_device=device,
        progress=False,
    )
    dummy_mu = torch.zeros(1, 4, 120, 120, 64, device=device)
    for _ in range(3):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = inferer(dummy_mu, decoder)
        torch.cuda.synchronize()
    del dummy_mu
    torch.cuda.empty_cache()


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

    use_dummy = bool(getattr(cfg.train, "dummy_data", True))

    # Dataset / dataloader
    n_batches = int(cfg.train.n_batches)
    batch_size = int(cfg.train.batch_size)

    if use_dummy:
        dataset = DummyLatentDataset(n_samples=n_batches * batch_size)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        val_loader = None
    else:
        from src.data.maisi_latent_dataset import MAISILatentDataset

        train_ds = MAISILatentDataset(split="train")
        dataset = train_ds
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        val_split = getattr(cfg.eval, "val_split", "valid")
        # load_ct_recon only when we'll need image metrics
        image_metric_every = int(getattr(cfg.eval, "image_metric_every_n_epochs", 5))
        need_ct_recon = image_metric_every > 0
        val_ds = MAISILatentDataset(split=val_split, load_ct_recon=need_ct_recon)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

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

    # Build MAISI decoder once if image-metric validation is enabled.
    maisi_net = None
    maisi_decoder = None
    image_metric_every = int(getattr(cfg.eval, "image_metric_every_n_epochs", 5))
    if not use_dummy and val_loader is not None and image_metric_every > 0:
        print("[eval] Loading MAISI decoder for image-domain validation...")
        try:
            maisi_net = _load_maisi_decoder(device)
            maisi_decoder = torch.compile(
                maisi_net.decode_stage_2_outputs, mode="reduce-overhead"
            )
            print("[eval] Warming up torch.compile (3 passes, ~30-60s)...")
            _warmup_maisi_decoder(maisi_decoder, device)
            print("[eval] MAISI decoder ready.")
        except Exception as e:
            print(f"[eval] Failed to load MAISI decoder: {e} — image metrics disabled.")
            maisi_decoder = None

    fast_n_samples = int(getattr(cfg.eval, "fast_n_samples", 200))
    image_metric_n_samples = int(getattr(cfg.eval, "image_metric_n_samples", 50))

    global_step = 0
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        for step, batch in enumerate(loader):
            # Support both dict-style (real dataset) and tensor-style (legacy dummy).
            if isinstance(batch, dict):
                mu = batch["mu"].to(device)
            else:
                mu = batch.to(device)

            opt.zero_grad(set_to_none=True)
            mu_hat, _triplane = model(mu)
            losses = loss_fn(mu_hat, mu)
            losses["total"].backward()
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

        # --- Validation ---
        if val_loader is not None:
            from src.evaluation.validate import run_validation

            # Cheap latent-only validation every epoch.
            with torch.no_grad():
                val_metrics = run_validation(
                    model=model,
                    maisi_decoder=None,
                    val_loader=val_loader,
                    n_samples=fast_n_samples,
                    device=str(device),
                    compute_image_metrics=False,
                )

            # Full image-domain validation every N epochs.
            if maisi_decoder is not None and epoch % image_metric_every == 0:
                with torch.no_grad():
                    img_metrics = run_validation(
                        model=model,
                        maisi_decoder=maisi_decoder,
                        val_loader=val_loader,
                        n_samples=image_metric_n_samples,
                        device=str(device),
                        compute_image_metrics=True,
                    )
                val_metrics.update(img_metrics)

            val_log = {f"val/{k}": v for k, v in val_metrics.items() if v is not None}
            if use_wandb:
                wandb.log(val_log, step=global_step)

            print(
                f"[val epoch {epoch}]  "
                + "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in val_log.items()
                    if isinstance(v, float)
                )
            )

    if use_wandb:
        wandb.finish()
        print(f"[wandb] run URL: {run.url}")
    else:
        print("[wandb] offline — no URL available")


if __name__ == "__main__":
    main()
