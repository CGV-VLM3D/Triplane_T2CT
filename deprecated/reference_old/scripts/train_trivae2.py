"""
Train TriVQAE (models.trivae2.TriVQAE) on pre-extracted MAISI latents.

Loss = lambda_recon * L1(z_hat, z) + lambda_vq * q_loss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.latent_dataset import build_latent_dataloader  # noqa: E402
from models.trivae2 import TriVQAEConv  # noqa: E402


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/workspace/configs/trivae2.yaml")
    known, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(known.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def build_model(cfg: DictConfig) -> TriVQAEConv:
    m = cfg.model
    decoder_channels = m.get("decoder_channels", None)
    if decoder_channels is not None:
        decoder_channels = tuple(decoder_channels)
    return TriVQAEConv(
        in_channels=int(m.in_channels),
        out_channels=m.get("out_channels", None),
        latent_shape=tuple(m.latent_shape),
        encoder_channels=tuple(m.encoder_channels),
        blocks_per_stage=int(m.get("blocks_per_stage", 2)),
        plane_channels=int(m.plane_channels),
        decoder_channels=decoder_channels,
        final_hidden_channels=int(m.get("final_hidden_channels", m.plane_channels)),
        final_depth=int(m.get("final_depth", 2)),
        output_act=m.get("output_act", None),
    )


def save_checkpoint(accelerator, out_dir: Path, epoch: int, model, opt, cfg):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "model": unwrapped.state_dict(),
        "opt": opt.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    path = out_dir / f"trivae2_epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)
    accelerator.print(f"[SAVE] {path}")


def maybe_resume(cfg, model, opt, accelerator) -> int:
    p = cfg.training.get("resume_ckpt", None)
    if not p:
        return 0
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    ckpt = torch.load(str(p), map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped.load_state_dict(ckpt["model"], strict=False)
    accelerator.print(f"[LOAD] {p} missing={len(missing)} unexpected={len(unexpected)}")
    try:
        opt.load_state_dict(ckpt["opt"])
    except (ValueError, KeyError) as e:
        accelerator.print(f"[LOAD] opt incompatible: {e}")
    return int(ckpt.get("epoch", 0))


def train_one_epoch(*, cfg, epoch, epochs, accelerator, loader, model, opt):
    model.train()
    lambda_recon = float(cfg.training.lambda_recon)

    totals = {"loss": 0.0, "recon": 0.0}
    n_iters = 0

    pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}",
                disable=not accelerator.is_local_main_process)

    for batch in pbar:
        z = batch["image"]
        if hasattr(z, "as_tensor"):
            z = z.as_tensor()

        opt.zero_grad(set_to_none=True)

        with accelerator.autocast():
            z_hat, _ = model(z)
            recon = F.l1_loss(z_hat, z)
            loss = lambda_recon * recon

        accelerator.backward(loss)
        grad_clip = float(cfg.training.get("grad_clip", 0.0))
        if grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        gat = lambda t: float(accelerator.gather_for_metrics(t.detach()).mean())
        totals["loss"] += gat(loss)
        totals["recon"] += gat(recon)
        n_iters += 1

        if accelerator.is_local_main_process and n_iters % int(cfg.training.get("log_every", 50)) == 0:
            pbar.set_postfix({k: f"{v / n_iters:.4f}" for k, v in totals.items()})

    n = max(1, n_iters)
    accelerator.print(
        f"[epoch {epoch}] " + " ".join(f"{k}={v / n:.6f}" for k, v in totals.items())
    )


def main():
    cfg = load_config()

    mp = cfg.training.mixed_precision
    if mp == "no":
        mp = "no"
    accelerator = Accelerator(mixed_precision=mp)

    out_dir = Path(cfg.data.out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    accelerator.wait_for_everyone()

    accelerator.print(OmegaConf.to_yaml(cfg))

    loader = build_latent_dataloader(
        latent_dir=cfg.data.latent_dir,
        split="train",
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        shuffle=True,
        drop_last=True,
        sample_z=bool(cfg.data.get("sample_z", True)),
        normalize=bool(cfg.data.get("normalize", True)),
    )

    model = build_model(cfg)
    betas = tuple(cfg.training.get("betas", [0.9, 0.999]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), betas=betas)

    model, opt, loader = accelerator.prepare(model, opt, loader)

    start_epoch = maybe_resume(cfg, model, opt, accelerator)

    accelerator.print(f"dataset size: {len(loader.dataset)}")
    accelerator.print(f"num batches:  {len(loader)}")

    epochs = int(cfg.training.epochs)
    save_every = int(cfg.training.save_every)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_one_epoch(
            cfg=cfg, epoch=epoch, epochs=epochs,
            accelerator=accelerator, loader=loader,
            model=model, opt=opt,
        )
        if epoch % save_every == 0:
            save_checkpoint(accelerator, out_dir, epoch, model, opt, cfg)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
