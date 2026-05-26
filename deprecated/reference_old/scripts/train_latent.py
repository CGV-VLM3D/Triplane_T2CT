"""
Train the conv-based triplane VQAE (`models.trivqvae_conv.TriVQAEConv`)
on pre-extracted MAISI latents.

Pipeline (per training step):
    z (normalized MAISI latent) -> TriVQAEConv -> z_hat
    loss = lambda_recon * L1(z_hat, z) + lambda_vq * commitment_loss

Optional image-domain losses (image L1, LPIPS, GAN) are kept as stubs
mirroring train_latent.py — set non-zero lambdas + provide MAISI bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.latent_dataset import build_latent_dataloader
from models.trivqvae_conv import TriVQAEConv


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/configs/trivqae_conv.yaml",
    )
    known, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(known.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def as_plain_tensor(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "as_tensor"):
        return x.as_tensor()
    return x


def set_requires_grad(model, flag: bool) -> None:
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad_(flag)


def unpack_output(output):
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    if isinstance(output, dict):
        return output["reconstruction"], output["q_loss"]
    raise RuntimeError(f"Unexpected output: {type(output)}")


def get_disc_logits(output):
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def build_generator(cfg: DictConfig) -> TriVQAEConv:
    latent_shape = tuple(cfg.training.spatial_size)

    print(f"[build_generator] latent_shape={latent_shape}")
    print(f"[build_generator] in/out channels={cfg.model.in_channels}")

    decoder_channels = cfg.model.get("decoder_channels", None)
    if decoder_channels is not None:
        decoder_channels = tuple(decoder_channels)

    return TriVQAEConv(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.in_channels),
        latent_shape=latent_shape,
        encoder_channels=tuple(cfg.model.encoder_channels),
        blocks_per_stage=int(cfg.model.get("blocks_per_stage", 2)),
        plane_channels=int(cfg.model.plane_channels),
        decoder_channels=decoder_channels,
        final_hidden_channels=int(cfg.model.get("final_hidden_channels", cfg.model.plane_channels)),
        final_depth=int(cfg.model.get("final_depth", 2)),
        output_act=cfg.model.get("output_act", None),
    )


# ---------------------------------------------------------------------------
# MAISI image-domain loss support (kept identical to train_latent.py).
# ---------------------------------------------------------------------------
def build_maisi_autoencoder(cfg: DictConfig):
    lambda_img = float(cfg.training.get("lambda_image_l1", 0.0))
    lambda_lpips = float(cfg.training.get("lambda_lpips", 0.0))
    lambda_adv = float(cfg.training.get("lambda_adv", 0.0))
    if lambda_img <= 0 and lambda_lpips <= 0 and lambda_adv <= 0:
        return None

    bundle_dir = cfg.maisi.get("bundle_dir", None)
    if bundle_dir is None:
        raise ValueError("maisi.bundle_dir is required when image-domain losses are enabled.")
    bundle_dir = Path(bundle_dir).resolve()

    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))

    from monai.bundle import ConfigParser

    cfg_path = bundle_dir / "configs" / "inference.json"
    if not cfg_path.exists():
        cfg_path = bundle_dir / "configs" / "inference.yaml"

    parser = ConfigParser()
    parser.read_config(str(cfg_path))
    parser.parse()

    network_key = cfg.maisi.get("network_key", "autoencoder_def")
    net = parser.get_parsed_content(network_key)

    ckpt_path = cfg.maisi.get("ae_ckpt", None)
    if ckpt_path is None:
        cands = sorted((bundle_dir / "models").glob("autoencoder*.pt"))
        cands += sorted((bundle_dir / "models").glob("autoencoder*.pth"))
        if not cands:
            raise FileNotFoundError(f"No autoencoder ckpt under {bundle_dir/'models'}")
        ckpt_path = cands[0]
    ckpt_path = Path(ckpt_path)

    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith(("encoder", "decoder", "quant")) for k in sd
    ):
        sd = sd["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[maisi] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")

    net.eval()
    set_requires_grad(net, False)
    return net


def maisi_decode(ae, mu_unnorm: torch.Tensor) -> torch.Tensor:
    if hasattr(ae, "decode_stage_2_outputs"):
        return ae.decode_stage_2_outputs(mu_unnorm)
    if hasattr(ae, "decode"):
        out = ae.decode(mu_unnorm)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
    raise RuntimeError("MAISI autoencoder has no decode/decode_stage_2_outputs.")


def load_norm_stats(cfg: DictConfig) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not cfg.data.get("normalize", True):
        return None, None
    stats_path = Path(cfg.data.latent_dir) / "stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["channel_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(stats["channel_std"], dtype=torch.float32).clamp(min=1e-6).view(1, -1, 1, 1, 1)
    return mean, std


def denormalize_latent(z_norm, mean, std):
    if mean is None:
        return z_norm
    return z_norm * std.to(z_norm.device, z_norm.dtype) + mean.to(z_norm.device, z_norm.dtype)


def random_slice_triplet_pair(fake, real, out_size):
    if fake.shape != real.shape:
        raise ValueError(f"fake {fake.shape} vs real {real.shape}")
    B, C, D, H, W = real.shape
    d = torch.randint(0, D, (1,), device=real.device).item()
    h = torch.randint(0, H, (1,), device=real.device).item()
    w = torch.randint(0, W, (1,), device=real.device).item()
    fakes = [fake[:, :, d, :, :], fake[:, :, :, h, :], fake[:, :, :, :, w]]
    reals = [real[:, :, d, :, :], real[:, :, :, h, :], real[:, :, :, :, w]]
    fakes = [F.interpolate(f, size=out_size, mode="bilinear", align_corners=False) for f in fakes]
    reals = [F.interpolate(r, size=out_size, mode="bilinear", align_corners=False) for r in reals]
    return torch.cat(fakes, dim=0), torch.cat(reals, dim=0)


def build_lpips_model(cfg: DictConfig, device: torch.device):
    if float(cfg.training.get("lambda_lpips", 0.0)) <= 0:
        return None
    import lpips
    net = cfg.training.get("lpips_net", "vgg")
    model = lpips.LPIPS(net=net).eval().to(device)
    set_requires_grad(model, False)
    return model


def compute_slice_lpips(lpips_model, x_hat, x, out_size=(224, 224)):
    if lpips_model is None:
        return torch.zeros((), device=x.device)
    fake_slices, real_slices = random_slice_triplet_pair(
        torch.clamp(x_hat, 0.0, 1.0),
        torch.clamp(x, 0.0, 1.0),
        out_size=out_size,
    )
    fake_slices = fake_slices.repeat(1, 3, 1, 1) * 2.0 - 1.0
    real_slices = real_slices.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.cuda.amp.autocast(enabled=False):
        return lpips_model(fake_slices.float(), real_slices.float()).mean()


def build_discriminator(cfg: DictConfig):
    if float(cfg.training.get("lambda_adv", 0.0)) <= 0:
        return None
    from generative.networks.nets import PatchDiscriminator
    disc = PatchDiscriminator(
        spatial_dims=int(cfg.discriminator.get("spatial_dims", 2)),
        num_layers_d=cfg.discriminator.disc_layers,
        num_channels=cfg.discriminator.disc_channels,
        in_channels=1,
        out_channels=1,
    )
    if cfg.discriminator.get("sync_batchnorm", False):
        disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disc)
    return disc


def get_discriminator_inputs(cfg, fake, real):
    disc_spatial_dims = int(cfg.discriminator.get("spatial_dims", 2))
    if disc_spatial_dims == 3:
        return fake, real
    if disc_spatial_dims == 2:
        slice_size = tuple(cfg.discriminator.get("slice_size", [256, 256]))
        return random_slice_triplet_pair(fake, real, out_size=slice_size)
    raise ValueError(f"Unsupported discriminator.spatial_dims={disc_spatial_dims}")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(accelerator, out_dir: Path, epoch: int, generator,
                    discriminator, opt_g, opt_d, cfg):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(generator)
    ckpt = {
        "epoch": epoch,
        "generator": unwrapped.state_dict(),
        "opt_g": opt_g.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if discriminator is not None:
        ckpt["discriminator"] = accelerator.unwrap_model(discriminator).state_dict()
    if opt_d is not None:
        ckpt["opt_d"] = opt_d.state_dict()
    path = out_dir / f"trivqae_conv_epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)
    accelerator.print(f"[SAVE] {path}")


def maybe_resume(cfg, generator, discriminator, opt_g, opt_d, accelerator) -> int:
    resume_ckpt = cfg.training.get("resume_ckpt", None)
    if not resume_ckpt:
        return 0
    p = Path(resume_ckpt)
    if not p.exists():
        raise FileNotFoundError(p)
    ckpt = torch.load(str(p), map_location="cpu")
    unwrapped = accelerator.unwrap_model(generator)
    missing, unexpected = unwrapped.load_state_dict(ckpt["generator"], strict=False)
    accelerator.print(f"[LOAD] {p} missing={len(missing)} unexpected={len(unexpected)}")
    mode = cfg.training.get("resume_mode", "generator_only")
    if mode == "generator_only":
        return 0
    if discriminator is not None and "discriminator" in ckpt:
        accelerator.unwrap_model(discriminator).load_state_dict(
            ckpt["discriminator"], strict=False)
    if mode == "full":
        try:
            opt_g.load_state_dict(ckpt["opt_g"])
        except (ValueError, KeyError) as e:
            accelerator.print(f"[LOAD] opt_g incompatible: {e}")
        if opt_d is not None and ckpt.get("opt_d") is not None:
            try:
                opt_d.load_state_dict(ckpt["opt_d"])
            except (ValueError, KeyError) as e:
                accelerator.print(f"[LOAD] opt_d incompatible: {e}")
    return int(ckpt.get("epoch", 0))


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def train_one_epoch(*, cfg, phase, epoch, epochs, accelerator, loader,
                    generator, discriminator, opt_g, opt_d,
                    maisi_ae, mean, std, lpips_model, adv_loss_fn):
    generator.train()
    if discriminator is not None:
        discriminator.train() if phase == "gan" else discriminator.eval()

    lambda_recon = float(cfg.training.lambda_recon)
    lambda_vq = 0.0  # no VQ in this conv variant
    lambda_img = float(cfg.training.get("lambda_image_l1", 0.0))
    lambda_lpips = float(cfg.training.get("lambda_lpips", 0.0))
    lambda_adv = float(cfg.training.get("lambda_adv", 0.0)) if phase == "gan" else 0.0
    use_image_domain = (lambda_img > 0 or lambda_lpips > 0 or lambda_adv > 0)
    lpips_slice_size = tuple(cfg.training.get("lpips_slice_size", [224, 224]))

    totals = {"G": 0.0, "D": 0.0, "recon": 0.0,
              "img_l1": 0.0, "lpips": 0.0, "adv_g": 0.0}

    unwrapped = accelerator.unwrap_model(generator)

    pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}",
                disable=not accelerator.is_local_main_process)

    for batch in pbar:
        z = as_plain_tensor(batch["image"])

        set_requires_grad(discriminator, False)
        set_requires_grad(generator, True)
        opt_g.zero_grad(set_to_none=True)

        with accelerator.autocast():
            output = generator(z)
            z_hat, q_loss = unpack_output(output)
            recon = F.smooth_l1_loss(z_hat, z, beta=0.3)

        if use_image_domain:
            mu_hat = denormalize_latent(z_hat, mean, std)
            mu_real = denormalize_latent(z, mean, std)
            with accelerator.autocast():
                x_hat = maisi_decode(maisi_ae, mu_hat)
            with torch.no_grad():
                x_target = maisi_decode(maisi_ae, mu_real)
            img_l1 = F.l1_loss(x_hat, x_target) if lambda_img > 0 else torch.zeros((), device=z.device)
            lpips_loss = compute_slice_lpips(lpips_model, x_hat, x_target, out_size=lpips_slice_size)
            if lambda_adv > 0:
                with accelerator.autocast():
                    fake_for_g, _ = get_discriminator_inputs(cfg, x_hat.contiguous(), x_target.contiguous())
                    fake_logits_g = get_disc_logits(discriminator(fake_for_g.float()))
                    adv_g = adv_loss_fn(fake_logits_g, target_is_real=True, for_discriminator=False)
            else:
                adv_g = torch.zeros((), device=z.device)
        else:
            x_hat = x_target = None
            img_l1 = torch.zeros((), device=z.device)
            lpips_loss = torch.zeros((), device=z.device)
            adv_g = torch.zeros((), device=z.device)

        loss_g = (
            lambda_recon * recon
            + lambda_vq * q_loss
            + lambda_img * img_l1
            + lambda_lpips * lpips_loss
            + lambda_adv * adv_g
        )

        accelerator.backward(loss_g)
        grad_clip = float(cfg.training.get("grad_clip", 0.0))
        if grad_clip > 0:
            accelerator.clip_grad_norm_(generator.parameters(), grad_clip)
        opt_g.step()

        if phase == "gan" and lambda_adv > 0 and use_image_domain:
            set_requires_grad(discriminator, True)
            set_requires_grad(generator, False)
            opt_d.zero_grad(set_to_none=True)

            x_hat_det = x_hat.detach()
            fake_for_d, real_for_d = get_discriminator_inputs(cfg, x_hat_det, x_target)

            with accelerator.autocast():
                fake_logits_d = get_disc_logits(discriminator(fake_for_d.float()))
                loss_d_fake = 0.5 * adv_loss_fn(fake_logits_d, target_is_real=False, for_discriminator=True)
            accelerator.backward(loss_d_fake)
            with accelerator.autocast():
                real_logits_d = get_disc_logits(discriminator(real_for_d.float()))
                loss_d_real = 0.5 * adv_loss_fn(real_logits_d, target_is_real=True, for_discriminator=True)
            accelerator.backward(loss_d_real)
            opt_d.step()
            loss_d = loss_d_fake.detach() + loss_d_real.detach()
            set_requires_grad(generator, True)
        else:
            loss_d = torch.zeros((), device=z.device)

        gat = lambda t: float(accelerator.gather_for_metrics(t.detach()).mean())
        totals["G"] += gat(loss_g)
        totals["D"] += gat(loss_d)
        totals["recon"] += gat(recon)
        totals["img_l1"] += gat(img_l1)
        totals["lpips"] += gat(lpips_loss)
        totals["adv_g"] += gat(adv_g)

        if accelerator.is_local_main_process:
            pbar.set_postfix({k: f"{v / max(1, pbar.n + 1):.4f}" for k, v in totals.items()})

    n = len(loader)
    accelerator.print(
        f"[epoch {epoch}] "
        + " ".join(f"{k}={v / n:.6f}" for k, v in totals.items())
    )


def main():
    cfg = load_config()

    phase = cfg.training.get("phase", "ae")
    if phase not in ("ae", "gan"):
        raise ValueError("training.phase must be 'ae' or 'gan'")

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision
        if cfg.training.mixed_precision != "no" else "no",
    )

    out_dir = Path(cfg.data.out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    accelerator.wait_for_everyone()

    accelerator.print("phase:", phase)
    accelerator.print(OmegaConf.to_yaml(cfg))

    loader = build_latent_dataloader(
        latent_dir=cfg.data.latent_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=True,
        normalize=cfg.data.get("normalize", True),
    )

    generator = build_generator(cfg)
    discriminator = build_discriminator(cfg) if phase == "gan" else None
    maisi_ae = build_maisi_autoencoder(cfg)
    mean, std = load_norm_stats(cfg)

    g_betas = tuple(cfg.training.get("betas_g", None) or
                    ((0.5, 0.9) if phase == "gan" else (0.9, 0.999)))
    d_betas = tuple(cfg.training.get("betas_d", None) or (0.5, 0.9))

    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.training.lr_g, betas=g_betas)
    opt_d = (torch.optim.AdamW(discriminator.parameters(),
                                lr=cfg.training.lr_d, betas=d_betas)
             if discriminator is not None else None)
    accelerator.print(f"[main] AdamW betas g={g_betas} d={d_betas}")

    from generative.losses import PatchAdversarialLoss
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    prep = [generator, opt_g, loader]
    if discriminator is not None:
        prep = [generator, discriminator, opt_g, opt_d, loader]
    prepared = accelerator.prepare(*prep)
    if discriminator is not None:
        generator, discriminator, opt_g, opt_d, loader = prepared
    else:
        generator, opt_g, loader = prepared

    if maisi_ae is not None:
        maisi_ae = maisi_ae.to(accelerator.device).eval()
        set_requires_grad(maisi_ae, False)

    lpips_model = build_lpips_model(cfg, accelerator.device)

    start_epoch = maybe_resume(cfg, generator, discriminator, opt_g, opt_d, accelerator)

    accelerator.print("dataset size:", len(loader.dataset))
    accelerator.print("num batches:", len(loader))

    epochs = int(cfg.training.epochs)
    save_every = int(cfg.training.save_every)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_one_epoch(
            cfg=cfg, phase=phase, epoch=epoch, epochs=epochs,
            accelerator=accelerator, loader=loader,
            generator=generator, discriminator=discriminator,
            opt_g=opt_g, opt_d=opt_d,
            maisi_ae=maisi_ae, mean=mean, std=std,
            lpips_model=lpips_model, adv_loss_fn=adv_loss_fn,
        )
        if epoch % save_every == 0:
            save_checkpoint(accelerator, out_dir, epoch, generator,
                            discriminator, opt_g, opt_d, cfg)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
