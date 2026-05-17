"""Hydra training script for the triplane autoencoder."""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Make src importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "reference" / "scripts"))

import hydra

from src.losses.recon_loss import ReconLoss
from src.metrics import (
    compute_latent_data_range,
    latent_cosine_similarity,
    latent_l1,
    latent_psnr,
)
from src.utils import discord_notifier as notifier


def build_model(cfg: DictConfig) -> torch.nn.Module:
    from src.models import MODEL_REGISTRY

    kind = str(getattr(cfg.model, "kind", "TriplaneAE"))
    if kind not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.kind={kind!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[kind].from_config(cfg)


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
# Snapshot save / resume helpers
# ---------------------------------------------------------------------------


def _capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as e:
            print(f"[resume] WARN: cuda RNG restore failed ({e}); continuing.")
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    completed_steps_in_epoch: int,
    epoch_done: bool,
    data_range: float,
    cfg: DictConfig,
    wandb_run_id: Optional[str],
    extra: Optional[dict] = None,
) -> None:
    """Atomically write a checkpoint via tmp+rename.

    Contents are sufficient to fully resume training: model + optimizer state,
    epoch / step counters, all RNG states, the data_range used for PSNR, and
    the wandb run id so the wandb run can be resumed too.
    """
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "completed_steps_in_epoch": completed_steps_in_epoch,
        "epoch_done": epoch_done,
        "data_range": data_range,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "rng": _capture_rng_state(),
        "wandb_run_id": wandb_run_id,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _resolve_resume_path(ckpt_dir: Path, resume_cfg) -> Optional[Path]:
    """Return path to checkpoint to resume from, or None for a fresh run.

    Accepts: True / "auto" / "true" / "yes"  -> use ckpt_dir/latest.pt if exists,
             False / None / "off" / "false" / "no" / "none" -> no resume,
             any other value -> treated as explicit checkpoint path.
    """
    if resume_cfg in (False, None) or (
        isinstance(resume_cfg, str)
        and resume_cfg.lower() in {"false", "off", "no", "none", ""}
    ):
        return None
    if resume_cfg is True or (
        isinstance(resume_cfg, str) and resume_cfg.lower() in {"true", "auto", "yes"}
    ):
        latest = ckpt_dir / "latest.pt"
        return latest if latest.exists() else None
    p = Path(str(resume_cfg))
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"[resume] checkpoint path does not exist: {p}")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(ROOT / "src" / "configs"),
    config_name="trial2",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    exp_name = cfg.exp_name
    # Mutable holder so the except branch can see the wandb URL once it's known.
    state = {"wandb_url": None}
    try:
        _run_training(cfg, state)
    except BaseException as e:
        notifier.notify_exception(
            exp_name=exp_name, exc=e, wandb_url=state.get("wandb_url")
        )
        raise


def _run_training(cfg: DictConfig, state: dict) -> None:
    exp_name = cfg.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-run directory: runs/<exp_name>/{checkpoints,figs,logs,hydra}
    run_dir = ROOT / "runs" / exp_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    # ---- Resume detection (must run BEFORE wandb.init so we can resume the run)
    resume_cfg = getattr(cfg.train, "resume", "auto")
    resume_path = _resolve_resume_path(ckpt_dir, resume_cfg)
    resume_state = None
    resumed_wandb_id: Optional[str] = None
    if resume_path is not None:
        print(f"[resume] loading snapshot: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        resumed_wandb_id = resume_state.get("wandb_run_id")
        print(
            f"[resume]   epoch={resume_state.get('epoch')}  "
            f"global_step={resume_state.get('global_step')}  "
            f"completed_in_epoch={resume_state.get('completed_steps_in_epoch')}  "
            f"epoch_done={resume_state.get('epoch_done')}"
        )
    else:
        print("[resume] starting fresh (no snapshot to resume from)")

    # Wandb
    try:
        import wandb

        wandb_kwargs = dict(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if resumed_wandb_id:
            wandb_kwargs["id"] = resumed_wandb_id
            wandb_kwargs["resume"] = "allow"
            print(f"[wandb] resuming run id={resumed_wandb_id}")
        run = wandb.init(**wandb_kwargs)
        wandb_run_id = run.id
        wandb_url = run.url
        state["wandb_url"] = wandb_url
        use_wandb = True
    except Exception as e:
        print(f"[warn] wandb init failed: {e} — continuing without wandb")
        run = None
        wandb_run_id = None
        wandb_url = None
        use_wandb = False

    from src.data.maisi_latent_dataset import MAISILatentDataset

    # data.root lets a config point at a non-default latent folder (e.g. the
    # toy Path-B latents at /workspace/data/latents_2mm). Falls back to the
    # collaborator's main 1mm latents when unset.
    data_root = str(
        getattr(
            getattr(cfg, "data", None), "root", "/workspace/datasets/datasets/latents"
        )
    )

    batch_size = int(cfg.train.batch_size)
    train_ds = MAISILatentDataset(split="train", root=data_root)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_split = getattr(cfg.eval, "val_split", "valid")
    image_metric_every = int(getattr(cfg.eval, "image_metric_every_n_epochs", 5))
    need_ct_recon = image_metric_every > 0
    val_ds = MAISILatentDataset(
        split=val_split, root=data_root, load_ct_recon=need_ct_recon
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # data_range for latent PSNR — computed once from the first batch.
    # stats.json is channel-wise, not suitable as a PSNR data_range; the first
    # batch's global p1-p99 is stable enough. On resume we reuse the saved
    # value so PSNR is comparable across the wandb run.
    if resume_state is not None and resume_state.get("data_range") is not None:
        data_range = float(resume_state["data_range"])
        print(f"[train] latent data_range from snapshot = {data_range:.4f}")
    else:
        first_batch = next(iter(loader))
        data_range = float(compute_latent_data_range(first_batch["mu"]).item())
        print(f"[train] latent data_range (p99-p1) = {data_range:.4f}")

    # Model, loss, optimizer
    model = build_model(cfg).to(device)
    loss_fn = ReconLoss(
        l1_weight=float(cfg.loss.l1_weight),
        l2_weight=float(getattr(cfg.loss, "l2_weight", 0.0)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    print(f"[train] exp={exp_name}  model={cfg.model.kind}  device={device}")
    print(
        f"[train] batches={len(loader)}  batch_size={batch_size}  epochs={cfg.train.epochs}"
    )

    notifier.notify_training_start(
        exp_name=exp_name,
        model_kind=str(cfg.model.kind),
        batch_size=batch_size,
        epochs=int(cfg.train.epochs),
        n_batches_per_epoch=len(loader),
        wandb_url=wandb_url,
        extras={
            "lr": str(cfg.train.lr),
            "amp": str(getattr(cfg.train, "amp", "none")),
            "device": str(device),
            "resumed": "yes" if resume_state is not None else "no",
        },
    )

    # ---- Restore model / optimizer / RNG / counters from snapshot.
    start_epoch = 1
    start_completed_in_epoch = 0
    global_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["opt"])
        # Move optimizer tensors onto the training device.
        for st in opt.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        if resume_state.get("rng") is not None:
            _restore_rng_state(resume_state["rng"])
        saved_epoch = int(resume_state["epoch"])
        completed = int(resume_state.get("completed_steps_in_epoch", 0))
        epoch_done = bool(resume_state.get("epoch_done", False))
        global_step = int(resume_state.get("global_step", 0))
        if epoch_done:
            start_epoch = saved_epoch + 1
            start_completed_in_epoch = 0
        else:
            start_epoch = saved_epoch
            start_completed_in_epoch = completed
        if start_epoch > int(cfg.train.epochs):
            print(
                f"[resume] saved epoch {saved_epoch} already at/past final "
                f"epoch {cfg.train.epochs} — nothing to do."
            )
            if use_wandb:
                wandb.finish()
            return
        print(
            f"[resume] continuing at epoch={start_epoch} "
            f"skip_first_steps_in_epoch={start_completed_in_epoch} "
            f"global_step={global_step}"
        )

    # Build MAISI decoder once if image-metric validation is enabled.
    maisi_net = None
    maisi_decoder = None
    if image_metric_every > 0:
        print("[eval] Loading MAISI decoder for image-domain validation...")
        try:
            maisi_net = _load_maisi_decoder(device)
            if bool(getattr(cfg.eval, "compile_maisi", True)):
                maisi_decoder = torch.compile(
                    maisi_net.decode_stage_2_outputs, mode="reduce-overhead"
                )
                print("[eval] Warming up torch.compile (3 passes, ~30-60s)...")
                _warmup_maisi_decoder(maisi_decoder, device)
            else:
                maisi_decoder = maisi_net.decode_stage_2_outputs
                print("[eval] torch.compile disabled (eval.compile_maisi=false).")
            print("[eval] MAISI decoder ready.")
        except Exception as e:
            print(f"[eval] Failed to load MAISI decoder: {e} — image metrics disabled.")
            maisi_decoder = None

    fast_n_samples = int(getattr(cfg.eval, "fast_n_samples", 200))
    image_metric_n_samples = int(getattr(cfg.eval, "image_metric_n_samples", 50))
    log_every = int(getattr(cfg.train, "log_every", 50))
    save_every_n_steps = int(getattr(cfg.train, "save_every_n_steps", 500))

    amp_mode = str(getattr(cfg.train, "amp", "none")).lower()
    use_bf16 = amp_mode == "bf16"
    if use_bf16:
        print("[train] AMP mode: bfloat16 autocast (no GradScaler needed)")

    # ---- Best-checkpoint tracker. Higher metric = better (PSNR-style).
    # Configurable via cfg.train.best_metric; defaults to "latent_psnr" (always
    # available). Set to "image_psnr_3d" to track the image-domain metric (only
    # updates on epochs where image validation runs).
    best_metric_name = str(getattr(cfg.train, "best_metric", "latent_psnr"))
    best_metric_value = float("-inf")
    if best_path.exists():
        try:
            _b_state = torch.load(best_path, map_location="cpu", weights_only=False)
            saved_name = _b_state.get("best_metric_name")
            saved_value = _b_state.get("best_metric_value")
            if saved_name == best_metric_name and saved_value is not None:
                best_metric_value = float(saved_value)
                print(
                    f"[best] existing best.pt: "
                    f"{best_metric_name}={best_metric_value:.4f}"
                )
            else:
                print(
                    f"[best] existing best.pt tracks {saved_name!r}, but config "
                    f"requests {best_metric_name!r} — will overwrite on next improvement"
                )
        except Exception as e:
            print(f"[best] WARN: could not read existing best.pt ({e})")

    # Wall-clock cap: stop training once `cfg.train.max_wall_minutes` of training
    # time has elapsed (excludes setup/warmup). 0 or unset = no cap.
    max_wall_minutes = float(getattr(cfg.train, "max_wall_minutes", 0) or 0)
    max_wall_seconds = max_wall_minutes * 60.0 if max_wall_minutes > 0 else 0.0
    train_start_time = time.time()
    wall_capped = False
    if max_wall_seconds > 0:
        print(f"[train] wall-clock cap: {max_wall_minutes:.1f} min")

    last_metrics: dict = {}
    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        model.train()
        last_total = float("nan")
        skip_in_this_epoch = start_completed_in_epoch if epoch == start_epoch else 0
        if skip_in_this_epoch > 0:
            print(
                f"[resume] epoch {epoch}: skipping first {skip_in_this_epoch} batches"
            )
        for step, batch in enumerate(loader):
            if step < skip_in_this_epoch:
                continue
            if (
                max_wall_seconds > 0
                and (time.time() - train_start_time) > max_wall_seconds
            ):
                elapsed = time.time() - train_start_time
                print(
                    f"[wall-clock cap] {elapsed / 60:.1f} min elapsed > "
                    f"{max_wall_minutes:.1f} min budget; stopping at "
                    f"epoch {epoch} step {step} (global_step={global_step})"
                )
                wall_capped = True
                break
            mu = batch["mu"].to(device)

            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    mu_hat, _triplane = model(mu)
                    losses = loss_fn(mu_hat, mu)
            else:
                mu_hat, _triplane = model(mu)
                losses = loss_fn(mu_hat, mu)
            losses["total"].backward()
            opt.step()

            log = {
                "loss/total": losses["total"].item(),
                "loss/l1": losses["l1"].item(),
                "loss/l2": losses["l2"].item(),
                "step": global_step,
            }
            last_total = log["loss/total"]

            if not math.isfinite(last_total):
                notifier.notify_nan(
                    exp_name=exp_name,
                    epoch=epoch,
                    step=global_step,
                    loss_value=last_total,
                    wandb_url=wandb_url,
                )
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch} step {global_step}: {last_total}"
                )

            if step % log_every == 0:
                with torch.no_grad():
                    psnr = float(latent_psnr(mu_hat, mu, data_range=data_range).mean())
                    l1_metric = float(latent_l1(mu_hat, mu).mean())
                    cos = float(latent_cosine_similarity(mu_hat, mu).mean())
                log["metrics/psnr"] = psnr
                log["metrics/l1"] = l1_metric
                log["metrics/cosine"] = cos
                last_metrics = {"psnr": psnr, "l1": l1_metric, "cosine": cos}
                print(
                    f"[epoch {epoch} step {step:03d}]  "
                    f"loss={log['loss/total']:.2e}  "
                    f"l1={log['loss/l1']:.2e}  "
                    f"l2={log['loss/l2']:.2e}  "
                    f"psnr={psnr:.1f}  "
                    f"cos={cos:.4f}"
                )

            if use_wandb:
                wandb.log(log, step=global_step)

            global_step += 1

            # Periodic snapshot — overwrite latest.pt every N global steps.
            if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                _save_snapshot(
                    latest_path,
                    model=model,
                    opt=opt,
                    epoch=epoch,
                    global_step=global_step,
                    completed_steps_in_epoch=step + 1,
                    epoch_done=False,
                    data_range=data_range,
                    cfg=cfg,
                    wandb_run_id=wandb_run_id,
                )

        # Wall-clock cap: save mid-epoch snapshot, run one cheap val, then exit.
        if wall_capped:
            _save_snapshot(
                latest_path,
                model=model,
                opt=opt,
                epoch=epoch,
                global_step=global_step,
                completed_steps_in_epoch=step,
                epoch_done=False,
                data_range=data_range,
                cfg=cfg,
                wandb_run_id=wandb_run_id,
            )
            print(f"[wall-clock cap] mid-epoch snapshot -> {latest_path}")
            from src.evaluation.validate import run_validation

            sw_batch_size_capped = int(cfg.eval.sw_batch_size)
            with torch.no_grad():
                val_metrics = run_validation(
                    model=model,
                    maisi_decoder=None,
                    val_loader=val_loader,
                    n_samples=fast_n_samples,
                    device=str(device),
                    compute_image_metrics=False,
                    sw_batch_size=sw_batch_size_capped,
                )
            val_log = {f"val/{k}": v for k, v in val_metrics.items() if v is not None}
            if use_wandb:
                wandb.log(val_log, step=global_step)
            print(
                "[val wall-capped]  "
                + "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in val_log.items()
                    if isinstance(v, float)
                )
            )
            break

        # End-of-epoch print uses the most recent heavy-metric snapshot.
        psnr_str = f"{last_metrics['psnr']:.2f}" if "psnr" in last_metrics else "n/a"
        cos_str = f"{last_metrics['cosine']:.4f}" if "cosine" in last_metrics else "n/a"
        print(
            f"[epoch {epoch} DONE]  "
            f"final_loss={last_total:.2e}  "
            f"psnr={psnr_str}  "
            f"cos={cos_str}"
        )

        # End-of-epoch checkpoints: per-epoch (kept for history) + latest (atomic).
        epoch_ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        steps_this_epoch = len(loader)
        for p in (epoch_ckpt_path, latest_path):
            _save_snapshot(
                p,
                model=model,
                opt=opt,
                epoch=epoch,
                global_step=global_step,
                completed_steps_in_epoch=steps_this_epoch,
                epoch_done=True,
                data_range=data_range,
                cfg=cfg,
                wandb_run_id=wandb_run_id,
            )
        print(f"[checkpoint] {epoch_ckpt_path}")

        # --- Validation ---
        from src.evaluation.validate import run_validation

        sw_batch_size = int(cfg.eval.sw_batch_size)

        # Cheap latent-only validation every epoch.
        with torch.no_grad():
            val_metrics = run_validation(
                model=model,
                maisi_decoder=None,
                val_loader=val_loader,
                n_samples=fast_n_samples,
                device=str(device),
                compute_image_metrics=False,
                sw_batch_size=sw_batch_size,
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
                    sw_batch_size=sw_batch_size,
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

        # Track best checkpoint when the configured metric improved.
        cur = val_metrics.get(best_metric_name)
        if cur is not None and float(cur) > best_metric_value:
            prev_value = best_metric_value
            prev_str = (
                f"{best_metric_value:.4f}"
                if best_metric_value != float("-inf")
                else "n/a"
            )
            best_metric_value = float(cur)
            _save_snapshot(
                best_path,
                model=model,
                opt=opt,
                epoch=epoch,
                global_step=global_step,
                completed_steps_in_epoch=steps_this_epoch,
                epoch_done=True,
                data_range=data_range,
                cfg=cfg,
                wandb_run_id=wandb_run_id,
                extra={
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                },
            )
            print(
                f"[best] new best {best_metric_name}={best_metric_value:.4f} "
                f"(prev={prev_str}) -> {best_path}"
            )
            notifier.notify_new_best(
                exp_name=exp_name,
                metric_name=best_metric_name,
                metric_value=best_metric_value,
                prev_value=prev_value,
                epoch=epoch,
                wandb_url=wandb_url,
            )

    final_metrics: dict = {}
    if "psnr" in last_metrics:
        final_metrics["latent_psnr"] = float(last_metrics["psnr"])
    if "cosine" in last_metrics:
        final_metrics["latent_cosine"] = float(last_metrics["cosine"])
    final_metrics["final_loss"] = (
        float(last_total) if math.isfinite(last_total) else last_total
    )

    notifier.notify_training_done(
        exp_name=exp_name,
        final_metrics=final_metrics,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value
        if best_metric_value != float("-inf")
        else None,
        wandb_url=wandb_url,
    )

    if use_wandb:
        wandb.finish()
        print(f"[wandb] run URL: {run.url}")
    else:
        print("[wandb] offline — no URL available")


if __name__ == "__main__":
    main()
