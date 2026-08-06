"""마스크 VAE 학습 루프 — NV `train_vae`를 마스크용으로 어댑트.

NV MAISI VAE 레시피에서 **마스크용으로 적응**:
  - recon: L1 → **DiceCELoss**(softmax + to_onehot_y, include_background=False)
  - perceptual(0.3)·adversarial(0.1) **제거** (라벨맵에 무의미)
  - KL 1e-6, Adam, bf16 autocast, LR warmup, grad clip

입력 표현 3종(`--input-mode`): raw(1ch, label/117) / onehot(118ch) / embed(learnable d-ch).
재구성 타깃은 정수 라벨(118 클래스).

산출물(`tests/mask_vae/results/<exp>/`): metrics.json, loss_curve.png, best.pt/last.pt, vis/(GT vs recon).
MAISI progressive(64³→128³): `--resume <64³ best.pt>` 로 이어학습.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from torch.amp import autocast

from tests.mask_vae.dataset import build_dataloader
from tests.mask_vae.model import build_mask_vae
from tests.mask_vae.preprocess import NUM_CLASSES, PATCH_SIZE, to_vae_input
from tests.mask_vae.viz import save_recon_vis

RESULTS_ROOT: Path = Path(__file__).resolve().parent / "results"


def kl_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """표준 VAE KL — q(z)=N(z_mu, z_sigma²) 대 N(0,1). z_sigma는 std (MONAI AutoencoderKL 규약).

    Args:
        z_mu: ``(B, 4, ·)`` latent 평균.
        z_sigma: ``(B, 4, ·)`` latent 표준편차(>0).

    Returns:
        배치 평균 스칼라 KL.
    """
    eps = 1e-10
    per = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, z_mu.ndim)),
    )
    return per.mean()


@torch.no_grad()
def foreground_dice(recon_logits: torch.Tensor, target: torch.Tensor) -> float:
    """argmax 복원 vs 정답의 **전경(배경 0 제외) 마이크로 Dice**. overfit/일반화 게이트용.

    Args:
        recon_logits: ``(B, 118, ·)`` 디코더 logit.
        target: ``(B, 1, ·)`` 정수 라벨.

    Returns:
        전경 voxel 기준 micro-Dice ``[0,1]`` (float).
    """
    pred = recon_logits.argmax(1, keepdim=True)  # (B,1,·)
    fg = target != 0
    inter = ((pred == target) & fg).sum().float()
    denom = fg.sum().float() + (pred != 0).sum().float()
    return float((2 * inter / denom.clamp_min(1.0)).item())


@torch.no_grad()
def eval_fixed(ae: torch.nn.Module, eval_mask: torch.Tensor, prep) -> float:
    """고정 패치 배치를 재구성해 foreground Dice를 반환(노이즈 없는 지표).

    Args:
        ae: 마스크 VAE.
        eval_mask: ``(N,1,·)`` 고정 정수 라벨.
        prep: 정수 라벨 → VAE 입력 변환 함수(embed면 현재 임베딩 반영).

    Returns:
        전경 micro-Dice ``[0,1]``.
    """
    ae.eval()
    x = prep(eval_mask)
    with autocast("cuda", dtype=torch.bfloat16):
        recon, _, _ = ae(x)
    fg = foreground_dice(recon.float(), eval_mask)
    ae.train()
    return fg


def _plot_curves(history: list[dict], out_path: Path, steps_per_epoch: int) -> None:
    """loss(좌축) + train_fg/val_fg(우축) 곡선을 저장. x축은 epoch 단위."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spe = max(1, steps_per_epoch)
    epochs = [h["step"] / spe for h in history]  # step → epoch
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        epochs, [h["loss"] for h in history], color="tab:red", alpha=0.6, label="loss"
    )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(
        epochs, [h["train_fg"] for h in history], color="tab:blue", label="train_fg"
    )
    vs = [h["step"] / spe for h in history if h["val_fg"] == h["val_fg"]]  # not nan
    vf = [h["val_fg"] for h in history if h["val_fg"] == h["val_fg"]]
    ax2.plot(vs, vf, color="tab:green", marker="o", ms=3, label="val_fg")
    ax2.set_ylabel("fg_dice")
    ax2.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def train(
    n_masks: int | None,
    steps: int,
    exp_name: str,
    batch_size: int = 2,
    lr: float = 1e-4,
    kl_weight: float = 1e-7,
    num_workers: int = 4,
    device: str = "cuda:0",
    log_every: int = 50,
    cache_rate: float = 1.0,
    clip_grad: float = 10.0,
    warmup_steps: int = 150,
    input_mode: str = "embed",
    embed_dim: int = 16,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    val_limit: int = 64,
    val_every: int = 100,
    vis_samples: int = 6,
    vis_seed: int = 0,
    resume: str | None = None,
) -> torch.nn.Module:
    """마스크 VAE를 학습하고 가중치·metrics·loss curve·recon 시각화를 저장한다.

    Args:
        n_masks: 학습에 쓸 train_fixed 마스크 수 (None=전체).
        steps: 학습 스텝 수.
        exp_name: 결과 폴더명 → ``tests/mask_vae/results/<exp_name>/``.
        input_mode: ``raw`` | ``onehot`` | ``embed``.
        val_limit: held-out valid_fixed 마스크 수(일반화 평가·시각화 대상).
        val_every: 검증 간격(스텝).
        vis_samples / vis_seed: 재구성 시각화 샘플 수 / 랜덤 시드.
        resume: 이어학습할 AE 가중치 경로(64³→128³ progressive).

    Returns:
        학습된 ``AutoencoderKlMaisi``.
    """
    dev = torch.device(device)
    dl = build_dataloader(
        "train_fixed",
        is_train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=n_masks,
        cache_rate=cache_rate,
        patch_size=patch_size,
    )

    # 입력 표현 모드: raw(1ch, label/117) / onehot(118ch) / embed(learnable d-ch).
    if input_mode == "raw":
        in_channels, embed_layer = 1, None
    elif input_mode == "onehot":
        in_channels, embed_layer = NUM_CLASSES, None
    elif input_mode == "embed":
        in_channels = embed_dim
        embed_layer = torch.nn.Embedding(NUM_CLASSES, embed_dim).to(
            dev
        )  # 라벨→d 벡터(공동학습)
    else:
        raise ValueError(f"input_mode must be raw|onehot|embed, got {input_mode}")

    def prep(mask: torch.Tensor) -> torch.Tensor:
        """정수 라벨 ``(B,1,·)`` → VAE 입력 ``(B,in_channels,·)`` (모드별)."""
        if input_mode == "raw":
            return to_vae_input(mask)  # (B,1,·) [0,1]
        idx = mask.long().squeeze(1)  # (B,·)
        if input_mode == "onehot":
            return (
                torch.nn.functional.one_hot(idx, NUM_CLASSES).movedim(-1, 1).float()
            )  # (B,118,·)
        return embed_layer(idx).movedim(-1, 1)  # (B,d,·)

    ae = build_mask_vae(in_channels=in_channels, norm_float16=False).to(
        dev
    )  # bf16 위해 fp16 groupnorm 끔
    if resume:  # MAISI progressive: 64³ 가중치로 128³ 이어학습 (ae + embed 모두 복원)
        ckpt = torch.load(resume, map_location=dev)
        ae.load_state_dict(ckpt["ae"])
        if embed_layer is not None and ckpt.get("embed") is not None:
            embed_layer.load_state_dict(ckpt["embed"])
        print(f"resumed from {resume}", flush=True)

    def _ckpt() -> dict:
        """체크포인트 = ae + embed(있으면) + 입력 설정. precompute/resume가 입력 매핑까지 복원."""
        return {
            "ae": ae.state_dict(),
            "embed": embed_layer.state_dict() if embed_layer is not None else None,
            "input_mode": input_mode,
            "embed_dim": embed_dim,
        }

    params = list(ae.parameters()) + (
        list(embed_layer.parameters()) if embed_layer else []
    )
    opt = torch.optim.Adam(params, lr=lr)
    # LR warmup (MAISI 관습): 초기 저속 램프업으로 초반 불안정 억제. 0.02→1.0 선형, 이후 유지.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, 0.02 + 0.98 * s / max(1, warmup_steps))
    )
    dice_ce = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False)

    def _fixed_batch(loader, n: int) -> torch.Tensor:
        """loader에서 n개 패치를 모아 고정 배치로. MetaTensor→plain."""
        chunks: list[torch.Tensor] = []
        t = iter(loader)
        while sum(c.shape[0] for c in chunks) < n:
            chunks.append(next(t)["mask"].as_tensor())
        return torch.cat(chunks)[:n].to(dev)

    n_fixed = max(8, vis_samples)
    eval_mask = _fixed_batch(
        dl, min(n_fixed, n_masks or n_fixed)
    )  # 학습 패치(overfit 신호)
    # held-out valid_fixed 고정 배치 — 일반화 신호 + 시각화 대상
    val_dl = build_dataloader(
        "valid_fixed",
        is_train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=val_limit,
        cache_rate=1.0,
        patch_size=patch_size,
    )
    val_mask = _fixed_batch(val_dl, min(n_fixed, val_limit))

    results_dir = RESULTS_ROOT / exp_name
    (results_dir / "vis").mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = -1.0

    it = iter(dl)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        mask = batch["mask"].to(dev)  # (B,1,·) long {0..117}
        x = prep(mask)  # (B,in_channels,·)

        opt.zero_grad(set_to_none=True)
        # bf16 autocast: fp16보다 동적 범위가 넓어 오버플로가 없어 GradScaler가 불필요.
        with autocast("cuda", dtype=torch.bfloat16):
            recon, z_mu, z_sigma = ae(x)  # recon (B,118,·); z_* (B,4,·/4)
        recon = recon.float()  # loss는 fp32에서 (softmax 안정)
        recon_l = dice_ce(recon, mask)  # to_onehot_y가 mask를 118로 확장
        kl = kl_loss(z_mu.float(), z_sigma.float())
        loss = recon_l + kl_weight * kl

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            params, clip_grad
        )  # ae+embed 모두; 클립 전 총 norm 반환
        opt.step()
        scheduler.step()  # warmup 진행

        if step % log_every == 0 or step == 1:
            tr_fg = eval_fixed(ae, eval_mask, prep)  # 학습 패치 (overfit 신호)
            do_val = (step % val_every == 0) or step == 1 or step == steps
            val_fg = eval_fixed(ae, val_mask, prep) if do_val else float("nan")
            cur_lr = scheduler.get_last_lr()[0]
            history.append(
                {
                    "step": step,
                    "lr": cur_lr,
                    "loss": loss.item(),
                    "dice_ce": recon_l.item(),
                    "kl": kl.item(),
                    "grad_norm": float(gnorm),
                    "train_fg": tr_fg,
                    "val_fg": val_fg,
                }
            )
            print(
                f"[{input_mode}:{exp_name}] step {step:4d} | lr {cur_lr:.2e} | loss {loss.item():.4f} "
                f"(dice_ce {recon_l.item():.4f}, kl {kl.item():.1f}) | grad_norm {float(gnorm):.2f} "
                f"| train_fg {tr_fg:.3f} | val_fg {val_fg:.3f}",
                flush=True,
            )
            if do_val and val_fg > best_val:  # 일반화 기준 best 저장
                best_val = val_fg
                torch.save(_ckpt(), str(results_dir / "best.pt"))

    # 최종 산출물 (전부 results/<exp>/ 아래)
    torch.save(_ckpt(), str(results_dir / "last.pt"))
    (results_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    _plot_curves(history, results_dir / "loss_curve.png", steps_per_epoch=len(dl))
    n_vis = save_recon_vis(
        ae, prep, val_mask, results_dir / "vis", vis_samples, vis_seed
    )
    print(
        f"results: {results_dir} (best_val_fg {best_val:.3f}, {n_vis} vis imgs)",
        flush=True,
    )
    return ae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-name", type=str, required=True, help="결과 폴더명 results/<exp>"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="train_fixed 마스크 수 (None=전체)"
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="MAISI 기본")
    ap.add_argument(
        "--kl-weight", type=float, default=1e-7, help="KL 가중치(MAISI 기본)"
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--cache-rate",
        type=float,
        default=1.0,
        help="CacheDataset 비율(전체학습은 낮게)",
    )
    ap.add_argument(
        "--clip-grad", type=float, default=10.0, help="grad norm 상한(스파이크만)"
    )
    ap.add_argument("--warmup-steps", type=int, default=150, help="LR warmup 스텝 수")
    ap.add_argument(
        "--input-mode",
        type=str,
        default="embed",
        choices=["raw", "onehot", "embed"],
        help="라벨→VAE 입력 표현",
    )
    ap.add_argument("--embed-dim", type=int, default=16, help="embed 모드 임베딩 차원")
    ap.add_argument("--log-every", type=int, default=50, help="로그 간격(스텝)")
    ap.add_argument(
        "--patch", type=int, default=64, help="cubic 패치 한 변 (64/128 등)"
    )
    ap.add_argument(
        "--val-limit", type=int, default=64, help="held-out valid 마스크 수"
    )
    ap.add_argument("--val-every", type=int, default=100, help="검증 간격(스텝)")
    ap.add_argument("--vis-samples", type=int, default=6, help="recon 시각화 샘플 수")
    ap.add_argument("--vis-seed", type=int, default=0, help="시각화 랜덤 시드")
    ap.add_argument(
        "--resume", type=str, default=None, help="이어학습 AE 가중치(64³→128³)"
    )
    args = ap.parse_args()
    train(
        n_masks=args.limit,
        steps=args.steps,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_weight=args.kl_weight,
        num_workers=args.num_workers,
        device=args.device,
        log_every=args.log_every,
        cache_rate=args.cache_rate,
        clip_grad=args.clip_grad,
        warmup_steps=args.warmup_steps,
        input_mode=args.input_mode,
        embed_dim=args.embed_dim,
        patch_size=(args.patch, args.patch, args.patch),
        val_limit=args.val_limit,
        val_every=args.val_every,
        vis_samples=args.vis_samples,
        vis_seed=args.vis_seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
