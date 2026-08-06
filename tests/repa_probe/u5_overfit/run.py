"""U5/S4 — 정렬이 **실제 대응**에서 오는지: 소규모 과적합 + 대조군 2개.

계획의 sanity-check 층 S4. 손실이 내려가고 cosine이 올라가는 것만으로는 부족하다 — projector가
아무 타깃에나 맞추고 있을 수도 있기 때문이다. 그래서 teacher를 망가뜨린 대조군을 같이 돌린다:

    real       볼륨과 짝이 맞는 teacher                     → cos가 올라가야 한다
    shuffled   **매 스텝** 배치 안에서 teacher를 뒤섞음      → 올라가면 안 된다
    gaussian   teacher를 같은 모먼트의 가우시안 노이즈로 대체 → 0 근처에 머물러야 한다

shuffle은 고정 순열이 아니라 매 스텝 새로 뽑는다. 고정이면 8개짜리 소규모에서 모델이 "볼륨 i →
teacher π(i)" 매핑 자체를 외워버려 대조군이 무력해진다.

실행:
    CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u5_overfit.run [--steps 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.data.report2ct_wan_repa_datamodule import Report2CTWanRepaDataModule
from src.models.components.repa import RepaAligner
from src.models.report2ct_module import Report2CTModule

DATALIST = Path("/workspace/data/report2ct_wan/datalist_wan_2560.json")
TEACHER_DIR = Path("/workspace/data/report2ct_wan/spectre_ssl_16")
TMP_DATALIST = Path("/workspace/data/report2ct_wan/_u5_overfit_datalist.json")
OUT_DIR = Path(__file__).parent
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figs"
TEACHER_GRID = (16, 16, 16)


def build_module(seed: int) -> Report2CTModule:
    """configs/model/report2ct_wan_repa.yaml과 동일한 UNet + aligner 설정."""
    torch.manual_seed(seed)
    unet = DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=16,
        out_channels=16,
        num_channels=[64, 128, 256, 512],
        num_res_blocks=2,
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        cross_attention_dim=2560,
        num_class_embeds=128,
        include_fc=True,
        include_spacing_input=True,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        resblock_updown=True,
        with_conditioning=True,
        use_flash_attention=False,
    )
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        scale=1.4,
        use_timestep_transform=True,
        use_discrete_timesteps=False,
        sample_method="uniform",
    )
    repa = RepaAligner(
        student_channels=512,
        teacher_dim=1080,
        student_grid=(8, 8, 8),
        teacher_grid=TEACHER_GRID,
        crop_tokens=(4, 4, 4),
        projector="mlp",
    )
    module = Report2CTModule(
        unet=unet, noise_scheduler=scheduler, repa=repa, repa_weight=0.5, lr=2e-4
    )
    module._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=1000, global_step=0
    )
    module.log = lambda *a, **k: None
    module.setup("fit")
    return module.cuda().train()


def corrupt(teacher: torch.Tensor, arm: str, gen: torch.Generator) -> torch.Tensor:
    """대조군용 teacher 변형. `real`은 그대로 통과."""
    if arm == "real":
        return teacher
    if arm == "shuffled":  # 매 스텝 새 순열 — 외울 수 있는 매핑을 남기지 않는다
        perm = torch.randperm(teacher.shape[0], generator=gen, device=teacher.device)
        return teacher[perm]
    if arm == "gaussian":  # 같은 평균/표준편차의 노이즈 — 구조만 지운다
        return (
            torch.randn(teacher.shape, generator=gen, device=teacher.device)
            * teacher.std()
            + teacher.mean()
        )
    raise ValueError(arm)


def run_arm(arm: str, batches: list[dict], steps: int, seed: int = 0) -> list[dict]:
    """한 arm을 `steps`만큼 과적합시키며 정렬 지표를 기록한다."""
    module = build_module(seed)
    opt = torch.optim.Adam(
        list(module.unet.parameters()) + list(module.repa.parameters()), lr=2e-4
    )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.manual_seed(seed)

    history = []
    for step in range(steps):
        batch = dict(batches[step % len(batches)])
        batch["spectre"] = corrupt(batch["spectre"], arm, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            module._shared_forward(batch)
        repa_loss, logs = module._last_repa
        total = repa_loss  # 정렬 항만 학습 — diffusion 손실과 섞이지 않게 격리
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "cos": float(logs["repa_cos"]),
                    "rel": float(logs["repa_rel"]),
                }
            )
    del module, opt
    torch.cuda.empty_cache()
    return history


def load_batches(n_volumes: int, batch_size: int) -> list[dict]:
    """teacher가 준비된 valid 스캔에서 소규모 배치를 만든다."""
    entries = json.loads(DATALIST.read_text())["validation"]
    ready = [
        e
        for e in entries
        if (
            TEACHER_DIR / f"{Path(e['image']).name[: -len('_emb.nii.gz')]}.npy"
        ).is_file()
    ][:n_volumes]
    if len(ready) < n_volumes:
        raise SystemExit(f"teacher feature가 {len(ready)}개뿐 — precompute 진행 중")
    TMP_DATALIST.write_text(json.dumps({"training": ready, "validation": ready}))

    dm = Report2CTWanRepaDataModule(
        datalist_path=str(TMP_DATALIST),
        batch_size=batch_size,
        num_workers=4,
        cache_rate=1.0,
        spacing_multiplier=100.0,
        spectre_dir=str(TEACHER_DIR),
    )
    dm.setup("fit")
    batches = [
        {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        for b in dm.train_dataloader()
    ]
    TMP_DATALIST.unlink(missing_ok=True)
    return batches


def plot(histories: dict[str, list[dict]]) -> str:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, hist in histories.items():
        x = [h["step"] for h in hist]
        axes[0].plot(x, [h["cos"] for h in hist], label=arm)
        axes[1].plot(x, [h["rel"] for h in hist], label=arm)
    axes[0].set_title("token-wise cosine (higher = better aligned)")
    axes[1].set_title("relational loss (lower = better aligned)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("S4 - 8-volume overfit: real teacher vs controls", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "S4_overfit_controls.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out.relative_to(Path("/workspace")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-volumes", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    batches = load_batches(args.n_volumes, args.batch_size)
    print(f"{args.n_volumes} volumes → {len(batches)} batches of {args.batch_size}")

    histories = {}
    for arm in ("real", "shuffled", "gaussian"):
        histories[arm] = run_arm(arm, batches, args.steps)
        first, last = histories[arm][0], histories[arm][-1]
        print(
            f"  {arm:9s} cos {first['cos']:+.4f} → {last['cos']:+.4f}   "
            f"rel {first['rel']:.4f} → {last['rel']:.4f}",
            flush=True,
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "overfit_controls.json"
    out.write_text(
        json.dumps(
            {
                "steps": args.steps,
                "n_volumes": args.n_volumes,
                "batch_size": args.batch_size,
                "figure": plot(histories),
                "histories": histories,
            },
            indent=2,
        )
    )
    real, shuf = histories["real"][-1]["cos"], histories["shuffled"][-1]["cos"]
    print(f"\n판정: real {real:+.4f} vs shuffled {shuf:+.4f} — 격차 {real - shuf:+.4f}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
