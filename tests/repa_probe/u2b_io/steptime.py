"""U2b — REPA를 켰을 때 학습 스텝이 얼마나 느려지는지 실측 (계획의 sanity-check 층 **S6**).

두 가지를 가른다:
  1. teacher feature I/O — 16³(8.85 MB/scan)와 32³(70.8 MB/scan) 중 무엇을 기본 arm으로 둘지
  2. 정렬 손실 자체의 연산 오버헤드 — 목표는 **< 25 %**

precompute가 아직 진행 중이므로, **이미 완성된 valid 스캔들**로 임시 datalist를 만들어 잰다.
파일 크기와 파일시스템이 동일하므로 수치는 train 분할에 그대로 전이된다.

실행:
    CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u2b_io.steptime [--steps 60]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from src.baselines.rflow import RFlowScheduler
from src.data.report2ct_datamodule import Report2CTDataModule
from src.data.report2ct_wan_repa_datamodule import Report2CTWanRepaDataModule
from src.models.components.repa import RepaAligner
from src.models.report2ct_module import Report2CTModule

DATALIST = Path("/workspace/data/report2ct_wan/datalist_wan_2560.json")
TEACHER_ROOT = Path("/workspace/data/report2ct_wan")
TMP_DATALIST = Path("/workspace/data/report2ct_wan/_u2b_steptime_datalist.json")
RESULT_DIR = Path(__file__).parent / "results"


def build_tmp_datalist(grid_n: int) -> int:
    """teacher feature가 이미 있는 스캔만 모아 임시 datalist를 쓴다. 반환값은 스캔 수."""
    entries = json.loads(DATALIST.read_text())["validation"]
    teacher_dir = TEACHER_ROOT / f"spectre_ssl_{grid_n}"
    ready = [
        e
        for e in entries
        if (
            teacher_dir / f"{Path(e['image']).name[: -len('_emb.nii.gz')]}.npy"
        ).is_file()
    ]
    TMP_DATALIST.write_text(json.dumps({"training": ready, "validation": ready[:8]}))
    return len(ready)


def build_module(grid_n: int | None) -> Report2CTModule:
    """REPA 유무만 다른 학습 모듈. UNet/스케줄러 kwargs는 configs/model/report2ct_wan.yaml 그대로."""
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (  # noqa: PLC0415
        DiffusionModelUNetMaisi,
    )

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
    repa = (
        None
        if grid_n is None
        else RepaAligner(
            student_channels=512,
            teacher_dim=1080,
            student_grid=(8, 8, 8),
            teacher_grid=(grid_n,) * 3,
            crop_tokens=(grid_n // 4,) * 3,
            projector="mlp",
        )
    )
    module = Report2CTModule(unet=unet, noise_scheduler=scheduler, repa=repa, lr=2e-4)
    module.log = lambda *a, **k: None
    return module


def time_arm(grid_n: int | None, steps: int, batch_size: int, num_workers: int) -> dict:
    """dataloader + forward + backward를 실제로 돌려 스텝 시간을 잰다."""
    common = dict(
        datalist_path=str(TMP_DATALIST),
        batch_size=batch_size,
        num_workers=num_workers,
        cache_rate=0.0,
        spacing_multiplier=100.0,
    )
    dm = (
        Report2CTDataModule(**common)
        if grid_n is None
        else Report2CTWanRepaDataModule(
            **common, spectre_dir=str(TEACHER_ROOT / f"spectre_ssl_{grid_n}")
        )
    )
    dm.setup("fit")
    loader = dm.train_dataloader()

    module = build_module(grid_n).cuda().train()
    # global_step만 필요하다 (HASTE 종료 판정). 실제 Trainer는 이 측정에 불필요.
    module._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=1000, global_step=0
    )
    opt = torch.optim.Adam(
        list(module.unet.parameters())
        + (list(module.repa.parameters()) if module.repa is not None else []),
        lr=2e-4,
    )
    module.setup("fit")

    warmup, timed, data_wait = 5, [], []
    it = iter(loader)
    t_prev = time.perf_counter()
    for i in range(steps + warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t_data = time.perf_counter()
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = module.training_step(batch, i)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t_now = time.perf_counter()
        if i >= warmup:
            timed.append(t_now - t_prev)
            data_wait.append(t_data - t_prev)
        t_prev = t_now

    mb_per_step = 0.0 if grid_n is None else batch_size * (grid_n**3) * 1080 * 2 / 1e6
    return {
        "arm": "baseline (no REPA)" if grid_n is None else f"REPA {grid_n}³",
        "step_s": round(sum(timed) / len(timed), 4),
        "data_wait_s": round(sum(data_wait) / len(data_wait), 4),
        "teacher_MB_per_step": round(mb_per_step, 1),
        "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=32)
    args = ap.parse_args()

    n_ready = build_tmp_datalist(32)
    print(f"{n_ready} scans with teacher features (both grids)")

    rows = []
    for grid_n in (None, 16, 32):
        torch.cuda.reset_peak_memory_stats()
        row = time_arm(grid_n, args.steps, args.batch_size, args.num_workers)
        rows.append(row)
        print(row, flush=True)
        torch.cuda.empty_cache()

    base = rows[0]["step_s"]
    for row in rows:
        row["overhead_pct"] = round(100 * (row["step_s"] / base - 1), 1)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "steptime.json"
    out.write_text(
        json.dumps(
            {
                "n_scans": n_ready,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "rows": rows,
            },
            indent=2,
        )
    )
    TMP_DATALIST.unlink(missing_ok=True)
    print(
        f"\n{'arm':22s} {'step_s':>8s} {'data_wait':>10s} {'MB/step':>9s} {'overhead':>9s}"
    )
    for row in rows:
        print(
            f"{row['arm']:22s} {row['step_s']:8.4f} {row['data_wait_s']:10.4f} "
            f"{row['teacher_MB_per_step']:9.1f} {row['overhead_pct']:8.1f}%"
        )
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
