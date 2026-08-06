"""U4b — U4의 두 기본값(projector=mlp, teacher grid=16³)을 **수렴까지** 다시 검증한다.

U4는 1500 스텝에서 끊었고 거기서 `mlp > conv`, `16³ > 32³`라는 기본값을 정했다. 그런데
300 → 1500 구간에서 **모든 arm이 +0.013~0.026씩 나란히** 올랐다 — 어느 곳에도 평탄부가 없었다.
순위가 유지된 건 "수렴 후에도 유지된다"가 아니라 "둘 다 미수렴 지점이었다"에 가깝고, 실제로
conv가 mlp보다 **더 빠르게** 오르고 있었다(+0.0155 vs +0.0134). 즉 U4의 결론 2개는 스텝을
늘리면 뒤집힐 수 있는 상태였다.

여기서 고치는 3가지:

  1. **곡선을 남긴다** — 한 번 적합하면서 여러 스텝 지점에서 held-out을 재 평탄부를 눈으로 확인.
  2. **seed 여러 개** — arm 간 차이가 seed 노이즈보다 큰지. U4는 seed 하나였다.
  3. **lr을 통제한다** — conv는 14.9M, mlp는 2.9M 파라미터인데 U4는 둘 다 lr=1e-3 고정이었다.
     "conv가 진 게 아니라 lr이 안 맞았을 뿐"이라는 반박을 없애려면 projector마다 lr을 쓸어야 한다.

⚠ 16³ vs 32³에 대해 **하지 않는** 통제: "토큰 수가 8배라 평균 cosine이 불리하다"는 교란은 없다 —
토큰 수는 평균의 **분산**만 바꾸고 기댓값은 바꾸지 않는다. 진짜 질문은 "32³가 느린 것인가, 천장이
낮은 것인가"이고 그건 곡선이 직접 답한다.

student feature는 UNet forward가 비싸므로 디스크에 캐시한다 (48 vol × 5 t × 512×8³ ≈ 251 MB).

실행:
    CUDA_VISIBLE_DEVICES=1 python -m tests.repa_probe.u4b_converge.run --phase lr
    CUDA_VISIBLE_DEVICES=1 python -m tests.repa_probe.u4b_converge.run --phase grid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.rflow import RFlowScheduler
from src.data.report2ct_datamodule import _build_transforms
from src.eval.samplers.report2ct_wan import _load_wan_checkpoint
from tests.repa_probe.u4_align.run import (
    DATALIST,
    TEACHER_DIR,
    TIMESTEP_FRACS,
    build_projector,
    collect_student_feats,
    cos_loss,
    load_teacher,
    project_and_upsample,
    rel_loss,
)
from tests.repa_probe.u4_align.run import CKPT_DIR as WAN_CKPT_DIR

OUT_DIR = Path(__file__).parent
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figs"
CACHE = Path("/workspace/data/report2ct_wan/_u4b_student_cache.pt")

#: held-out cosine은 1500 언저리에서 **정점을 찍고 내려간다** (첫 실행에서 확인:
#: 0.5308 → 0.5517 → 0.5382 → 0.5193 @ 300/1500/5000/15000). 그러므로 끝값이 아니라 **정점**을
#: 봐야 하고, 그러려면 초반이 촘촘해야 한다. 6000 이후는 감쇠 구간이라 잴 가치가 없다.
EVAL_AT = (250, 500, 1000, 1500, 2500, 4000, 6000)
EVAL_CHUNK = 4  # held-out 평가 시 한 번에 올리는 볼륨 수
TAP = "middle"
CKPT = 299


def student_cache(n_volumes: int, device: str) -> tuple[torch.Tensor, list[str]]:
    """`middle_block` tap feature `(N, 5, 512, 8, 8, 8)` + 볼륨 id. 있으면 디스크에서 읽는다.

    U4와 **같은 볼륨·같은 노이즈 seed**(`torch.manual_seed(1000 + i)`)를 쓴다 — 그래야 두 probe의
    수치가 같은 축 위에 있다.
    """
    if CACHE.is_file():
        blob = torch.load(CACHE, map_location="cpu")
        if blob["n_volumes"] >= n_volumes:
            print(f"[cache] {CACHE} ({list(blob['student'].shape)})", flush=True)
            return blob["student"][:n_volumes], blob["ids"][:n_volumes]

    entries = json.loads(DATALIST.read_text())["validation"]
    entries = [
        e
        for e in entries
        if (
            TEACHER_DIR / f"{Path(e['image']).name[: -len('_emb.nii.gz')]}.npy"
        ).is_file()
    ][:n_volumes]
    ids = [Path(e["image"]).name[: -len("_emb.nii.gz")] for e in entries]

    transforms = _build_transforms(spacing_multiplier=1e2)
    batches = [transforms(dict(e)) for e in entries]
    for b in batches:
        for k in ("image", "spacing", "context_f", "context_i"):
            v = b[k]
            v = v.as_tensor() if hasattr(v, "as_tensor") else torch.as_tensor(v)
            b[k] = v.unsqueeze(0).float()

    unet, scale_factor, _ = _load_wan_checkpoint(
        WAN_CKPT_DIR / f"epoch_{CKPT:03d}.ckpt", torch.device(device)
    )
    sched = RFlowScheduler(
        num_train_timesteps=1000,
        scale=1.4,
        use_timestep_transform=True,
        use_discrete_timesteps=False,
        sample_method="uniform",
    )
    feats = []
    for i, b in enumerate(batches):
        torch.manual_seed(1000 + i)  # U4와 동일한 볼륨별 고정 노이즈
        feats.append(collect_student_feats(unet, sched, scale_factor, b, device)[TAP])
        if (i + 1) % 8 == 0:
            print(f"  student feats {i + 1}/{len(batches)}", flush=True)
    del unet
    torch.cuda.empty_cache()

    student = torch.stack(feats)  # (N, 5, 512, 8, 8, 8)
    torch.save({"student": student, "ids": ids, "n_volumes": len(ids)}, CACHE)
    print(f"[cache] wrote {CACHE} ({list(student.shape)})", flush=True)
    return student, ids


def fit_curve(
    student: torch.Tensor,
    teacher: torch.Tensor,
    split: int,
    kind: str,
    grid_n: int,
    lr: float,
    seed: int,
    device: str,
    eval_at: tuple[int, ...] = EVAL_AT,
    batch_vols: int = 4,
) -> list[dict]:
    """projector를 한 번 적합하면서 `eval_at`의 각 지점에서 held-out을 잰다.

    U4의 `fit_and_eval`은 끝값 하나만 돌려줘서 평탄부인지 아닌지 알 수 없었다. 곡선을 남기는 게
    이 probe의 핵심 변경점이다.

    Args:
        student: ``(N, 5, 512, 8, 8, 8)`` frozen UNet tap feature.
        teacher: ``(N, T, 1080)``.
        split: 앞 `split`개가 train.

    Returns:
        `eval_at` 지점마다 `{step, cos, cos_std, rel}` dict 리스트.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    grid = (grid_n,) * 3
    n_vol, n_t = student.shape[:2]
    proj = build_projector(kind, student.shape[2]).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    def evaluate() -> dict:
        """held-out 전체를 timestep마다 훑는다. 볼륨은 묶어서 처리 (한 장씩이면 GPU가 논다)."""
        proj.eval()
        cos_vals, rel_vals = [], []
        with torch.no_grad():
            for ti in range(n_t):
                for lo in range(split, n_vol, EVAL_CHUNK):
                    hi = min(lo + EVAL_CHUNK, n_vol)
                    s = student[lo:hi, ti].to(device)
                    t_ = teacher[lo:hi].to(device).float()
                    z = project_and_upsample(proj, s, grid)
                    cos_vals.append(-float(cos_loss(z, t_)) * (hi - lo))
                    rel_vals.append(float(rel_loss(z, t_, gen)) * (hi - lo))
        proj.train()
        n = (n_vol - split) * n_t
        return {"cos": float(np.sum(cos_vals) / n), "rel": float(np.sum(rel_vals) / n)}

    curve, step, t0 = [], 0, time.time()
    for target in eval_at:
        while step < target:
            sel = torch.randperm(split, generator=gen, device=device)[:batch_vols]
            ti = torch.randint(0, n_t, (batch_vols,), generator=gen, device=device)
            s = torch.stack(
                [student[v, t] for v, t in zip(sel.tolist(), ti.tolist())]
            ).to(device)
            z = project_and_upsample(proj, s, grid)
            loss = cos_loss(z, teacher[sel.cpu()].to(device).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        curve.append({"step": step, **evaluate(), "sec": round(time.time() - t0, 1)})
        print(
            f"    {kind:4s} {grid_n:2d}³ lr={lr:g} seed={seed} step={step:6d} "
            f"cos {curve[-1]['cos']:+.4f} rel {curve[-1]['rel']:.5f} "
            f"({curve[-1]['sec']:.0f}s)",
            flush=True,
        )
    return curve


def peak(curve: list[dict]) -> dict:
    """곡선의 **정점** — held-out cosine이 최대인 지점과 그때의 스텝.

    끝값을 쓰면 안 된다: cosine은 정점을 지나 감쇠하므로 끝값 비교는 "어느 arm이 더 심하게
    과적합했는가"를 재게 된다. arm마다 정점 스텝이 다를 수 있으므로 각자의 정점에서 비교한다.
    """
    best = max(curve, key=lambda p: p["cos"])
    return {"peak_cos": best["cos"], "peak_step": best["step"], "peak_rel": best["rel"]}


def phase_arms(phase: str) -> list[dict]:
    """실행할 조합. lr 단계는 16³에서 projector×lr를 쓸고, grid 단계는 최적 lr에서 해상도를 본다."""
    if phase == "lr":
        return [
            {"kind": k, "grid_n": 16, "lr": lr, "seed": s}
            for k in ("mlp", "conv")
            for lr in (3e-4, 1e-3, 3e-3)
            for s in (0, 1, 2)
        ]
    if phase == "grid":
        best = json.loads((RESULT_DIR / "best_lr.json").read_text())
        return [
            {"kind": k, "grid_n": 32, "lr": best[k], "seed": s}
            for k in ("mlp", "conv")
            for s in (0, 1, 2)
        ]  # 16³는 lr 단계에서 이미 3 seed로 측정됨 — 다시 돌리지 않는다
    raise SystemExit(f"unknown phase {phase!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("lr", "grid"), required=True)
    # 48은 train 32뿐이라 2.9M projector가 그냥 외운다 — 그게 위 정점-후-감쇠의 원인이다.
    ap.add_argument("--n-volumes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=EVAL_AT[-1])
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_at = tuple(s for s in EVAL_AT if s <= args.max_steps)

    student, ids = student_cache(args.n_volumes, device)
    split = int(len(ids) * 2 / 3)
    print(
        f"{len(ids)} volumes — train {split} / held-out {len(ids) - split}", flush=True
    )

    arms = phase_arms(args.phase)
    # fp16으로 들고 있다가 배치마다 float으로 올린다 — 150 볼륨 32³는 fp32면 21 GB다.
    teachers = {
        g: torch.stack([load_teacher(s, g, norm=True) for s in ids]).half()
        for g in sorted({a["grid_n"] for a in arms})
    }
    for g, t in teachers.items():
        print(
            f"  teacher {g}³: {list(t.shape)} {t.element_size() * t.nelement() / 2**30:.1f} GiB",
            flush=True,
        )

    rows = []
    for arm in arms:
        curve = fit_curve(
            student,
            teachers[arm["grid_n"]],
            split=split,
            kind=arm["kind"],
            grid_n=arm["grid_n"],
            lr=arm["lr"],
            seed=arm["seed"],
            device=device,
            eval_at=eval_at,
        )
        rows.append(arm | peak(curve) | {"curve": curve})
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / f"converge_{args.phase}.json").write_text(
            json.dumps(
                {
                    "phase": args.phase,
                    "n_volumes": len(ids),
                    "n_train": split,
                    "tap": TAP,
                    "ckpt": CKPT,
                    "timestep_fracs": list(TIMESTEP_FRACS),
                    "eval_at": list(eval_at),
                    "rows": rows,
                },
                indent=2,
            )
        )  # arm마다 덮어써 저장 — 중간에 죽어도 여기까지는 남는다
    print(f"\n[done] {RESULT_DIR / f'converge_{args.phase}.json'}")


if __name__ == "__main__":
    main()
