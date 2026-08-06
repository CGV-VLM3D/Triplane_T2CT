"""U4 — student↔teacher 정렬 가능성: 학습을 돌리기 전에 "맞춰지긴 하는가"를 실측한다.

이 probe가 REPA 도입의 **go/no-go**다. 학습된 wan UNet의 중간 feature를 hook으로 뽑고, 그 위에
projector만 얹어 teacher token에 얼마나 가까워질 수 있는지 잰다. 여기서 cosine도 관계형 일치도도
오르지 않으면 REPA 학습은 착수하지 않는다.

축:
  tap        `down_blocks[3]` / **`middle_block`** / `up_blocks[0]`  (U-REPA는 middle 주장)
  projector  MLP 3-layer (U-REPA) vs Conv3d k=3 (iREPA)              ← "conv로 되나"의 직접 답
  grid       teacher 16³(pooled) vs 32³(native)                       ← pooling으로 잃는 양
  ckpt       epoch 009 / 099 / 299                                     ← 사전학습된 표현일수록 hard가 어려운가
  fit loss   cos(hard) vs rel(관계형 Gram L2)                          ← Track A/B 결정 근거
  teacher    spatial norm on/off

프로토콜: 볼륨을 train/held-out으로 나눠 projector를 **train에서만** 학습하고 held-out에서 보고한다.
UNet은 완전히 frozen — 이 단계에서 재는 건 "정렬이 가능한 표현인가"이지 "학습이 되는가"가 아니다.

실행:
    CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u4_align.run [--n-volumes 48]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.spectre_adapter import SpectreBackbone
from src.data.report2ct_datamodule import _build_transforms
from src.eval.samplers.report2ct_wan import _load_wan_checkpoint
from tests.repa_probe._metrics import spatial_norm

DATALIST = Path("/workspace/data/report2ct_wan/datalist_wan_2560.json")
TEACHER_DIR = Path("/workspace/data/report2ct_wan/spectre_ssl_32")
CKPT_DIR = Path("/workspace/outputs/report2ct_wan/2026-07-16_3/checkpoints")
OUT_DIR = Path(__file__).parent
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figs"

#: RFlow 시간축에서 뽑을 지점. add_noise는 t/1000을 노이즈 비율로 쓴다
#: (`third_party/maisi_bundle/scripts/rectified_flow.py` add_noise: timepoints = 1 − t/1000).
TIMESTEP_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
TAPS = ("down3", "middle", "up0")
TEACHER_GRID = (32, 32, 32)
TEACHER_DIM = 1080
REL_TOKENS = 1024  # 관계형 손실용 Gram 부분표본 (전체 32768²는 계산 불가)


class TapRecorder:
    """UNet의 여러 블록 출력을 한 번의 forward에서 잡아두는 hook 묶음."""

    def __init__(self, unet: nn.Module) -> None:
        self.feats: dict[str, torch.Tensor] = {}
        self._handles = [
            unet.down_blocks[3].register_forward_hook(self._make("down3")),
            unet.middle_block.register_forward_hook(self._make("middle")),
            unet.up_blocks[0].register_forward_hook(self._make("up0")),
        ]

    def _make(self, name: str):
        def hook(_module, _inputs, output):
            # down/up 블록은 (hidden, residual_tuple)을 반환할 수 있다 — 첫 텐서만 쓴다.
            self.feats[name] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def build_projector(kind: str, in_ch: int, out_ch: int = TEACHER_DIM) -> nn.Module:
    """U-REPA의 MLP 또는 iREPA의 conv projector.

    둘 다 **student의 native 해상도에서** 채널을 투영한다. 업샘플은 그 뒤에 별도로 한다
    (U-REPA Table 7: "MLP 먼저, 그 다음 업스케일"이 5.72 vs upscale-first 5.84 / in-MLP 6.36).
    """
    if kind == "conv":
        # iREPA Algorithm 1의 3D판 — nn.Conv2d(D_in, D_out, kernel_size=3, padding=1)
        return nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
    if kind == "mlp":
        # U-REPA build_mlp: Linear → SiLU → Linear → SiLU → Linear (1×1×1 conv로 동일 연산)
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 1),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 1),
        )
    raise ValueError(kind)


def project_and_upsample(
    proj: nn.Module, feat: torch.Tensor, grid: tuple[int, int, int]
) -> torch.Tensor:
    """`(B, C, h, w, d)` student feature → `(B, T, 1080)` teacher 격자 위 토큰."""
    z = proj(feat)  # (B, 1080, h, w, d)
    if tuple(z.shape[2:]) != grid:
        z = F.interpolate(z, size=grid, mode="trilinear", align_corners=False)
    return z.flatten(2).transpose(1, 2)  # (B, T, 1080)


def cos_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """REPA/U-REPA Eq.1 — 토큰별 cosine 유사도의 음수 평균."""
    return -(F.normalize(student, dim=-1) * F.normalize(teacher, dim=-1)).sum(-1).mean()


def rel_loss(
    student: torch.Tensor, teacher: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """U-REPA Eq.3-4 manifold loss — cosine Gram 행렬의 Frobenius² 차이 (부분표본)."""
    idx = torch.randperm(student.shape[1], generator=generator, device=student.device)[
        :REL_TOKENS
    ]
    s = F.normalize(student[:, idx], dim=-1)
    t = F.normalize(teacher[:, idx], dim=-1)
    return ((s @ s.transpose(1, 2)) - (t @ t.transpose(1, 2))).pow(2).mean()


@torch.no_grad()
def collect_student_feats(
    unet: nn.Module,
    scheduler,
    scale_factor: float,
    batch: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    """한 볼륨에 대해 TIMESTEP_FRACS 각각에서 tap feature를 모은다.

    `src/models/report2ct_module.py:_shared_forward`의 입력 구성을 그대로 따른다 —
    `images = latent × scale_factor` → `add_noise` → `unet(x, timesteps, spacing_tensor,
    context, class_labels=1)`.

    Returns:
        tap 이름 → `(len(TIMESTEP_FRACS), C, h, w, d)`.
    """
    images = batch["image"].to(device) * scale_factor  # (1, 16, 64, 64, 64)
    context = torch.stack(
        [batch["context_f"].to(device), batch["context_i"].to(device)], dim=1
    )  # (1, 2, 2560)
    spacing = batch["spacing"].to(device)  # (1, 3), 이미 ×100
    modality = torch.ones(1, dtype=torch.long, device=device)  # CT

    rec = TapRecorder(unet)
    out: dict[str, list[torch.Tensor]] = {t: [] for t in TAPS}
    try:
        for frac in TIMESTEP_FRACS:
            t = torch.full((1,), frac * scheduler.num_train_timesteps, device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=t
            )
            unet(
                x=noisy,
                timesteps=t,
                spacing_tensor=spacing,
                context=context,
                class_labels=modality,
            )
            for name in TAPS:
                out[name].append(rec.feats[name][0].float().cpu())
    finally:
        rec.close()
    return {k: torch.stack(v) for k, v in out.items()}


def load_teacher(scan_id: str, grid_n: int, norm: bool) -> torch.Tensor:
    """저장된 32³ teacher를 읽어 필요하면 pool + spatial norm 한다 → `(T, 1080)` fp32."""
    arr = np.load(TEACHER_DIR / f"{scan_id}.npy")  # (32768, 1080) fp16
    t = torch.from_numpy(arr).float()
    if grid_n != TEACHER_GRID[0]:
        t = SpectreBackbone.pool_grid(
            t.reshape(*TEACHER_GRID, -1), (grid_n,) * 3
        ).reshape(-1, TEACHER_DIM)
    return spatial_norm(t) if norm else t


def fit_and_eval(
    student: torch.Tensor,
    teacher: torch.Tensor,
    split: int,
    kind: str,
    grid: tuple[int, int, int],
    fit_loss: str,
    device: str,
    steps: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    """projector를 train 볼륨에서만 학습하고 held-out에서 cosine / 관계형 일치도를 보고한다.

    Args:
        student: `(N, T_steps, C, h, w, d)` — hook으로 잡은 tap feature (frozen UNet).
        teacher: `(N, T, 1080)`.
        split: 앞 `split`개가 train, 나머지가 held-out.
        kind: "mlp" | "conv".
        fit_loss: "cos" | "rel" — 무엇을 최소화해 적합할지.

    Returns:
        held-out 기준 `cos`(높을수록 좋음), `rel`(낮을수록 좋음), 그리고 학습 곡선 끝값.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    n_vol, n_t = student.shape[:2]
    proj = build_projector(kind, student.shape[2]).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    def batch(vol_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ti = torch.randint(0, n_t, (len(vol_idx),), generator=gen, device=device)
        s = torch.stack(
            [student[v, t] for v, t in zip(vol_idx.tolist(), ti.tolist())]
        ).to(device)
        t_ = teacher[vol_idx.cpu()].to(device)
        return s, t_

    train_idx = torch.arange(split, device=device)
    for _ in range(steps):
        sel = train_idx[torch.randperm(split, generator=gen, device=device)[:2]]
        s, t_ = batch(sel)
        z = project_and_upsample(proj, s, grid)
        loss = cos_loss(z, t_) if fit_loss == "cos" else rel_loss(z, t_, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    proj.eval()
    cos_vals, rel_vals = [], []
    with torch.no_grad():
        for v in range(split, n_vol):
            for ti in range(n_t):
                s = student[v, ti].unsqueeze(0).to(device)
                t_ = teacher[v : v + 1].to(device)
                z = project_and_upsample(proj, s, grid)
                cos_vals.append(-float(cos_loss(z, t_)))
                rel_vals.append(float(rel_loss(z, t_, gen)))
    return {
        "cos": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "rel": float(np.mean(rel_vals)),
        "n_heldout_pairs": len(cos_vals),
        "proj_params_M": round(sum(p.numel() for p in proj.parameters()) / 1e6, 2),
    }


def arms() -> list[dict]:
    """중복 없이 필요한 조합만. 기준 arm(ckpt299 / middle / conv / 16³ / cos / norm)을 축마다 하나씩 흔든다."""
    base = dict(ckpt=299, tap="middle", proj="conv", grid_n=16, fit="cos", tnorm=True)
    out = [base]
    out += [base | {"tap": t} for t in ("down3", "up0")]  # tap 선택
    out += [base | {"proj": "mlp"}]  # projector 선택
    out += [base | {"grid_n": 32}, base | {"grid_n": 32, "proj": "mlp"}]  # 해상도
    out += [base | {"ckpt": c} for c in (9, 99)]  # 학습 단계
    out += [base | {"fit": "rel"}, base | {"ckpt": 9, "fit": "rel"}]  # hard vs soft
    out += [base | {"tnorm": False}]  # teacher spatial norm
    seen, uniq = set(), []
    for a in out:
        key = tuple(sorted(a.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-volumes", type=int, default=48)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    entries = json.loads(DATALIST.read_text())["validation"]
    entries = [
        e
        for e in entries
        if (
            TEACHER_DIR / f"{Path(e['image']).name[: -len('_emb.nii.gz')]}.npy"
        ).is_file()
    ]
    entries = entries[: args.n_volumes]
    ids = [Path(e["image"]).name[: -len("_emb.nii.gz")] for e in entries]
    print(f"{len(ids)} volumes with teacher features ready")
    if len(ids) < 16:
        raise SystemExit(
            "teacher feature가 아직 충분히 만들어지지 않았다 — precompute 진행 중"
        )

    transforms = _build_transforms(spacing_multiplier=1e2)
    batches = [transforms(dict(e)) for e in entries]
    for b in batches:  # (C,H,W,D) → (1,C,H,W,D) 등 배치 축 추가
        for k in ("image", "spacing", "context_f", "context_i"):
            v = b[k]
            v = v.as_tensor() if hasattr(v, "as_tensor") else torch.as_tensor(v)
            b[k] = v.unsqueeze(0).float()

    needed_ckpts = sorted({a["ckpt"] for a in arms()})
    students: dict[int, dict[str, torch.Tensor]] = {}
    for ep in needed_ckpts:
        unet, scale_factor, _ = _load_wan_checkpoint(
            CKPT_DIR / f"epoch_{ep:03d}.ckpt", torch.device(device)
        )
        from src.baselines.rflow import RFlowScheduler  # noqa: PLC0415

        sched = RFlowScheduler(
            num_train_timesteps=1000,
            scale=1.4,
            use_timestep_transform=True,
            use_discrete_timesteps=False,
            sample_method="uniform",
        )
        per_tap: dict[str, list[torch.Tensor]] = {t: [] for t in TAPS}
        for i, b in enumerate(batches):
            torch.manual_seed(1000 + i)  # 볼륨마다 고정 노이즈 → arm 간 비교 가능
            feats = collect_student_feats(unet, sched, scale_factor, b, device)
            for t in TAPS:
                per_tap[t].append(feats[t])
        students[ep] = {t: torch.stack(v) for t, v in per_tap.items()}
        shapes = {t: list(v.shape) for t, v in students[ep].items()}
        print(
            f"  ckpt ep{ep:03d}: scale_factor={scale_factor:.4f} taps={shapes}",
            flush=True,
        )
        del unet
        torch.cuda.empty_cache()

    teachers: dict[tuple[int, bool], torch.Tensor] = {}
    for grid_n, tnorm in {(a["grid_n"], a["tnorm"]) for a in arms()}:
        teachers[(grid_n, tnorm)] = torch.stack(
            [load_teacher(s, grid_n, tnorm) for s in ids]
        )
        print(
            f"  teacher grid={grid_n}³ norm={tnorm}: {list(teachers[(grid_n, tnorm)].shape)}",
            flush=True,
        )

    split = int(len(ids) * 2 / 3)
    rows = []
    for arm in arms():
        res = fit_and_eval(
            students[arm["ckpt"]][arm["tap"]],
            teachers[(arm["grid_n"], arm["tnorm"])],
            split=split,
            kind=arm["proj"],
            grid=(arm["grid_n"],) * 3,
            fit_loss=arm["fit"],
            device=device,
            steps=args.steps,
        )
        rows.append(arm | res)
        print(
            f"  ep{arm['ckpt']:03d} {arm['tap']:6s} {arm['proj']:4s} {arm['grid_n']:2d}³ "
            f"fit={arm['fit']:3s} tnorm={int(arm['tnorm'])} → cos {res['cos']:+.4f} "
            f"rel {res['rel']:.5f} ({res['proj_params_M']}M)",
            flush=True,
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "alignability.json"
    out.write_text(
        json.dumps(
            {
                "n_volumes": len(ids),
                "n_train": split,
                "ids": ids,
                "timestep_fracs": list(TIMESTEP_FRACS),
                "steps": args.steps,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
