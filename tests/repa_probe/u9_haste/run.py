"""U9 — HASTE stage-wise termination 진단.

repa300은 `repa_stop_step=null`로 300 epoch(187.5k step) 내내 정렬 손실을 켜 둔 채로 학습됐다.
HASTE("REPA Works Until It Doesn't", arXiv:2505.16792)는 이게 후반부에 오히려 해가 될 수 있다고
주장하고(capacity mismatch), 그 근거로 두 손실의 그래디언트 방향 코사인 유사도

    rho_t = cos(grad_theta L_diff, grad_theta L_repa)

를 재서 양수(도움) -> 0(무해) -> 음수(방해)로 넘어가는 지점을 찾는다(논문 Sec 2.2, Fig 3/4). 이
probe는 재학습 없이 repa300의 기존 30개 체크포인트(ep009~ep299, 10 epoch 간격)에 그대로 이 방법을
적용해, 우리 U-Net에서도 같은 전환이 일어나는지 + 몇 epoch쯤인지를 잰다.

측정 대상 파라미터: REPA gradient가 실제로 닿는 범위 — `unet.conv_in` + `unet.down_blocks` +
`unet.middle_block` (hook은 middle_block 출력 하나만 잡지만, backprop은 그 조상 전체로 흐른다).
`up_blocks`/`out`은 diffusion loss만 받으므로 애초에 REPA gradient가 없어 비교 대상이 아니다.

타임스텝은 `tests/repa_probe/u4_align`과 동일한 관례로 `frac * num_train_timesteps`를
`use_timestep_transform` 없이 직접 주입한다(HASTE Fig.4 재현 축: t in {0.1,...,0.9} — 낮은 t=
fine-detail 구간에서 conflict가 먼저 온다는 게 HASTE의 관찰).

체크포인트 재구성은 `hydra.utils.instantiate(cfg.model)`로 unet+noise_scheduler+repa를 한 번에
새로 만든 뒤 `load_state_dict(strict=True)`로 학습된 가중치를 얹는다 — 기존 eval 샘플러
(`_load_wan_checkpoint`)는 unet만 읽고 repa 서브모듈을 버리므로 이 진단에는 쓸 수 없다.

v1(8 volume, 노이즈 draw 1회/포인트)은 rho_overall이 전 구간 +0.02~+0.17 사이에서 오르내리고
전반(ep9-149) 평균 0.069 vs 후반(ep159-299) 평균 0.077로 하락 추세가 안 보였다 — 그런데 포인트당
노이즈 밴드(±0.02~0.05)가 신호 크기와 비슷해서 느린 하락이 노이즈에 묻혔을 가능성이 있었다. v2는
그 노이즈를 두 방향에서 줄인다: ① volume 수 8→16, ② 같은 (batch, timestep, repeat) 조합에
**체크포인트 간 동일한 노이즈 시드**를 강제(common random numbers) + repeat=3회 평균 — 체크포인트
고유의 신호만 남기고 노이즈 draw로 인한 분산을 제거한다.

실행:
    CUDA_VISIBLE_DEVICES=1 python -m tests.repa_probe.u9_haste.run
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
import torch.nn.functional as F
from monai.data import list_data_collate
from omegaconf import OmegaConf

CONFIG_PATH = Path(
    "/workspace/outputs/report2ct_wan_repa_300/2026-07-30/.hydra/config.yaml"
)
CKPT_DIR = Path("/workspace/outputs/report2ct_wan_repa_300/2026-07-30/checkpoints")
OUT_DIR = Path(__file__).parent
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figs"

#: HASTE Fig.4과 동일 축 (u4_align의 TIMESTEP_FRACS와도 동일).
TIMESTEP_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_PROBE_VOLUMES = 16  # v1의 2배
PROBE_BATCH_SIZE = 4
#: 같은 (batch, frac) 조합을 이 횟수만큼 독립된 노이즈 draw로 반복 — v1은 1회였다.
NUM_REPEATS = 3
#: REPA gradient가 실제로 흐르는 파라미터 prefix (middle_block hook의 조상 전체).
GRAD_PREFIXES = ("unet.conv_in.", "unet.down_blocks.", "unet.middle_block.")


def epoch_of(ckpt: Path) -> int:
    """``epoch_009.ckpt`` -> ``9``."""
    return int(ckpt.stem.split("_")[1])


def build_probe_batches(cfg: OmegaConf) -> list[dict]:
    """train split 앞 ``N_PROBE_VOLUMES``개를 고정 probe set으로 배치화.

    모든 체크포인트가 동일한 probe set을 보게 해서, 체크포인트(=epoch) 하나만 변수로 남긴다.
    """
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    n = min(N_PROBE_VOLUMES, len(dm.train_ds))
    samples = [dm.train_ds[i] for i in range(n)]
    return [
        list_data_collate(samples[i : i + PROBE_BATCH_SIZE])
        for i in range(0, n, PROBE_BATCH_SIZE)
    ]


def to_device(batch: dict, device: str) -> dict:
    """MetaTensor -> plain tensor, 지정 device로, float으로."""
    out = {}
    for k, v in batch.items():
        if hasattr(v, "as_tensor"):
            v = v.as_tensor()
        out[k] = v.to(device).float() if torch.is_tensor(v) else v
    return out


def load_module(cfg: OmegaConf, ckpt_path: Path, device: str):
    """체크포인트 하나를 grad 가능한 ``Report2CTModule``로 재구성.

    `tests/test_report2ct_wan_repa.py::_build_module`과 동일한 fake-trainer 패턴 —
    실제 ``Trainer.fit()`` 없이 ``self.trainer``/``self.log`` 접근을 무해하게 만든다.
    """
    module = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    # `scale_factor`는 버퍼라 이미 로드됐지만, `_scale_factor_initialized`는 일반 bool 속성이라
    # fresh instantiate 후 False로 리셋된다 — 그대로 두면 첫 forward가 8개짜리 probe 배치의
    # std로 재계산해 체크포인트 값을 덮어쓴다.
    module._scale_factor_initialized = True
    module._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=100, global_step=0
    )
    module.log = lambda *a, **k: None
    module.to(device)
    module.setup("fit")  # repa forward hook 등록
    module.train()
    return module


def flat_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat(
        [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    )


def rho_at(module, batch: dict, frac: float, seed: int) -> float:
    """한 배치·한 timestep·한 노이즈 draw에서 rho_t = cos(grad L_diff, grad L_repa).

    `seed`는 호출자가 (batch_idx, frac, repeat_idx)로부터 만들어 넘긴다 — 모든 체크포인트가
    동일한 seed 집합을 쓰므로 `_shared_forward` 내부의 `torch.randn_like(images)` 노이즈가
    체크포인트 간에 완전히 동일해진다(common random numbers). 그러면 포인트 간 차이가
    "어쩌다 뽑힌 노이즈"가 아니라 진짜 체크포인트(=epoch) 신호만 반영한다.
    """
    torch.manual_seed(seed)
    n_train = module.noise_scheduler.num_train_timesteps
    original = module.noise_scheduler.sample_timesteps
    module.noise_scheduler.sample_timesteps = lambda x_start, _f=frac, _n=n_train: (
        torch.full((x_start.shape[0],), _f * _n, device=x_start.device)
    )
    try:
        params = [
            p for n, p in module.named_parameters() if n.startswith(GRAD_PREFIXES)
        ]
        module.zero_grad(set_to_none=True)
        loss_diff = module._shared_forward(batch)
        repa_loss, _ = module._last_repa
        if repa_loss is None:
            raise RuntimeError("repa_loss is None — REPA가 이 스텝에서 비활성 상태")
        loss_diff.backward(retain_graph=True)
        g_diff = flat_grad(params).clone()
        module.zero_grad(set_to_none=True)
        repa_loss.backward()
        g_repa = flat_grad(params).clone()
        return F.cosine_similarity(g_diff.unsqueeze(0), g_repa.unsqueeze(0)).item()
    finally:
        module.noise_scheduler.sample_timesteps = original


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in TIMESTEP_FRACS:
        ax.plot(
            epochs,
            [r["rho_by_t"][str(f)] for r in rows],
            marker="o",
            ms=3,
            label=f"t={f}",
        )
    ax.plot(epochs, [r["rho_overall"] for r in rows], "k--", lw=2, label="overall")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\rho = \cos(\nabla L_{diff}, \nabla L_{repa})$")
    ax.set_title("repa300 — HASTE gradient-conflict diagnostic")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "haste_rho_curve_v2.png", dpi=150)
    print(f"[fig] {FIG_DIR / 'haste_rho_curve_v2.png'}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = OmegaConf.load(CONFIG_PATH)

    probe_batches_cpu = build_probe_batches(cfg)
    n_vol = sum(b["image"].shape[0] for b in probe_batches_cpu)
    print(
        f"{len(probe_batches_cpu)} probe batch(es), {n_vol} volumes "
        "(fixed across all checkpoints)",
        flush=True,
    )

    ckpts = sorted(CKPT_DIR.glob("epoch_*.ckpt"), key=epoch_of)
    rows = []
    for ckpt_path in ckpts:
        ep = epoch_of(ckpt_path)
        module = load_module(cfg, ckpt_path, device)

        per_t: dict[float, list[float]] = {f: [] for f in TIMESTEP_FRACS}
        for b_idx, batch_cpu in enumerate(probe_batches_cpu):
            batch = to_device(batch_cpu, device)
            for f_idx, frac in enumerate(TIMESTEP_FRACS):
                for r_idx in range(NUM_REPEATS):
                    # 체크포인트와 무관한 seed → 모든 체크포인트가 같은 노이즈를 본다.
                    seed = b_idx * 10_000 + f_idx * 1_000 + r_idx
                    per_t[frac].append(rho_at(module, batch, frac, seed))

        def _mean(v: list[float]) -> float:
            return sum(v) / len(v)

        def _std(v: list[float]) -> float:
            m = _mean(v)
            return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

        row = {
            "epoch": ep,
            "rho_by_t": {str(f): _mean(v) for f, v in per_t.items()},
            "rho_by_t_std": {str(f): _std(v) for f, v in per_t.items()},
            "n_samples_per_t": len(next(iter(per_t.values()))),
        }
        row["rho_overall"] = _mean(list(row["rho_by_t"].values()))
        rows.append(row)
        print(
            f"  ep{ep:03d}: rho_overall={row['rho_overall']:+.4f}  "
            f"by_t={ {k: round(v, 3) for k, v in row['rho_by_t'].items()} }",
            flush=True,
        )

        del module
        gc.collect()
        torch.cuda.empty_cache()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "haste_rho_curve_v2.json"
    out_path.write_text(
        json.dumps(
            {
                "timestep_fracs": list(TIMESTEP_FRACS),
                "n_probe_volumes": n_vol,
                "num_repeats": NUM_REPEATS,
                "grad_param_prefixes": list(GRAD_PREFIXES),
                "note": (
                    "v2: v1(haste_rho_curve.json, 8 volume/1 draw)이 하락 추세를 못 봤는데 "
                    "포인트당 노이즈가 신호와 비슷한 크기였다. v2는 volume 16개 + 체크포인트 간 "
                    "common-random-numbers(같은 (batch,frac,repeat) 조합은 전 체크포인트가 동일 "
                    "노이즈 시드) + repeat=3 평균으로 노이즈를 줄인다. timestep은 "
                    "use_timestep_transform 없이 frac*num_train_timesteps로 직접 주입 "
                    "(tests/repa_probe/u4_align과 동일 관례)."
                ),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n[done] {out_path}")

    try:
        plot(rows)
    except Exception as e:  # noqa: BLE001 — 그림은 부가 산출물, 실패해도 json은 지킨다
        print(f"plot 실패(무시): {e}")


if __name__ == "__main__":
    main()
