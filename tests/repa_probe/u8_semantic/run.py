"""U8 — 표현의 **의미적 품질**: REPA 논문의 linear-probe / CKNN 분석을 CT-native로.

REPA(§3, Fig 2)는 "diffusion 모델 내부 표현이 DINOv2 같은 SSL 인코더보다 얼마나 덜 semantic한가"를
**linear probing**과 **CKNN(k-NN 정확도)**으로 재고, 그 격차가 REPA로 좁혀지는 걸 보였다. 그리고
**CKA**로 teacher와의 정렬을 잰다. ImageNet 라벨 대신 우리는 CT-RATE의 **18개 abnormality 라벨**을
쓴다 — 다중 라벨이라 정확도 대신 **평균 AUROC**로 보고한다.

이 probe가 답하는 것:
  1. **어느 layer가 가장 semantic한가** — bottleneck만이 답이 아닐 수 있다 (`repa_tap` 선택 근거)
  2. **REPA가 실제로 표현을 semantic하게 만드는가** — baseline UNet vs REPA-학습 UNet 같은 epoch 비교
  3. **teacher와의 CKA** — 어느 layer가 teacher와 이미 가까운가
  4. teacher 후보들 자체의 semantic 상한 (SPECTRE / CT-FM / CT-CLIP)

실행:
    CUDA_VISIBLE_DEVICES=0 python -m tests.repa_probe.u8_semantic.run \\
        --ckpts baseline=outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_029.ckpt \\
                repa=outputs/report2ct_wan_repa/<dir>/checkpoints/epoch_029.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader  # ⚠ torch의 것이 아니다 — 아래 주석 참조

LABELS_CSV = Path(
    "/workspace/datasets/datasets/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
)
DATALIST = Path("/workspace/data/report2ct_wan/datalist_wan_2560.json")
TEACHER_DIRS = {
    "spectre_ssl": Path("/workspace/data/report2ct_wan/spectre_ssl_16"),
    "spectre_vla": Path("/workspace/data/report2ct_wan/spectre_vla_16"),
}
OUT_DIR = Path(__file__).parent
RESULT_DIR, FIG_DIR = OUT_DIR / "results", OUT_DIR / "figs"

#: UNet의 모든 tap 지점 — bottleneck만 보는 게 아니라 전 stage를 훑는다 (REPA Fig 2의 layer 스윕).
TAPS = (
    "conv_in",
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "down_blocks.3",
    "middle_block",
    "up_blocks.0",
    "up_blocks.1",
    "up_blocks.2",
)
TIMESTEP_FRAC = 0.5  # REPA도 중간 노이즈 수준에서 표현 품질을 잰다


def load_labels(vol_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """18개 abnormality 라벨을 `(N, 18)` 0/1 행렬로."""
    df = pd.read_csv(LABELS_CSV)
    df["vid"] = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    df = df.set_index("vid")
    names = [c for c in df.columns if c not in ("VolumeName",)]
    return df.loc[vol_ids, names].to_numpy().astype(np.float32), names


def mean_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """라벨별 AUROC의 평균 (한 클래스만 있는 라벨은 건너뜀)."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    scores = [
        roc_auc_score(y_true[:, j], y_score[:, j])
        for j in range(y_true.shape[1])
        if 0 < y_true[:, j].sum() < len(y_true)
    ]
    return float(np.mean(scores)) if scores else float("nan")


def linear_probe(x_tr, y_tr, x_te, y_te) -> float:
    """라벨마다 로지스틱 회귀 하나 — REPA의 linear probing에 대응."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    sc = StandardScaler().fit(x_tr)
    x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    pred = np.zeros_like(y_te)
    for j in range(y_tr.shape[1]):
        if not 0 < y_tr[:, j].sum() < len(y_tr):
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, y_tr[:, j])
        pred[:, j] = clf.predict_proba(x_te)[:, 1]
    return mean_auroc(y_te, pred)


def cknn(x_tr, y_tr, x_te, y_te, k: int = 20) -> float:
    """CKNN — 코사인 최근접 k개 이웃의 라벨 평균을 점수로 (REPA가 쓴 지표의 다중라벨판)."""
    a = torch.nn.functional.normalize(torch.from_numpy(x_tr).float(), dim=-1)
    b = torch.nn.functional.normalize(torch.from_numpy(x_te).float(), dim=-1)
    idx = (b @ a.T).topk(min(k, len(a)), dim=-1).indices.numpy()  # (N_te, k)
    return mean_auroc(y_te, y_tr[idx].mean(axis=1))


def cka(x: np.ndarray, y: np.ndarray) -> float:
    """선형 CKA — 두 표현이 같은 샘플 관계 구조를 갖는가 (0~1, 클수록 정렬)."""
    xc = torch.from_numpy(x).float()
    yc = torch.from_numpy(y).float()
    xc = xc - xc.mean(0, keepdim=True)
    yc = yc - yc.mean(0, keepdim=True)
    xy = (xc.T @ yc).pow(2).sum()
    xx = (xc.T @ xc).pow(2).sum().sqrt()
    yy = (yc.T @ yc).pow(2).sum().sqrt()
    return float(xy / (xx * yy))


class MultiTap:
    """여러 블록 출력을 한 forward에서 잡아 global-pool까지 해두는 hook 묶음."""

    def __init__(self, unet: torch.nn.Module, taps: tuple[str, ...]) -> None:
        self.pooled: dict[str, torch.Tensor] = {}
        self._handles = []
        for name in taps:
            module = unet
            for part in name.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            self._handles.append(module.register_forward_hook(self._make(name)))

    def _make(self, name: str):
        def hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            self.pooled[name] = t.detach().float().mean(dim=(2, 3, 4))  # (B, C)

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()


@torch.no_grad()
def student_reps(
    ckpt: Path, batches: list[dict], device: str, drop_context: bool = True
) -> dict[str, np.ndarray]:
    """체크포인트 하나에서 모든 tap의 global-pool 표현을 뽑는다.

    ⚠ `drop_context=True`가 기본인 이유: 18개 abnormality 라벨은 **radiology report에서 추출된
    것**이고 UNet은 바로 그 리포트를 cross-attention 조건으로 받는다. 조건을 켠 채로 probe하면
    "이미지 표현이 얼마나 semantic한가"가 아니라 "텍스트 조건을 얼마나 잘 베껴 왔는가"를 재게 된다.
    CFG의 unconditional 경로처럼 context를 0으로 두고 이미지 정보만 남긴다.
    """
    from src.baselines.rflow import RFlowScheduler  # noqa: PLC0415
    from src.eval.samplers.report2ct_wan import _load_wan_checkpoint  # noqa: PLC0415

    unet, scale_factor, _ = _load_wan_checkpoint(ckpt, torch.device(device))
    sched = RFlowScheduler(
        num_train_timesteps=1000,
        scale=1.4,
        use_timestep_transform=True,
        use_discrete_timesteps=False,
        sample_method="uniform",
    )
    rec = MultiTap(unet, TAPS)
    out: dict[str, list[np.ndarray]] = {t: [] for t in TAPS}
    try:
        for i, b in enumerate(batches):
            torch.manual_seed(2000 + i)  # 볼륨마다 고정 노이즈 → ckpt 간 비교 가능
            images = b["image"].to(device) * scale_factor
            t = torch.full((images.shape[0],), TIMESTEP_FRAC * 1000, device=device)
            noisy = sched.add_noise(
                original_samples=images, noise=torch.randn_like(images), timesteps=t
            )
            context = torch.stack(
                [b["context_f"].to(device), b["context_i"].to(device)], dim=1
            )
            if drop_context:
                context = torch.zeros_like(context)  # CFG unconditional 경로
            unet(
                x=noisy,
                timesteps=t,
                spacing_tensor=b["spacing"].to(device),
                context=context,
                class_labels=torch.ones(
                    images.shape[0], dtype=torch.long, device=device
                ),
            )
            for name in TAPS:
                out[name].append(rec.pooled[name].cpu().numpy())
    finally:
        rec.close()
        del unet
        torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


def teacher_reps(vol_ids: list[str]) -> dict[str, np.ndarray]:
    """precompute된 teacher 격자를 global mean-pool → `(N, 1080)`."""
    out = {}
    for name, d in TEACHER_DIRS.items():
        if not d.is_dir():
            continue
        out[name] = np.stack(
            [np.load(d / f"{v}.npy").astype(np.float32).mean(axis=0) for v in vol_ids]
        )
        # SPECTRE combiner의 scan-level 임베딩 — dense 격자를 평균낸 것보다 공정한 global 표현.
        # VLA 백본에서만 학습된 combiner라 vla에만 존재한다.
        gdir = d.parent / f"{d.name[:-3]}_32"
        if (gdir / f"{vol_ids[0]}_global.npy").is_file():
            out[f"{name}_scanlevel"] = np.stack(
                [np.load(gdir / f"{v}_global.npy").astype(np.float32) for v in vol_ids]
            )
    return out


def plot(rows: list[dict], out_name: str) -> Path:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PANELS = (
        ("linear_auroc", "linear probe (mean AUROC)"),
        ("cknn_auroc", "k-NN probe (mean AUROC)"),
        ("cka_spectre_ssl", "CKA vs SPECTRE-SSL"),
    )
    students = sorted({r["model"] for r in rows if r["kind"] == "student"})
    # context ON/OFF는 **행으로 분리**한다. 한 축에 22개 선을 겹치면 읽을 수 없다.
    modes = [("", "context OFF")]
    if any(m.endswith("+ctx") for m in students):
        # 그림 안 문자열은 **영문**으로 — matplotlib 기본 폰트에 한글 글리프가 없다.
        modes.append(("+ctx", "context ON (labels are report-derived: optimistic)"))
    # arm(base/repa)은 **마커와 색계열**로, epoch은 **같은 계열 안의 명암**으로 구분한다.
    arms = sorted({m.split("_ep")[0] for m in students if not m.endswith("+ctx")})
    marks = ("o", "^", "s", "D")
    maps = ("Blues", "Reds", "Greens", "Purples")

    # sharey="col" — OFF/ON을 같은 눈금에 놓아야 context가 얼마나 끌어올리는지가 눈에 보인다.
    fig, axes = plt.subplots(
        len(modes), 3, figsize=(17.5, 4.6 * len(modes)), squeeze=False, sharey="col"
    )
    for row, (suffix, mode_label) in enumerate(modes):
        for ai, arm in enumerate(arms):
            # ⚠ `m.endswith("")`는 항상 참이라 OFF 행에 +ctx까지 딸려온다 — 명시적으로 가른다.
            eps = sorted(
                (
                    m
                    for m in students
                    if m.split("_ep")[0] == arm
                    and (m.endswith("+ctx") if suffix else not m.endswith("+ctx"))
                ),
                key=lambda m: int(m.split("_ep")[1][:3]),
            )
            cmap = plt.get_cmap(maps[ai % len(maps)])
            for ei, model in enumerate(eps):
                sel = sorted(
                    (r for r in rows if r["model"] == model and r["kind"] == "student"),
                    key=lambda r: TAPS.index(r["tap"]),
                )
                # 진할수록 늦은 epoch (한 계열 안에서 0.35 → 0.95)
                shade = cmap(0.35 + 0.6 * ei / max(len(eps) - 1, 1))
                for col, (key, _) in enumerate(PANELS):
                    axes[row][col].plot(
                        range(len(sel)),
                        [r[key] for r in sel],
                        marker=marks[ai % len(marks)],
                        ms=4,
                        lw=1.4,
                        color=shade,
                        label=model.replace(suffix, "") if suffix else model,
                    )
        for col, (key, title) in enumerate(PANELS):
            ax = axes[row][col]
            ax.set_xticks(range(len(TAPS)))
            ax.set_xticklabels(TAPS, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
            if col == 0:  # 행 이름은 왼쪽에 한 번만 — 패널마다 붙이면 제목이 겹친다
                ax.set_ylabel(mode_label, fontsize=10, labelpad=8)
            ax.grid(alpha=0.3)
            if key != "cka_spectre_ssl":  # teacher 상한선은 probe 패널에만 의미가 있다
                for r in (r for r in rows if r["kind"] == "teacher"):
                    ax.axhline(r[key], ls=":", lw=1.0, color="0.35")
                    ax.text(
                        len(TAPS) - 1,
                        r[key],
                        f"{r['model']} {r[key]:.3f} ",
                        fontsize=6,
                        color="0.25",
                        ha="right",
                        va="bottom",
                    )
        axes[row][2].legend(  # 범례는 축 **바깥** — 안에 두면 데이터를 덮는다
            loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False
        )
    fig.suptitle(
        "U8 — semantic quality by UNet stage (CT-RATE 18 abnormality labels)\n"
        f"marker: {' / '.join(f'{m} = {a}' for a, m in zip(arms, marks))}"
        "     ·     darker shade = later epoch",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path/to.ckpt")
    ap.add_argument("--n-volumes", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--with-context",
        action="store_true",
        help="리포트 조건을 켠 채로도 잰다 (라벨이 리포트에서 나왔으므로 낙관적 상한)",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.data.report2ct_datamodule import Report2CTDataModule  # noqa: PLC0415

    entries = json.loads(DATALIST.read_text())["validation"][: args.n_volumes]
    tmp = Path("/workspace/data/report2ct_wan/_u8_datalist.json")
    tmp.write_text(json.dumps({"training": entries, "validation": entries[:8]}))
    dm = Report2CTDataModule(
        datalist_path=str(tmp),
        batch_size=args.batch_size,
        num_workers=8,
        cache_rate=0.0,
        spacing_multiplier=100.0,
    )
    dm.setup("fit")
    # ⚠ `dm.train_dataloader()`는 shuffle=True다. 그 순서로 train/test를 자르면 **분할이 런마다
    # 바뀌어** probe AUROC를 런 간에 비교할 수 없다 (같은 teacher가 0.670 vs 0.711로 갈렸다).
    # datalist 순서 그대로 도는 로더를 직접 만들어 분할을 고정한다. CKA는 300볼륨 전체를 쓰고
    # 샘플 순서에 불변이라 원래 영향이 없었다.
    # ⚠ 반드시 **monai**의 DataLoader여야 한다. torch 기본 collate는 meta의
    # `filename_or_obj`를 문자열 하나로 뭉쳐서, 아래 `order` 순회가 글자 단위로 쪼개진다.
    batches = list(
        DataLoader(
            dm.train_ds,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
        )
    )
    tmp.unlink(missing_ok=True)
    order = [
        Path(p).name[: -len("_emb.nii.gz")]
        for b in batches
        for p in b["image"].meta["filename_or_obj"]
    ]
    y, label_names = load_labels(order)
    n_tr = int(len(order) * 2 / 3)
    print(
        f"{len(order)} volumes, {y.shape[1]} labels, train {n_tr} / test {len(order) - n_tr}"
    )

    teachers = teacher_reps(order)
    rows = []
    for name, rep in teachers.items():
        rows.append(
            {
                "kind": "teacher",
                "model": name,
                "tap": "-",
                "dim": rep.shape[1],
                "linear_auroc": linear_probe(
                    rep[:n_tr], y[:n_tr], rep[n_tr:], y[n_tr:]
                ),
                "cknn_auroc": cknn(rep[:n_tr], y[:n_tr], rep[n_tr:], y[n_tr:]),
                "cka_spectre_ssl": cka(rep, teachers["spectre_ssl"]),
            }
        )
        print(
            f"  teacher {name:14s} linear {rows[-1]['linear_auroc']:.4f}  cknn {rows[-1]['cknn_auroc']:.4f}"
        )

    modes = [("", True)] + ([("+ctx", False)] if args.with_context else [])
    for spec in args.ckpts:
        base_model, path = spec.split("=", 1)
        for suffix, drop in modes:
            model = base_model + suffix
            reps = student_reps(Path(path), batches, device, drop_context=drop)
            for tap, rep in reps.items():
                rows.append(
                    {
                        "kind": "student",
                        "model": model,
                        "tap": tap,
                        "dim": rep.shape[1],
                        "linear_auroc": linear_probe(
                            rep[:n_tr], y[:n_tr], rep[n_tr:], y[n_tr:]
                        ),
                        "cknn_auroc": cknn(rep[:n_tr], y[:n_tr], rep[n_tr:], y[n_tr:]),
                        "cka_spectre_ssl": cka(rep, teachers["spectre_ssl"]),
                    }
                )
                r = rows[-1]
                print(
                    f"  {model:10s} {tap:16s} dim {r['dim']:4d}  linear {r['linear_auroc']:.4f}  "
                    f"cknn {r['cknn_auroc']:.4f}  CKA {r['cka_spectre_ssl']:.4f}",
                    flush=True,
                )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot(rows, "U8_semantic_by_stage.png")
    (RESULT_DIR / "semantic.json").write_text(
        json.dumps(
            {
                "n_volumes": len(order),
                "labels": label_names,
                "timestep_frac": TIMESTEP_FRAC,
                "figure": str(fig.relative_to(Path("/workspace"))),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n[done] {fig}")


if __name__ == "__main__":
    main()
