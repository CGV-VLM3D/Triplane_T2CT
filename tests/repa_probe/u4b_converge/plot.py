"""U4b 곡선 그림 — held-out cosine이 스텝에 따라 어떻게 움직이는가.

U4가 1500 스텝 한 점만 보고 기본값 2개(projector=mlp, teacher grid=16³)를 정했는데, 곡선을
그려보니 **정점을 지나 감쇠**하고 있었다. 이 그림이 그 사실과, 학습 볼륨을 32 → 100으로 늘리면
감쇠가 사라지고 더 높은 곳에서 평탄해진다는 것을 한 장에 보여준다.

실행: `python -m tests.repa_probe.u4b_converge.plot`
"""

from __future__ import annotations

import json
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figs"

#: 48 볼륨(train 32)에서 얻은 첫 실행 — 과적합 증거로 남겨둔 곡선.
#: `logs/u4b_lr_20260729_*.log`의 mlp/16³/lr=3e-4/seed=0.
OVERFIT_48 = {
    "steps": [300, 1500, 5000, 15000],
    "cos": [0.5308, 0.5517, 0.5382, 0.5193],
}

COLORS = {"mlp": "#1f77b4", "conv": "#d62728"}
STYLES = {0.0003: ":", 0.001: "--", 0.003: "-"}


def main() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    paths = sorted(RESULT_DIR.glob("converge_*.json"))
    if not paths:
        raise SystemExit(f"결과가 없다 — 먼저 run.py를 돌릴 것 ({RESULT_DIR})")
    rows = [r for p in paths for r in json.loads(p.read_text())["rows"]]
    n_train = json.loads(paths[0].read_text())["n_train"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # 왼쪽: 이번 실행의 모든 arm. seed는 같은 선 위에 평균낸다.
    grouped: dict[tuple, list] = {}
    for r in rows:
        grouped.setdefault((r["kind"], r["grid_n"], r["lr"]), []).append(r["curve"])
    for (kind, grid_n, lr), curves in sorted(grouped.items()):
        steps = [p["step"] for p in curves[0]]
        mean = [
            sum(c[i]["cos"] for c in curves) / len(curves) for i in range(len(steps))
        ]
        spread = [
            max(c[i]["cos"] for c in curves) - min(c[i]["cos"] for c in curves)
            for i in range(len(steps))
        ]
        ax0.errorbar(
            steps,
            mean,
            yerr=[s / 2 for s in spread],
            color=COLORS.get(kind, "k"),
            linestyle=STYLES.get(lr, "-"),
            marker="o",
            markersize=3,
            linewidth=1.4,
            capsize=2,
            label=f"{kind} {grid_n}³ lr={lr:g} (n={len(curves)})",
        )
        best = max(range(len(mean)), key=lambda i: mean[i])
        ax0.plot(
            steps[best],
            mean[best],
            "*",
            color=COLORS.get(kind, "k"),
            markersize=13,
            markeredgecolor="k",
            markeredgewidth=0.6,
        )
    ax0.set_xscale("log")
    ax0.set_xlabel("projector fit steps (log)")
    ax0.set_ylabel("held-out token-wise cosine")
    ax0.set_title(
        f"U4b — reachable alignment vs fit steps\n(train {n_train} volumes; ★ = peak)",
        fontsize=10,
    )
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=7, loc="lower right")

    # 오른쪽: 왜 U4의 1500-step 한 점이 위험했는가.
    ax1.plot(
        OVERFIT_48["steps"],
        OVERFIT_48["cos"],
        "o-",
        color="#d62728",
        label="train 32 volumes (U4 protocol) — peaks then decays",
    )
    best_key = max(grouped, key=lambda k: max(p["cos"] for p in grouped[k][0]))
    c = grouped[best_key][0]
    ax1.plot(
        [p["step"] for p in c],
        [p["cos"] for p in c],
        "o-",
        color="#1f77b4",
        label=f"train {n_train} volumes ({best_key[0]} {best_key[1]}³ lr={best_key[2]:g}) — plateaus",
    )
    ax1.axvline(1500, color="k", linestyle=":", linewidth=1)
    ax1.annotate(
        "U4 stopped here", xy=(1500, 0.53), fontsize=8, rotation=90, va="bottom"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("projector fit steps (log)")
    ax1.set_ylabel("held-out token-wise cosine")
    ax1.set_title(
        "Why the U4 single-point read was unsafe\n(fit-set size, not step count, was the limit)",
        fontsize=10,
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "U4b_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 표: arm별 정점 (seed 평균 ± 범위)
    print(f"{'arm':28s} {'peak cos':>18s} {'peak step':>10s}")
    for key, curves in sorted(
        grouped.items(),
        key=lambda kv: (
            -max(
                sum(c[i]["cos"] for c in kv[1]) / len(kv[1])
                for i in range(len(kv[1][0]))
            )
        ),
    ):
        peaks = [max(p["cos"] for p in c) for c in curves]
        at = [max(c, key=lambda p: p["cos"])["step"] for c in curves]
        lo, hi = min(peaks), max(peaks)
        print(
            f"{key[0]} {key[1]}³ lr={key[2]:<8g} "
            f"{sum(peaks) / len(peaks):+.4f} [{lo:.4f},{hi:.4f}] {sorted(at)[len(at) // 2]:>10d}"
        )
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
