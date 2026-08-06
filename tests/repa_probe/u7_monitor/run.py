"""U7 monitor — 학습 중 정렬이 실제로 일어나는지 / grad가 예상대로 흐르는지 그림으로 확인.

계획의 sanity-check 층 **S7**. wandb 로컬 run 파일에서 지표를 읽어 4개 패널을 그린다:

  1. `loss_diff` vs `loss` — 정렬 항이 denoising을 갉아먹고 있지 않은가
     (REPA의 주장은 둘 다 좋아진다는 것이지 denoising을 파는 게 아니다 → 나빠지면 λ 과대)
  2. `repa_cos` **와 `repa_cos_shuffled`를 겹쳐서** — 둘의 격차가 볼륨 고유 정렬량이다.
     U5에서 확인했듯 raw cosine은 해부학적 위치 prior만으로 0.45까지 오른다.
  3. `repa_rel` — 관계형 손실
  4. `grad_norm` — 폭주/소실 감시

실행:
    python -m tests.repa_probe.u7_monitor.run [--run outputs/report2ct_wan_repa/<dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figs"
RESULT_DIR = OUT_DIR / "results"

KEYS = (
    "train/loss_step",
    "train/loss_diff_step",
    "train/loss_repa_step",
    "train/repa_cos_step",
    "train/repa_cos_shuffled_step",
    "train/repa_cos_gap_step",
    "train/repa_rel_step",
    "train/grad_norm",
)


def latest_run(root: Path) -> Path:
    """가장 최근에 수정된 wandb run 디렉토리."""
    runs = sorted(root.glob("*/wandb/run-*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"{root} 아래에 wandb run이 없다")
    return runs[-1]


def read_history(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """`.wandb` 바이너리를 wandb SDK로 파싱해 스칼라 히스토리를 뽑는다."""
    from wandb.sdk.lib.runid import generate_id  # noqa: F401, PLC0415  (SDK 존재 확인)
    from wandb.proto import wandb_internal_pb2  # noqa: PLC0415
    from wandb.sdk.internal import datastore  # noqa: PLC0415

    wandb_file = next(run_dir.glob("*.wandb"))
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    out: dict[str, list[tuple[int, float]]] = {k: [] for k in KEYS}
    step = 0
    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        # 계층 키("train/loss_step")는 `key`가 비어 있고 `nested_key`에 조각으로 들어온다.
        row = {
            (item.key or "/".join(item.nested_key)): item.value_json
            for item in record.history.item
        }
        if "_step" in row:
            step = int(json.loads(row["_step"]))
        for key in KEYS:
            if key in row:
                try:
                    out[key].append((step, float(json.loads(row[key]))))
                except (ValueError, TypeError):
                    pass
    return {k: v for k, v in out.items() if v}


def plot(history: dict[str, list[tuple[int, float]]], title: str) -> Path:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def series(key):
        pts = history.get(key, [])
        return [p[0] for p in pts], [p[1] for p in pts]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    for key, label in (
        ("train/loss_diff_step", "diffusion only"),
        ("train/loss_step", "total"),
    ):
        x, y = series(key)
        if x:
            ax.plot(x, y, label=label, lw=0.8)
    ax.set_title(
        "loss: diffusion vs total\n(diffusion이 나빠지면 lambda 과대)", fontsize=9
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for key, label in (
        ("train/repa_cos_step", "real teacher"),
        ("train/repa_cos_shuffled_step", "shuffled (position prior)"),
    ):
        x, y = series(key)
        if x:
            ax.plot(x, y, label=label, lw=0.8)
    xg, yg = series("train/repa_cos_gap_step")
    if xg:
        ax.plot(xg, yg, label="gap = real - shuffled", lw=1.2, color="k")
    ax.set_title("alignment cosine\n(gap이 볼륨 고유 정렬량)", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    x, y = series("train/repa_rel_step")
    if x:
        ax.plot(x, y, lw=0.8, color="tab:green")
    ax.set_title("relational loss (하강해야 함)", fontsize=9)

    ax = axes[1, 1]
    x, y = series("train/grad_norm")
    if x:
        ax.plot(x, y, lw=0.8, color="tab:red")
    ax.set_yscale("log")
    ax.set_title("grad norm (폭주/소실 감시)", fontsize=9)

    for a in axes.ravel():
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "U7_training_monitor.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/report2ct_wan_repa")
    ap.add_argument("--run", default=None, help="wandb run 디렉토리 (없으면 가장 최근)")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else latest_run(Path(args.root))
    history = read_history(run_dir)
    if not history:
        raise SystemExit(f"{run_dir}에서 지표를 읽지 못했다")

    last_step = max(pts[-1][0] for pts in history.values())
    summary = {
        "run_dir": str(run_dir),
        "last_step": last_step,
        "metrics": {
            k: {"n": len(v), "first": v[0][1], "last": v[-1][1]}
            for k, v in history.items()
        },
    }
    fig = plot(history, f"{run_dir.name} — step {last_step}")
    summary["figure"] = str(fig.relative_to(Path("/workspace")))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "monitor.json").write_text(json.dumps(summary, indent=2))
    print(f"{'metric':32s} {'n':>6s} {'first':>10s} {'last':>10s}")
    for k, v in summary["metrics"].items():
        print(f"{k:32s} {v['n']:6d} {v['first']:10.4f} {v['last']:10.4f}")
    print(f"\n[done] {fig}")


if __name__ == "__main__":
    main()
