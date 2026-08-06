"""학습 산출물 저장 — loss/AUC 곡선·label별 AUC·confusion matrix·metrics JSON.

matplotlib는 이 모듈에서만 import(Agg, headless). 모든 산출물은 results/<exp_name>/ 하위에 저장.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import multilabel_confusion_matrix  # noqa: E402
from sklearn.metrics import roc_curve, auc as _auc
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tests.alignment_probe.utils.cases import ABNORMALITY_LABELS  # noqa: E402
from tests.alignment_probe.utils.io import (  # noqa: E402
    aligned,
    emb_dict,
    encoders,
    label_dict,
)

_TSNE_SEED = 42
_TSNE_PERPLEXITIES = (30,)  # (10, 30, 50)  # 국소→전역 robustness 확인용
_TSNE_TOPK = 3  # 색칠할 abnormality 수(valid_v2 양성 빈도 상위)


def _epochs(history: list[dict]) -> list[int]:
    """history에서 epoch 리스트를 추출한다."""
    return [h["epoch"] for h in history]


def save_loss_curve(history: list[dict], out_dir: Path) -> None:
    """epoch별 train_loss/valid 곡선을 loss_curve.png로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = _epochs(history)
    fig, ax = plt.subplots()
    ax.plot(eps, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(eps, [h["val_loss"] for h in history], marker="o", label="valid")
    ax.set(xlabel="epoch", ylabel="BCE loss", title="Train vs valid loss")
    ax.legend()
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, logits: np.ndarray, out_dir: Path) -> None:
    """라벨별 ROC + macro-average ROC를 roc_curve.png로 저장한다.

    Args:
        y_true: 이진 정답. ``(N, 18)``
        logits: 분류기 logit(ROC는 단조변환에 불변이라 sigmoid 불필요). ``(N, 18)``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    all_fpr = np.linspace(0, 1, 200)  # 공통 FPR 격자
    mean_tpr = np.zeros_like(all_fpr)
    n_valid = 0
    for j in range(y_true.shape[1]):  # class 수만큼 iter
        if len(np.unique(y_true[:, j])) < 2:  # 단일 클래스 라벨 skip
            continue
        fpr, tpr, _ = roc_curve(y_true[:, j], logits[:, j])  # 각 클래스별 roc 계산
        ax.plot(fpr, tpr, lw=0.8, alpha=0.4)
        mean_tpr += np.interp(
            all_fpr, fpr, tpr
        )  # 라벨별 곡선을 공통 격자로 보간해 누적
        n_valid += 1
    # macro-average: 라벨마다 동일 가중으로 평균 (mean_auc와 동일 기준)
    mean_tpr /= n_valid
    ax.plot(
        all_fpr,
        mean_tpr,
        lw=2.5,
        color="k",
        label=f"macro-average (AUC={_auc(all_fpr, mean_tpr):.3f})",
    )
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)  # chance 대각선
    ax.set(
        xlabel="FPR",
        ylabel="TPR",
        xlim=(0, 1),
        ylim=(0, 1),
        title="ROC (valid_v2, per-label + macro)",
    )
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)


def save_auc_curve(history: list[dict], out_dir: Path) -> None:
    """epoch별 valid_v2 mean-AUC 곡선을 auc_curve.png로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(_epochs(history), [h["val_mean_auc"] for h in history], marker="o")
    ax.set(xlabel="epoch", ylabel="valid_v2 mean-AUC", title="Validation mean-AUC")
    fig.savefig(out_dir / "auc_curve.png", bbox_inches="tight")
    plt.close(fig)


def save_per_label_auc(per_label: dict[str, float], out_dir: Path) -> None:
    """최종 18-label AUC 막대그래프(내림차순)를 per_label_auc.png로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(per_label.items(), key=lambda kv: kv[1])  # barh는 아래→위
    names, vals = [k for k, _ in items], [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set(xlabel="ROC-AUC", xlim=(0, 1), title="Per-label AUC (valid_v2, final)")
    fig.savefig(out_dir / "per_label_auc.png", bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices(
    y_true: np.ndarray, logits: np.ndarray, out_dir: Path
) -> None:
    """label별 2×2 confusion matrix 그리드를 confusion_matrices.png로 저장한다.

    Args:
        y_true: 이진 정답. ``(N, 18)``
        logits: 분류기 logit(임계 0 ⇔ prob 0.5로 이진화). ``(N, 18)``
        out_dir: 저장 디렉토리.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cms = multilabel_confusion_matrix(y_true, (logits > 0).astype(int))  # (18, 2, 2)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for j, ax in enumerate(axes.flat):
        if j >= len(ABNORMALITY_LABELS):
            ax.axis("off")
            continue
        ax.imshow(cms[j], cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cms[j][r, c]), ha="center", va="center")
        ax.set(
            title=ABNORMALITY_LABELS[j][:20],
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="pred",
            ylabel="true",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def mean_pairwise_cosine(z: np.ndarray) -> dict[str, float]:
    """img_feat의 평균 pairwise cosine(raw + mean-centered)을 계산한다.

    Args:
        z: feature 행렬. ``(N, D)``

    Returns:
        {"raw": float, "centered": float} — 상삼각(i<j) cosine 평균.
    """

    def _mpc(a: np.ndarray) -> float:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        s = a @ a.T  # (N, N)
        iu = np.triu_indices(a.shape[0], k=1)
        return float(s[iu].mean())

    return {"raw": _mpc(z), "centered": _mpc(z - z.mean(0, keepdims=True))}


def save_metrics_json(
    out_dir: Path,
    final_auc: float,
    per_label: dict[str, float],
    history: list[dict],
    cosine: dict[str, float],
) -> None:
    """최종 mean-AUC·label별 AUC·epoch history·img_feat cosine을 metrics.json으로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "final_mean_auc": final_auc,
        "per_label_auc": per_label,
        "img_feat_mean_pairwise_cosine": cosine,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


# ── eval probe 시각화 (probe.json / probe_a_confusion.json 기반) ──────────────


def save_probe_a_bar(probe_a: dict, out_dir: Path) -> None:
    """Probe A macro-AUROC / macro-F1 막대그래프(인코더별, raw vs pca256)를 probe_a_bar.png로 저장.

    Args:
        probe_a: {enc: {"raw": {...}, "pca256": {...}}} (probe.json["probe_a"]).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = list(probe_a)
    x = np.arange(len(encs))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in (
        (axes[0], "mean_auc", "Probe A macro-AUROC"),
        (axes[1], "macro_f1", "Probe A macro-F1"),
    ):
        raw = [probe_a[e]["raw"][key] for e in encs]
        pca = [probe_a[e]["pca256"][key] for e in encs]
        b_raw = ax.bar(x - w / 2, raw, w, label="raw")
        b_pca = ax.bar(x + w / 2, pca, w, label="pca256")
        for b in (b_raw, b_pca):  # 막대 상단 값 표기(3자리: 0.97↔0.98 구분)
            ax.bar_label(b, fmt="%.3f", padding=2, fontsize=7, rotation=90)
        ax.set(
            xticks=x, ylim=(0, 1.12), title=title
        )  # 라벨이 1 근처에서 안 잘리게 여유
        ax.set_xticklabels(encs, rotation=30, ha="right")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probe_a_bar.png", bbox_inches="tight")
    plt.close(fig)


def save_probe_b_bar(probe_b: dict, out_dir: Path) -> None:
    """Probe B R² / Pearson r 막대그래프(인코더별)를 probe_b_bar.png로 저장.

    Args:
        probe_b: {enc: {"r2": float, "pearson": float, ...}} (probe.json["probe_b"]).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = list(probe_b)
    x = np.arange(len(encs))
    w = 0.38
    r2 = [probe_b[e]["r2"] for e in encs]
    pr = [probe_b[e]["pearson"] for e in encs]
    fig, ax = plt.subplots(figsize=(8, 5))
    b_r2 = ax.bar(x - w / 2, r2, w, label="R²")
    b_pr = ax.bar(x + w / 2, pr, w, label="Pearson r")
    for b in (b_r2, b_pr):  # 막대 상단 값 표기
        ax.bar_label(b, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)  # R²는 음수 가능 → 0선 표시
    ax.set(xticks=x, title="Probe B (text → img_feat): R² / Pearson r")
    ax.set_xticklabels(encs, rotation=30, ha="right")
    ax.legend()
    fig.savefig(out_dir / "probe_b_bar.png", bbox_inches="tight")
    plt.close(fig)


def save_probe_radar(
    probe_a: dict, probe_b: dict, out_dir: Path, normalize: bool = False
) -> None:
    """인코더별 4지표(ProbeA AUROC/F1 raw, ProbeB R²/Pearson) radar를 저장한다.

    normalize=False: 반경 [0, 1] 절대 스케일(probe_radar.png) — higher=better 개요.
    normalize=True: 축별 min–max로 재스케일(probe_radar_minmax.png) — 인코더 간 차이를
      반지름 전체로 펼침(중심=그 축 최저, 바깥=최고). 절대값은 축 라벨의 [lo–hi]로 병기.
    probe_b가 비면 Probe A 두 축만 그린다.

    Args:
        probe_a: probe.json["probe_a"].
        probe_b: probe.json["probe_b"] (없으면 {}).
        normalize: 축별 min–max 정규화 여부.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = list(probe_a)
    has_b = bool(probe_b)
    names = ["A:macro-AUROC", "A:macro-F1"] + (["B:R²", "B:Pearson"] if has_b else [])

    def vec(e: str) -> list[float]:
        v = [probe_a[e]["raw"]["mean_auc"], probe_a[e]["raw"]["macro_f1"]]
        if has_b:
            v += [probe_b[e]["r2"], probe_b[e]["pearson"]]
        return v

    raw = np.array([vec(e) for e in encs])  # (E, M) 절대값
    if normalize:
        margin = 0.1  # min/max가 정중앙·바깥선에 딱 붙지 않게 양쪽 여백
        lo, hi = raw.min(0), raw.max(0)  # 축별 min/max (M,)
        span = np.where(hi > lo, hi - lo, 1.0)  # 동률 축 0division 방지
        plot = margin + (raw - lo) / span * (
            1 - 2 * margin
        )  # (E, M) ∈ [margin, 1-margin]
        plot[:, hi == lo] = 0.5  # 동률 축은 전부 중심에 붙으니 중간으로
        labels = [f"{n}\n[{l:.2f}–{h:.2f}]" for n, l, h in zip(names, lo, hi)]
        title = "Probe A/B — per-axis min–max (center=worst, edge=best; abs in […])"
        pad, ylabels = 20, []  # 반경은 상대값 → 숫자 라벨 숨김
    else:
        plot = raw
        labels, title, pad, ylabels = (
            names,
            "Probe A/B comparison (radial 0–1, higher=better)",
            14,
            None,
        )

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]  # 폴리곤 닫기
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw=dict(polar=True))
    for i, e in enumerate(encs):
        v = plot[i].tolist()
        v += v[:1]
        ax.plot(angles, v, marker="o", lw=1.5, label=e)
        ax.fill(angles, v, alpha=0.08)
    ax.set(xticks=angles[:-1], ylim=(0, 1))
    ax.set_xticklabels(labels)
    if ylabels is not None:
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis="x", pad=pad)  # 긴 축 라벨이 플롯/범례에 겹치지 않게 바깥으로
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
    fname = "probe_radar_minmax.png" if normalize else "probe_radar.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)


def save_probe_confusion(confusion: dict, out_dir: Path, variant: str = "raw") -> None:
    """인코더별 유효 라벨 2×2 confusion grid를 confusion_<enc>.png로 저장한다.

    Args:
        confusion: {enc: {variant: {"labels": [...], "matrix": (L, 2, 2)}}}
            (probe_a_confusion.json). 각 행렬은 sklearn 규약 [[TN, FP], [FN, TP]].
        variant: 'raw' 또는 'pca256' — 그릴 variant.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for enc, byvar in confusion.items():
        c = byvar.get(variant)
        if not c or not c["matrix"]:  # 유효 라벨 없으면 skip
            continue
        labels = c["labels"]
        mats = np.array(c["matrix"])  # (L, 2, 2), L은 라벨 수
        n, ncol = len(labels), 5
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False
        )
        for j, ax in enumerate(axes.flat):
            if j >= n:
                ax.axis("off")
                continue
            ax.imshow(mats[j], cmap="Blues")
            for r in range(2):  # row
                for cc in range(2):  # column
                    ax.text(cc, r, int(mats[j][r, cc]), ha="center", va="center")
            ax.set(
                title=labels[j][:20],
                xticks=[0, 1],
                yticks=[0, 1],
                xlabel="pred",
                ylabel="true",
            )
        fig.suptitle(f"Probe A confusion — {enc} ({variant})")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{enc}.png", bbox_inches="tight")
        plt.close(fig)


def save_probe_label_heatmap(
    probe_a: dict, out_dir: Path, variant: str = "raw"
) -> None:
    """라벨 × 인코더 per-label AUROC 히트맵을 probe_label_auc.png로 저장한다.

    macro 한 숫자(probe_a_bar/radar)를 라벨 단위로 분해해 인코더 우열을 한눈에 비교한다.
    행 = 18 abnormality(ABNORMALITY_LABELS 순), 열 = 인코더, 색 = AUROC([0.5, 1.0]).
    해당 인코더에서 단일 클래스로 빠진 라벨은 NaN(빈 칸).

    Args:
        probe_a: probe.json["probe_a"] — {enc: {variant: {"per_label_auc": {label: auc}}}}.
        variant: 'raw' 또는 'pca256'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = list(probe_a)
    labels = ABNORMALITY_LABELS
    M = np.full((len(labels), len(encs)), np.nan)  # (L, E) AUROC, 누락은 NaN
    for ci, e in enumerate(encs):
        pla = probe_a[e][variant].get("per_label_auc", {})
        for ri, lab in enumerate(labels):
            if lab in pla:
                M[ri, ci] = pla[lab]
    fig, ax = plt.subplots(figsize=(1.6 * len(encs) + 3, 0.45 * len(labels) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")  # 0.5=chance
    ax.set(xticks=np.arange(len(encs)), yticks=np.arange(len(labels)))
    ax.set_xticklabels(encs, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for ri in range(len(labels)):
        for ci in range(len(encs)):
            if not np.isnan(
                M[ri, ci]
            ):  # viridis: 낮으면 어두움→흰 글씨, 높으면 밝음→검은 글씨
                ax.text(
                    ci,
                    ri,
                    f"{M[ri, ci]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if M[ri, ci] < 0.85 else "k",
                    fontsize=8,
                )
    ax.set_title(f"Probe A per-label AUROC ({variant})")
    fig.colorbar(im, ax=ax, label="AUROC", fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "probe_label_auc.png", bbox_inches="tight")
    plt.close(fig)


# ── t-SNE 보조 시각화 (probe.json 비종속, valid_v2 임베딩 직접 로딩) ───────────


def _tsne_2d(X: np.ndarray, perplexity: int) -> np.ndarray:
    """한 임베딩 행렬을 표준화→PCA-50→2D t-SNE 좌표로 투영한다.

    sklearn TSNE 문서 권장("reduce to a reasonable amount (e.g. 50) ... suppress some noise
    and speed up ... pairwise distances")에 따라 고차원(512–2560)을 PCA-50으로 먼저 축소한다.
    단 여기서 실측상 속도 이점은 없었고(회당 ~38s는 PCA 유무 무관 — 병목이 pairwise distance가
    아니라 N=1304 경사하강 iteration), 남는 효과는 노이즈 억제다.

    Args:
        X: 텍스트 임베딩 행렬. ``(N, D)``
        perplexity: t-SNE perplexity(유효 이웃 수).

    Returns:
        2D 좌표. ``(N, 2)``
    """
    Xs = StandardScaler().fit_transform(X)  # (N, D) 차원별 표준화
    n_pca = min(50, Xs.shape[1])  # D<50인 인코더 방어
    Xp = PCA(n_components=n_pca, random_state=_TSNE_SEED).fit_transform(
        Xs
    )  # (N, n_pca)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",  # PCA 초기화 → 실행 간 안정적
        learning_rate="auto",
        random_state=_TSNE_SEED,
    ).fit_transform(Xp)  # (N, 2)


def save_tsne_grid(
    out_dir: Path,
    labels: list[str] | None = None,
    perplexities: tuple[int, ...] = _TSNE_PERPLEXITIES,
    top_k: int = _TSNE_TOPK,
) -> None:
    """인코더×perplexity t-SNE 격자를 abnormality별로 색칠해 tsne_<label>.png로 저장한다.

    valid_v2 텍스트 임베딩을 인코더 나란히(열) × perplexity(행)로 투영하고, 한 abnormality의
    present(빨강)/absent(회색)로 이진 색칠한다. 좌표는 (인코더, perplexity)에만 의존하므로
    인코더·perplexity당 1회만 계산하고 라벨별 색칠에 재사용한다.

    산출물은 out_dir/tsne/ 하위에 모은다(상위 디렉토리가 plot으로 붐비지 않게).

    Args:
        out_dir: eval 산출물 디렉토리(t-SNE PNG는 out_dir/tsne/에 저장).
        labels: 색칠할 abnormality 이름 목록(ABNORMALITY_LABELS 중). None/빈 리스트면
            valid_v2 양성 빈도 상위 top_k개를 자동 선택한다.
        perplexities: 행으로 비교할 perplexity 목록(robustness 확인).
        top_k: labels 미지정 시 자동 선택할 abnormality 수(양성 빈도 상위).
    """
    tsne_dir = out_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    encs = encoders()
    lab_dict = label_dict("valid_v2")  # scan_id -> (18,)

    # 인코더별로 (좌표[perp], 라벨행렬)을 준비. enc마다 자기 scan_id 순서로 self-consistent.
    coords: dict[str, dict[int, np.ndarray]] = {}
    Y_by_enc: dict[str, np.ndarray] = {}
    for enc in encs:
        X, Y = aligned(emb_dict(enc, "valid_v2"), lab_dict)  # (N, D), (N, 18)
        Y_by_enc[enc] = Y
        coords[enc] = {p: _tsne_2d(X, p) for p in perplexities}  # 각 (N, 2)

    prevalence = next(iter(Y_by_enc.values())).sum(0)  # (18,) 라벨별 양성 수
    if labels:  # config 지정 라벨 → 인덱스(미존재 라벨은 에러로 조기 차단)
        unknown = [a for a in labels if a not in ABNORMALITY_LABELS]
        if unknown:
            raise ValueError(f"unknown tsne_labels: {unknown}")
        sel = [ABNORMALITY_LABELS.index(a) for a in labels]
    else:  # 미지정: 양성 빈도 상위 top_k 자동
        sel = np.argsort(prevalence)[::-1][:top_k].tolist()

    nrow, ncol = len(perplexities), len(encs)
    for li in sel:
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False
        )
        for r, perp in enumerate(perplexities):
            for c, enc in enumerate(encs):
                ax = axes[r][c]
                xy = coords[enc][perp]  # (N, 2)
                pos = Y_by_enc[enc][:, li] > 0.5  # 양성 마스크 (N,)
                ax.scatter(
                    xy[~pos, 0],
                    xy[~pos, 1],
                    s=4,
                    c="lightgray",
                    alpha=0.4,
                    label="absent",
                )
                ax.scatter(
                    xy[pos, 0], xy[pos, 1], s=8, c="crimson", alpha=0.8, label="present"
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(enc, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"perplexity={perp}", fontsize=9)
        axes[0][-1].legend(loc="upper right", fontsize=7, markerscale=1.5)
        fig.suptitle(
            f"t-SNE (valid_v2 text emb) — colored by '{ABNORMALITY_LABELS[li]}' "
            f"(n_pos={int(prevalence[li])})"
        )
        fig.tight_layout()
        safe = ABNORMALITY_LABELS[li].replace(" ", "_").replace("/", "-")
        fig.savefig(tsne_dir / f"tsne_{safe}.png", bbox_inches="tight")
        plt.close(fig)


def render_probe_figures(out_dir: Path, tsne_labels: list[str] | None = None) -> None:
    """out_dir의 probe.json (+ probe_a_confusion.json)을 읽어 Probe 시각화 PNG를 생성한다.

    생성물: probe_a_bar.png · probe_b_bar.png · probe_radar.png ·
    probe_label_auc.png · confusion_<enc>.png · tsne/tsne_<label>.png.
    파일이 없으면 해당 그림만 건너뛴다.

    Args:
        tsne_labels: t-SNE 색칠 abnormality 목록(save_tsne_grid로 전달; None이면 자동 선택).
    """
    save_tsne_grid(
        out_dir, labels=tsne_labels
    )  # 보조: valid_v2 임베딩 직접 로딩(probe.json 비종속)
    probe = json.loads((out_dir / "probe.json").read_text())
    probe_a, probe_b = probe.get("probe_a", {}), probe.get("probe_b", {})
    if probe_a:
        save_probe_a_bar(probe_a, out_dir)
        save_probe_radar(probe_a, probe_b, out_dir, normalize=False)  # 절대 [0,1]
        save_probe_radar(probe_a, probe_b, out_dir, normalize=True)  # 축별 min–max
        save_probe_label_heatmap(probe_a, out_dir)
    if probe_b:
        save_probe_b_bar(probe_b, out_dir)
    conf_path = out_dir / "probe_a_confusion.json"
    if conf_path.exists():
        save_probe_confusion(json.loads(conf_path.read_text()), out_dir)
