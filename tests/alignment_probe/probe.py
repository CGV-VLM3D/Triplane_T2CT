"""Unit 3+4 — Probe A (content) + Probe B (image-predictability).

frozen text 임베딩(Unit 1b)에 **표준 선형 probe**를 얹어 학습하고 valid_v2에서 평가한다.
모든 인코더 동일 프로토콜 → 공정 비교. 차원 d는 head가 흡수하되, 용량 교란을 위해
raw와 PCA-256(공통 차원) 두 버전을 함께 본다.

표준 HP 출처: Radford et al. 2021 CLIP linear-probe appendix (sklearn LogisticRegression,
lbfgs, max_iter=1000, L2 C sweep). Probe B 회귀는 RidgeCV(alpha sweep) — 선형 회귀 probe 표준.

Probe A: text → LogisticRegressionCV(per-label) → proxy mean ROC-AUC / macro-F1.
         metric은 공식 abnclass 평가(third_party/ct_clip/scripts/eval.py, per-label AUROC)와
         **동일 함수(sklearn roc_auc_score)**. 대조군: 다 높게 나와야 정상.
Probe B: text → RidgeCV → z②(분류기 penultimate, semantic). proxy cosine / R².
         가설: T5 < CT-CLIP < fVLM.
(Probe C = z② 분류기 자신의 proxy AUC = classifier.py에서 보고; Probe B의 천장.)

사용: python -m tests.alignment_probe.probe   (CPU; 임베딩이 작음)
출력: tests/alignment_probe/results/probe.json + stdout 표.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tests.alignment_probe.cases import ABNORMALITY_LABELS, load_cases

_EMB_ROOT = Path("tests/alignment_probe/embeddings")
_OUT = Path("tests/alignment_probe/results/probe.json")
_SEED = 0
_CS = 10  # LogisticRegressionCV: 10개 log-spaced C
_ALPHAS = np.logspace(-3, 3, 13)  # RidgeCV alpha sweep


def _encoders() -> list[str]:
    """train과 valid_v2 양쪽에 임베딩이 있는 인코더만(z_classifier 제외)."""
    here = lambda split: {  # noqa: E731
        p.stem for p in (_EMB_ROOT / split).glob("*.npz") if p.stem != "z_classifier"
    }
    return sorted(here("train") & here("valid_v2"))


def _emb_dict(encoder: str, split: str) -> dict[str, np.ndarray]:
    z = np.load(_EMB_ROOT / split / f"{encoder}.npz", allow_pickle=True)
    return dict(zip(z["scan_ids"], z["emb"]))


def _label_dict(split: str) -> dict[str, np.ndarray]:
    return {
        c.scan_id: np.array([c.labels[a] for a in ABNORMALITY_LABELS], dtype=np.float32)
        for c in load_cases(split)
    }


def _z_dict(split: str) -> dict[str, np.ndarray] | None:
    p = _EMB_ROOT / split / "z_classifier.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return dict(zip(z["scan_ids"], z["emb"]))


def _aligned(x_dict: dict, y_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """공통 scan_id(정렬)에 대해 (X, Y) 행렬 구성."""
    ids = sorted(set(x_dict) & set(y_dict))
    return np.stack([x_dict[i] for i in ids]), np.stack([y_dict[i] for i in ids])


def _maybe_pca(Xtr, Xte, variant):
    if variant == "pca256" and Xtr.shape[1] > 256:
        pca = PCA(n_components=256, random_state=_SEED).fit(Xtr)
        return pca.transform(Xtr), pca.transform(Xte)
    return Xtr, Xte


def _probe_a_one(enc: str, variant: str) -> dict:
    """Probe A: per-label LogisticRegressionCV, proxy mean-AUROC / macro-F1."""
    Xtr, ytr = _aligned(_emb_dict(enc, "train"), _label_dict("train"))
    Xte, yte = _aligned(_emb_dict(enc, "valid_v2"), _label_dict("valid_v2"))
    Xtr, Xte = _maybe_pca(Xtr, Xte, variant)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    aucs, preds, valid = [], [], []
    for j in range(18):
        if len(np.unique(ytr[:, j])) < 2:  # train에 한 클래스만 → 학습 불가
            continue
        clf = LogisticRegressionCV(
            Cs=_CS,
            cv=3,
            scoring="roc_auc",
            max_iter=1000,
            n_jobs=-1,
            random_state=_SEED,
        ).fit(Xtr, ytr[:, j])
        score = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(yte[:, j])) == 2:  # proxy에 양/음 둘 다 있어야 AUROC 정의됨
            aucs.append(roc_auc_score(yte[:, j], score))
            preds.append((score > 0.5).astype(int))
            valid.append(j)
    f1 = (
        f1_score(
            yte[:, valid], np.stack(preds, axis=1), average="macro", zero_division=0
        )
        if valid
        else float("nan")
    )
    return {
        "dim": int(Xtr.shape[1]),
        "mean_auc": float(np.mean(aucs)),
        "macro_f1": float(f1),
        "n_labels": len(valid),
        "n_eval": int(len(yte)),
    }


def _mean_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    """target 차원별 Pearson r의 평균(상수 차원 제외)."""
    rs = [
        np.corrcoef(true[:, j], pred[:, j])[0, 1]
        for j in range(true.shape[1])
        if true[:, j].std() > 1e-8 and pred[:, j].std() > 1e-8
    ]
    return float(np.mean(rs)) if rs else float("nan")


def _probe_b_one(enc: str, ztr: dict, zte: dict) -> dict:
    """Probe B: RidgeCV로 text → z②(semantic) 예측, train→proxy cross-split.

    train·proxy z②가 동일 latent 공간(둘 다 Report2CT `_emb.nii.gz`)이라 cross-split이 유효.
    z②는 ReLU feature(전부 비음수, 평균 pairwise cosine≈0.92)라 cosine은 비변별 → target을
    per-dim 표준화(train 통계)하고 R²(variance-weighted) + 차원별 Pearson r로 본다.
    """
    Xtr, Ztr = _aligned(_emb_dict(enc, "train"), ztr)
    Xte, Zte = _aligned(_emb_dict(enc, "valid_v2"), zte)
    xsc = StandardScaler().fit(Xtr)
    zsc = StandardScaler().fit(Ztr)  # target per-dim 표준화(train 통계)
    ridge = RidgeCV(alphas=_ALPHAS).fit(xsc.transform(Xtr), zsc.transform(Ztr))
    pred = ridge.predict(xsc.transform(Xte))
    true = zsc.transform(Zte)
    return {
        "dim": int(Xtr.shape[1]),
        "r2": float(r2_score(true, pred, multioutput="variance_weighted")),
        "pearson": _mean_pearson(true, pred),
        "n_eval": int(len(Zte)),
    }


def main() -> None:
    encoders = _encoders()
    print(f"encoders: {encoders}\n")
    results: dict = {"probe_a": {}, "probe_b": {}}

    print("=== Probe A (text → 18-label) — proxy mean-AUROC / macro-F1 ===")
    for enc in encoders:
        results["probe_a"][enc] = {v: _probe_a_one(enc, v) for v in ("raw", "pca256")}
        r, p = results["probe_a"][enc]["raw"], results["probe_a"][enc]["pca256"]
        print(
            f"  {enc:10s} raw(d={r['dim']:4d}) AUC={r['mean_auc']:.3f} F1={r['macro_f1']:.3f}"
            f"  | pca256 AUC={p['mean_auc']:.3f} F1={p['macro_f1']:.3f}"
        )

    ztr, zte = _z_dict("train"), _z_dict("valid_v2")
    if ztr and zte:
        print("\n=== Probe B (text → z② semantic, train→proxy) — R² / Pearson r ===")
        for enc in encoders:
            results["probe_b"][enc] = _probe_b_one(enc, ztr, zte)
            b = results["probe_b"][enc]
            print(
                f"  {enc:10s} R²={b['r2']:+.3f}  Pearson r={b['pearson']:+.3f}  (n={b['n_eval']})"
            )
    else:
        print("\n[Probe B] z②(z_classifier.npz) 미존재 — classifier.py 먼저 실행.")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {_OUT}")


if __name__ == "__main__":
    main()
