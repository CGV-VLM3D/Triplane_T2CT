"""Probe A (content) + Probe B (image-predictability).

frozen text 임베딩에 linear probe -> train으로 학습, valid_v2에서 평가
raw와 PCA-256 두 버전 함께 평가

Probe A는 CLIP 논문 참고함: Radford et al. 2021 CLIP linear-probe appendix (sklearn LogisticRegression,
lbfgs, max_iter=1000, L2 C sweep).
Probe B는 RidgeCV(alpha sweep).

Probe A: text → LogisticRegressionCV(per-label) → valid_v2 macro-AUROC / macro-F1 (대조군).
 - Binary classification
Probe B: text → RidgeCV → img_feat(분류기 penultimate, semantic) → R² / Pearson r.
 - RidgeCV = ridge regression with built-in cross-validation
 - loss = MSE + \alpha*{ridge regularization}

사용: python -m tests.alignment_probe.probes.linear_probe
출력: results/probe.json + stdout 표.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import multilabel_confusion_matrix, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from tests.alignment_probe.metric.metrics import (
    macro_f1,
    mean_pearson,
    per_sample_pearson,
    r2_dim_distribution,
    retrieval_metrics,
)
from tests.alignment_probe.utils.cases import ABNORMALITY_LABELS
from tests.alignment_probe.utils.io import (
    aligned,
    emb_dict,
    encoders,
    label_dict,
    img_feat_dict,
)

_RESULTS = Path(
    "tests/alignment_probe/results"
)  # standalone 기본; eval 스테이지는 results/<exp_name>
_SEED = 42
_CS = 10  # LogisticRegressionCV: 10개 log-spaced C
_ALPHAS = np.logspace(-3, 5, 20)  # RidgeCV alpha sweep


def _maybe_pca(X_train, X_test, variant):
    """variant가 'pca256'이고 차원이 256 초과이면 PCA-256으로 축소해 반환한다.

    Args:
        Xtr: 학습 임베딩 행렬. ``(N_train, D)``
        Xte: 평가 임베딩 행렬. ``(N_test, D)``
        variant: 'raw' 또는 'pca256'. 'raw'이면 입력 그대로 반환.

    Returns:
        (Xtr_out, Xte_out) — variant=='pca256'이면 ``(N_train, 256)``, ``(N_test, 256)``;
        그 외에는 입력과 동일한 shape.
    """
    if variant == "pca256" and X_train.shape[1] > 256:
        pca = PCA(n_components=256, random_state=_SEED).fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    return X_train, X_test


def _probe_a_one(enc: str, variant: str) -> tuple[dict, dict]:
    """Probe A: per-label LogisticRegressionCV, valid_v2 macro-AUROC / macro-F1.

    train text embedding으로 18 multi-label classifier를 각각 학습, valid_v2에서 평가
    AUROC를 계산할 수 없는 라벨(단일 클래스)은 자동 제외한다.

    Args:
        enc: 인코더 이름 (예: 'ctclip', 'text2ct').
        variant: 'raw' 또는 'pca256' — _maybe_pca에 전달.

    Returns:
        (metrics, confusion) 튜플.
          metrics: dim / mean_auc / macro_f1 / per_label_auc(라벨명→AUROC) /
            n_labels / n_eval (probe.json 용).
          confusion: {"labels": [유효 라벨명], "matrix": (n_labels, 2, 2) 정수 리스트}
            — 각 행렬은 sklearn 규약 [[TN, FP], [FN, TP]] (probe_a_confusion.json 용).

    Note:
        Xtr ``(N_train, D)``, ytr ``(N_train, 18)`` — 18종 abnormality 이진 라벨.
        Xte ``(N_test, D)``, yte ``(N_test, 18)``.
    """
    Xtr, ytr = aligned(emb_dict(enc, "train"), label_dict("train"))
    Xte, yte = aligned(emb_dict(enc, "valid_v2"), label_dict("valid_v2"))
    Xtr, Xte = _maybe_pca(Xtr, Xte, variant)
    sc = StandardScaler().fit(Xtr)  # Xtr 전체 평균, 분산
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)  # 정규화

    aucs, preds, valid = [], [], []
    for j in range(18):
        if len(np.unique(ytr[:, j])) < 2:  # train에 한 클래스만 → 학습 불가
            continue
        with warnings.catch_warnings():  # 무해한 sklearn 경고만 국소 억제
            warnings.simplefilter("ignore", FutureWarning)
            # warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegressionCV(
                Cs=_CS,  # 정규화 강도를 자동 탐색, CS는 후보 10개를 넘겨주는 hyperparam, C는 정규화 역수 (1/lambda = C)
                cv=3,  # 3-fold cross validation
                scoring="roc_auc",  # C 선택의 평가지표
                max_iter=5000,  # 1000으로 하면 수렴 안되는 오류 발생
                n_jobs=32,  # cpu 병렬 사용 개수
                random_state=_SEED,
            ).fit(Xtr, ytr[:, j])  # 해당 class에 대한 binary classification 모델 학습
        score = clf.predict_proba(Xte)[:, 1]  # positive probability, (N,)
        if len(np.unique(yte[:, j])) == 2:  # 양/음 둘 다 있어야 AUROC 정의됨
            aucs.append(roc_auc_score(yte[:, j], score))
            preds.append((score > 0.5).astype(int))  # 0.5 이상은 true로 pred 정의
            valid.append(j)
    P = (
        np.stack(preds, axis=1) if valid else np.empty((len(yte), 0), dtype=int)
    )  # pred (N,), P: [(N,),(N,)...]
    f1 = macro_f1(yte[:, valid], P) if valid else float("nan")
    # 유효 라벨별 2×2 혼동행렬 (n_labels, 2, 2): sklearn 규약 [[TN, FP], [FN, TP]]
    cms = (
        multilabel_confusion_matrix(yte[:, valid], P)
        if valid
        else np.empty((0, 2, 2), dtype=int)
    )
    # valid·aucs는 같은 순서의 병렬 리스트 → 라벨명→AUROC dict (per-label 히트맵용)
    per_label_auc = {ABNORMALITY_LABELS[j]: float(a) for j, a in zip(valid, aucs)}
    metrics = {
        "dim": int(Xtr.shape[1]),
        "mean_auc": float(np.mean(aucs)),  # macro auc
        "macro_f1": float(f1),
        "per_label_auc": per_label_auc,
        "n_labels": len(valid),
        "n_eval": int(len(yte)),
    }
    confusion = {
        "labels": [ABNORMALITY_LABELS[j] for j in valid],
        "matrix": cms.tolist(),
    }
    return metrics, confusion


def _probe_b_one(enc: str, ztr: dict, zte: dict) -> dict:
    """Probe B: RidgeCV로 text → img_feat(semantic) regression, train→valid_v2 cross-split.

    img_feat의 scale이 차원마다 다름, 따라서 target을 per-dim standardization(train stat으로)
    R²(variance-weighted) + 차원별 Pearson r로 본다(표준화가 mean-centering을 포함).

    Args:
        enc: 인코더 이름.
        ztr: train split scan_id → img_feat 벡터 dict. 각 벡터 ``(D_z,)``.
        zte: valid_v2 split scan_id → img_feat 벡터 dict. 각 벡터 ``(D_z,)``.

    Returns:
        dim: 학습에 사용된 text 임베딩 차원.
        r2: variance-weighted R² (float).
        pearson: 차원별 Pearson r 평균 (float).
        per_sample_pearson: 샘플별 프로파일 Pearson r 평균 (float).
        n_eval: valid_v2 평가 샘플 수.
        r2_dim_frac_pos / r2_dim_median / r2_dim_p25 / r2_dim_p75: 차원별 R² 분포 요약.
        recall@1/5/10, median_rank, median_rank_pct, chance_median_rank,
        perm_recall@1, perm_median_rank: cosine retrieval 지표 (retrieval_metrics 참고).

    Note:
        Xtr ``(N_train, D_text)``, Ztr ``(N_train, D_z)`` — train 기준 표준화 후 Ridge 학습.
        Xte ``(N_test, D_text)``, Zte ``(N_test, D_z)`` — valid_v2 cross-split 평가.
    """
    Xtr, Ztr = aligned(emb_dict(enc, "train"), ztr)
    Xte, Zte = aligned(emb_dict(enc, "valid_v2"), zte)
    xsc = StandardScaler().fit(Xtr)  # target per-dim 표준화(train 통계)
    zsc = StandardScaler().fit(Ztr)  # target per-dim 표준화(train 통계)
    ridge = RidgeCV(alphas=_ALPHAS).fit(
        xsc.transform(Xtr), zsc.transform(Ztr)
    )  # norm 시켜서 입력
    pred = ridge.predict(xsc.transform(Xte))  # (N_test, D_z)
    true = zsc.transform(Zte)  # (N_test, D_z)
    return {
        "dim": int(Xtr.shape[1]),
        "r2": float(r2_score(true, pred, multioutput="variance_weighted")),
        "pearson": mean_pearson(true, pred),
        "per_sample_pearson": per_sample_pearson(true, pred),
        "n_eval": int(len(Zte)),
        **r2_dim_distribution(true, pred),  # 차원별 R² 분포 요약
        **retrieval_metrics(pred, true),  # cosine retrieval + permutation null
    }


def main(out_dir: Path = _RESULTS) -> None:
    """Probe A / Probe B를 전체 인코더에 대해 실행, ``out_dir/probe.json``에 저장한다."""
    encs = encoders()
    print(f"encoders: {encs}\n")
    results: dict = {"probe_a": {}, "probe_b": {}}
    confusion: dict = {}  # enc -> variant -> {labels, matrix}; 별도 파일로 저장

    print("=== Probe A (text → 18-label) — valid_v2 macro-AUROC / macro-F1 ===")
    for enc in encs:
        results["probe_a"][enc], confusion[enc] = {}, {}
        for v in ("raw", "pca256"):
            results["probe_a"][enc][v], confusion[enc][v] = _probe_a_one(enc, v)
        r, p = results["probe_a"][enc]["raw"], results["probe_a"][enc]["pca256"]
        print(
            f"  {enc:10s} raw(d={r['dim']:4d}) AUC={r['mean_auc']:.3f} F1={r['macro_f1']:.3f}"
            f"  | pca256 AUC={p['mean_auc']:.3f} F1={p['macro_f1']:.3f}"
        )

    ztr, zte = img_feat_dict("train"), img_feat_dict("valid_v2")
    if ztr and zte:
        print(
            "\n=== Probe B (text → img_feat semantic, train→valid_v2) — R² / Pearson r ==="
        )
        for enc in encs:
            results["probe_b"][enc] = _probe_b_one(enc, ztr, zte)
            b = results["probe_b"][enc]
            print(
                f"  {enc:10s} R²={b['r2']:+.3f}  r={b['pearson']:+.3f}"
                f"  | R@1={b['recall@1']:.3f} R@5={b['recall@5']:.3f} R@10={b['recall@10']:.3f}"
                f"  medR={b['median_rank']:.0f}/{b['n_eval']}"
                f"  (chance={b['chance_median_rank']:.0f}, perm R@1={b['perm_recall@1']:.3f} medR={b['perm_median_rank']:.0f})"
            )
            print(
                f"  {'':10s} per-sample r={b['per_sample_pearson']:+.3f}"
                f"  | R²/dim frac_pos={b['r2_dim_frac_pos']:.3f} median={b['r2_dim_median']:+.3f}"
                f" [p25={b['r2_dim_p25']:+.3f}, p75={b['r2_dim_p75']:+.3f}]"
            )
    else:
        print("\n[Probe B] img_feat(img_feat.npz) 미존재 — train 먼저 실행.")

    out = out_dir / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    conf_out = out_dir / "probe_a_confusion.json"
    conf_out.write_text(json.dumps(confusion, indent=2))
    print(f"\nsaved → {out}\nsaved → {conf_out}")


if __name__ == "__main__":
    main()
