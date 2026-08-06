"""공유 metric 함수 — train(Probe C mean-AUC)·probe(A/B) 공용. sklearn/numpy 기반."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, r2_score, roc_auc_score

from tests.alignment_probe.utils.cases import ABNORMALITY_LABELS


def mean_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """라벨별 ROC-AUC의 평균(단일 클래스 라벨 제외) → (mean, {label: auc}).

    Args:
        y_true: 이진 정답 레이블. ``(N, num_labels)``
        y_score: 클래스 양성 확률(또는 logit). ``(N, num_labels)``

    Returns:
        (mean_auc, per_label_dict) — 유효 라벨 AUC의 산술 평균과 라벨별 AUC 딕셔너리.
    """
    per = {}
    for j, name in enumerate(ABNORMALITY_LABELS):
        if len(np.unique(y_true[:, j])) < 2:  # 모두 음성 케이스 같은건 넘어가야함
            continue
        per[name] = roc_auc_score(y_true[:, j], y_score[:, j])
    return float(np.mean(list(per.values()))), per  # macro


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """macro-F1 (sklearn, zero_division=0).

    Args:
        y_true: 이진 정답 레이블. ``(N, num_labels)``
        y_pred: 이진 예측값. ``(N, num_labels)``

    Returns:
        macro F1 score
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def retrieval_metrics(
    pred: np.ndarray, true: np.ndarray, n_perm: int = 20, seed: int = 42
) -> dict:
    """cross-split cosine retrieval: text embedding -> image feature retrieval.

    pred_i(텍스트로 예측한 image feature)를 query, {true_j}(실제 image feature)를 gallery로 두고
    cosine 유사도로 정렬한 뒤 정답 true_i의 등수를 본다. permutation test는 짝을 무작위로 섞었을 때의
    같은 지표(경험적 null)이며, 실제 등수가 이 null과 analytic chance보다 확실히 낮아야 정렬이 의미 있다.

    Args:
        pred: 예측 image feature. ``(N, D)``
        true: 실제 image feature. ``(N, D)``
        n_perm: permutation test 반복 횟수.
        seed: permutation 시드.

    Returns:
        recall@1/5/10 (정답 등수 ≤ k 인 샘플의 비율), median_rank, median_rank_pct (median_rank/N),
        chance_median_rank ((N+1)/2 analytic null), perm_recall@1 / perm_median_rank
        (n_perm회 평균 경험적 null).
    """
    n = pred.shape[0]
    p = pred / (
        np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12
    )  # (N, D) 행 단위 L2 정규화
    t = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-12)  # (N, D)
    sim = p @ t.T  # (N, N) cosine 유사도, sim[i,j] = cos(pred_i, true_j)
    diag = np.diag(sim)  # (N,) 정답쌍 유사도 sim[i,i]
    ranks = 1 + (sim > diag[:, None]).sum(axis=1)  # (N,) 정답보다 높은 후보 수 + 1
    out = {
        "recall@1": float(np.mean(ranks <= 1)), 
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "median_rank_pct": float(np.median(ranks) / n),
        "chance_median_rank": float((n + 1) / 2),
    }
    rng = np.random.default_rng(seed)
    pr1, pmr = [], []
    for _ in range(n_perm):
        perm = rng.permutation(n)  # 무작위 짝
        corr = sim[np.arange(n), perm]  # (N,) 섞인 "정답" 유사도
        r = 1 + (sim > corr[:, None]).sum(axis=1)  # (N,)
        pr1.append(np.mean(r <= 1))
        pmr.append(np.median(r))
    out["perm_recall@1"] = float(np.mean(pr1))
    out["perm_median_rank"] = float(np.mean(pmr))
    return out


def mean_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    """target 차원별 Pearson r의 평균(상수 차원 제외).

    Args:
        true: 정답 연속값 배열. ``(N, D)`` — D = target 차원 수.
        pred: 예측 연속값 배열. ``(N, D)``

    Returns:
        유효 차원(표준편차 > 1e-8) Pearson r의 산술 평균. 유효 차원 없으면 ``nan``.
    """
    rs = [
        np.corrcoef(true[:, j], pred[:, j])[0, 1]  # 각 클래스별 피어슨 계수
        for j in range(true.shape[1])  # D 수만큼 iter
        if true[:, j].std() > 1e-8 and pred[:, j].std() > 1e-8
    ]
    return float(np.mean(rs)) if rs else float("nan")


def per_sample_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    """샘플별로 D 차원에 걸친 Pearson r의 평균(상수 샘플 제외).

    mean_pearson은 차원축으로 보지만(각 dim이 샘플들 사이에서 맞나), 이건 샘플축으로 본다:
    한 샘플의 D차원 feature 프로파일이 정답 프로파일과 모양이 맞나.

    Args:
        true: 정답 연속값 배열. ``(N, D)``
        pred: 예측 연속값 배열. ``(N, D)``

    Returns:
        유효 샘플(분모 > 1e-12) Pearson r의 산술 평균. 유효 샘플 없으면 ``nan``.
    """
    t = true - true.mean(axis=1, keepdims=True)  # (N, D) 샘플별 mean-center
    p = pred - pred.mean(axis=1, keepdims=True)  # (N, D)
    num = (t * p).sum(axis=1)  # (N,) 공분산 분자
    den = np.sqrt((t**2).sum(axis=1) * (p**2).sum(axis=1))  # (N,) 표준편차 곱
    mask = den > 1e-12
    return float(np.mean(num[mask] / den[mask])) if mask.any() else float("nan")


def r2_dim_distribution(true: np.ndarray, pred: np.ndarray) -> dict:
    """target 차원별 R²의 분포 요약 — 평균 하나가 숨기는 "소수 dim만 캐리" 여부를 본다.

    Args:
        true: 정답 연속값 배열. ``(N, D)``
        pred: 예측 연속값 배열. ``(N, D)``

    Returns:
        r2_dim_frac_pos (R²>0 차원 비율), r2_dim_median, r2_dim_p25, r2_dim_p75.
    """
    r2 = r2_score(true, pred, multioutput="raw_values")  # (D,) 차원별 R²
    return {
        "r2_dim_frac_pos": float(np.mean(r2 > 0)),
        "r2_dim_median": float(np.median(r2)),
        "r2_dim_p25": float(np.percentile(r2, 25)),
        "r2_dim_p75": float(np.percentile(r2, 75)),
    }
