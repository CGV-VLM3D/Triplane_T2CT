"""공간 구조(spatial self-similarity) 지표 — iREPA §4 / Appendix B 정의를 3D로 옮긴 것.

iREPA는 27개 2D 인코더에서 이 지표들이 gFID와 |r| > 0.85로 상관한다고 보고했다 (linear probing은
|r| = 0.26). 그 상관계수 자체는 2D ImageNet에서 얻은 값이라 3D CT로의 외삽은 **가정**이다 —
여기서는 "이 teacher에 공간 구조가 존재하는가 / spatial norm이 그걸 키우는가"를 같은 축에서
비교하는 용도로만 쓴다.

정의 (Appendix B):
  자기유사도 커널   K(t,t') = cos(x_t, x_t')
  LDS   = E[K | d < r] − E[K | d ≥ r],  d = 격자 위 Manhattan 거리, r = 축 길이/2
  CDS   = −slope,  correlogram g(δ) = E[K | d = δ] 를 δ = 1..Δ 에서 직선 적합
  SRSS  = E_anchor[ cos(anchor, pos) − cos(anchor, neg) ],  pos = 같은 장기 마스크 내부 근방,
          neg = 마스크 외부 (iREPA는 SAM2 마스크, 우리는 ts_seg 장기 마스크)
  RMSC  = sqrt( mean_t || x̂_t − x̄ ||² ),  x̂ = L2 정규화 토큰, x̄ = 그 평균
"""

from __future__ import annotations

import torch


def token_coords(grid: tuple[int, int, int], device: str = "cpu") -> torch.Tensor:
    """격자 좌표 `(H*W*D, 3)` — 토큰 flatten 순서(C-order, depth fastest)와 동일하게."""
    axes = [torch.arange(n, device=device) for n in grid]
    return (
        torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).float()
    )


def spatial_norm(
    tokens: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """iREPA spatial normalization — 토큰축에서 평균을 빼고 표준편차로 나눈다.

    `y = (x − γ·E[x]) / sqrt(Var[x] + ε)`, 기대값/분산은 **spatial(토큰) 축**에서 계산.
    teacher feature에만 적용한다 (iREPA Algorithm 1).

    Args:
        tokens: `(T, D)` 또는 `(B, T, D)`.

    Returns:
        같은 shape.
    """
    dim = -2  # 토큰 축
    mean = tokens.mean(dim=dim, keepdim=True)
    var = tokens.var(dim=dim, unbiased=False, keepdim=True)
    return (tokens - gamma * mean) / torch.sqrt(var + eps)


def _cos_and_dist(
    tokens: torch.Tensor, coords: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """부분 표본 `idx`에 대한 상삼각 (cosine, Manhattan 거리) 쌍을 만든다."""
    x = torch.nn.functional.normalize(tokens[idx].float(), dim=-1)
    k = x @ x.T  # (M, M)
    d = (coords[idx].unsqueeze(1) - coords[idx].unsqueeze(0)).abs().sum(-1)  # (M, M)
    iu = torch.triu_indices(len(idx), len(idx), offset=1, device=tokens.device)
    return k[iu[0], iu[1]], d[iu[0], iu[1]]


def lds(cos: torch.Tensor, dist: torch.Tensor, radius: float) -> float:
    """Local vs Distant Similarity — 가까운 쌍과 먼 쌍의 평균 유사도 차이 (클수록 구조적)."""
    near, far = dist < radius, dist >= radius
    if not near.any() or not far.any():
        return float("nan")
    return float(cos[near].mean() - cos[far].mean())


def cds(cos: torch.Tensor, dist: torch.Tensor, max_delta: int = 8) -> float:
    """Correlation Decay Slope — correlogram의 기울기 부호를 뒤집은 값 (클수록 빨리 감쇠)."""
    deltas, means = [], []
    for delta in range(1, max_delta + 1):
        sel = dist == delta
        if int(sel.sum()) >= 32:
            deltas.append(float(delta))
            means.append(float(cos[sel].mean()))
    if len(deltas) < 3:
        return float("nan")
    x = torch.tensor(deltas)
    y = torch.tensor(means)
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return float(-slope)


def rmsc(tokens: torch.Tensor) -> float:
    """RMS Spatial Contrast — 정규화 토큰이 평균 토큰에서 얼마나 흩어져 있는가."""
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    return float(((x - x.mean(dim=0, keepdim=True)) ** 2).sum(-1).mean().sqrt())


def srss(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    organ_mask: torch.Tensor,
    r_near: float = 4.0,
    n_anchor: int = 64,
    generator: torch.Generator | None = None,
) -> float:
    """Semantic-Region Self-Similarity — 같은 장기 내부 근방 vs 장기 외부의 유사도 차이.

    Args:
        tokens: `(T, D)`.
        coords: `(T, 3)` 격자 좌표.
        organ_mask: `(T,)` bool — 해당 장기에 속하는 토큰.
        r_near: positive로 인정할 anchor로부터의 Manhattan 반경.
        n_anchor: 표본 anchor 수.

    Returns:
        E_anchor[ mean cos(anchor, pos) − mean cos(anchor, neg) ]. 마스크/격자 대응이 깨져
        있으면 0 근처로 떨어진다.
    """
    inside = torch.nonzero(organ_mask).flatten()
    outside = torch.nonzero(~organ_mask).flatten()
    if len(inside) < 8 or len(outside) < 8:
        return float("nan")
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    sel = inside[
        torch.randperm(len(inside), generator=generator, device=inside.device)[
            :n_anchor
        ]
    ]
    neg_sel = outside[
        torch.randperm(len(outside), generator=generator, device=outside.device)[:512]
    ]

    vals = []
    for a in sel.tolist():
        d = (coords[inside] - coords[a]).abs().sum(-1)
        pos = inside[(d > 0) & (d <= r_near)]
        if len(pos) < 4:
            continue
        vals.append(float((x[pos] @ x[a]).mean() - (x[neg_sel] @ x[a]).mean()))
    return float(torch.tensor(vals).mean()) if vals else float("nan")


def srss_dm(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    organ_mask: torch.Tensor,
    r_near: float = 4.0,
    n_anchor: int = 64,
    generator: torch.Generator | None = None,
) -> float:
    """Distance-matched SRSS — positive와 negative를 **같은 거리 대역**에서 뽑는다.

    `srss`는 positive를 anchor 근방(`d <= r_near`)에서, negative를 **격자 전체**에서 뽑는다.
    그래서 의미가 전혀 없어도 표현이 **매끄럽기만 하면** 점수가 높게 나온다 — 가까운 토큰은
    자동으로 비슷하고 먼 토큰은 다르기 때문이다. CNN teacher(CT-FM)의 feature가 정확히 그
    형태라 이 교란이 실제로 순위를 바꾼다.

    여기서는 negative도 `d <= r_near`에서 뽑아 거리 효과를 제거한다. 남는 신호는 하나뿐이다:
    **장기 경계를 넘을 때 표현이 실제로 바뀌는가.** 그 대신 anchor가 경계 근처여야 negative가
    존재하므로, 이 지표는 자연히 경계 판별력을 재게 된다 — REPA teacher에게 필요한 바로 그 성질.

    Args:
        tokens: ``(T, D)``.
        coords: ``(T, 3)`` 격자 좌표.
        organ_mask: ``(T,)`` bool.
        r_near: positive·negative 공통 Manhattan 반경.

    Returns:
        E_anchor[ mean cos(anchor, 같은장기 근방) − mean cos(anchor, 다른장기 **근방**) ].
        순수하게 매끄럽기만 한 표현은 0 근처가 된다.
    """
    inside = torch.nonzero(organ_mask).flatten()
    outside = torch.nonzero(~organ_mask).flatten()
    if len(inside) < 8 or len(outside) < 8:
        return float("nan")
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    sel = inside[
        torch.randperm(len(inside), generator=generator, device=inside.device)[
            :n_anchor
        ]
    ]

    vals = []
    for a in sel.tolist():
        d_in = (coords[inside] - coords[a]).abs().sum(-1)
        d_out = (coords[outside] - coords[a]).abs().sum(-1)
        pos = inside[(d_in > 0) & (d_in <= r_near)]
        neg = outside[d_out <= r_near]  # ← 여기가 srss와 다른 유일한 지점
        if len(pos) < 4 or len(neg) < 4:
            continue  # 장기 깊숙한 anchor는 같은 대역에 negative가 없다 → 건너뛴다
        vals.append(float((x[pos] @ x[a]).mean() - (x[neg] @ x[a]).mean()))
    return float(torch.tensor(vals).mean()) if vals else float("nan")


def crop_seam(
    tokens: torch.Tensor,
    grid: tuple[int, int, int],
    crop_tokens: tuple[int, int, int] = (8, 8, 8),
) -> dict[str, float]:
    """crop 경계에서 유사도가 얼마나 끊기는지 — **거리 1인 이웃 쌍만** 보고 비교한다.

    SPECTRE의 self-attention은 crop(128×128×64) 내부에서만 일어나므로, 재조립한 전역 격자에서는
    crop 경계를 가로지르는 이웃끼리 유사도가 뚝 떨어질 수 있다(U0 그림에서 육안 확인됨).

    ⚠ 해석 주의: 이 seam은 **token-wise cosine 손실에는 영향이 없다**(각 student 토큰이 자기
    teacher 토큰과만 맞춰지므로). 오염되는 건 **관계형 손실**(Gram/manifold/TRD)뿐이고, 그건
    Gram에서 crop 경계를 가로지르는 쌍을 빼면 공짜로 제거된다.

    Returns:
        within — 같은 crop 안 이웃 쌍의 평균 cosine
        across — crop 경계를 가로지르는 이웃 쌍의 평균 cosine
        gap    — within − across (0에 가까울수록 seam 없음)
        drop   — gap / within, 상대적 낙폭
    """
    x = torch.nn.functional.normalize(tokens.float(), dim=-1).reshape(*grid, -1)
    within, across = [], []
    for axis, n_ct in enumerate(crop_tokens):
        a = x.narrow(axis, 0, grid[axis] - 1)
        b = x.narrow(axis, 1, grid[axis] - 1)
        cos = (a * b).sum(-1)  # 축을 따라 이웃한 쌍의 cosine
        # 위치 i의 쌍 (i, i+1)이 crop 경계를 넘는가 = (i+1) % crop_tokens == 0
        pos = torch.arange(grid[axis] - 1, device=tokens.device)
        is_across = ((pos + 1) % n_ct) == 0
        shape = [1, 1, 1]
        shape[axis] = -1
        m = is_across.reshape(shape).expand_as(cos)
        across.append(cos[m])
        within.append(cos[~m])
    w = float(torch.cat(within).mean())
    a_ = float(torch.cat(across).mean())
    return {
        "within": w,
        "across": a_,
        "gap": w - a_,
        "drop": (w - a_) / w if w else float("nan"),
    }


def all_metrics(
    tokens: torch.Tensor,
    grid: tuple[int, int, int],
    organ_masks: dict[str, torch.Tensor] | None = None,
    subset: torch.Tensor | None = None,
    n_sample: int = 3000,
    seed: int = 0,
    voxel_shape: tuple[int, int, int] = (512, 512, 256),
    mm_per_voxel: tuple[float, float, float] = (1.0, 1.0, 1.0),
    srss_radius_mm: float = 50.0,
) -> dict[str, float]:
    """LDS / CDS / RMSC / seam (+ 장기별 SRSS)을 한 번에 계산한다.

    Args:
        tokens: `(T, D)` — 이미 spatial norm 등 전처리가 끝난 상태.
        grid: `(H, W, D)` 토큰 격자.
        organ_masks: 이름 → `(T,)` bool. None이면 SRSS를 건너뛴다.
        subset: `(T,)` bool — LDS/CDS/RMSC/cos_mean을 이 토큰들로만 계산한다 (예: 몸통만).
            `seam_gap`은 격자 구조 지표라 항상 전체 격자에서 계산한다.
        n_sample: LDS/CDS용 토큰 부분 표본 수 (전체 쌍은 32768² = 1.07e9로 계산 불가).
        voxel_shape: teacher 격자가 덮는 원본 볼륨 크기 (토큰당 복셀 수 환산용).
        mm_per_voxel: 그 프레임 한 복셀의 물리 크기(mm). **프레임마다 다르다** — 공통 격자는
            스캔별로 약 0.70~0.79 × 0.70~0.79 × 1.15~1.48 mm, fVLM은 1×1×3 mm 고정.
            복셀 단위 반경을 쓰면 fVLM의 z 한 걸음이 2~2.6배 넓어 negative를 더 멀리서 뽑고
            SRSSdm이 부풀어 오른다 — 그래서 반경을 **mm로 환산**해 프레임 간 공정성을 맞춘다.
        srss_radius_mm: SRSS/SRSSdm의 이웃 반경(mm). 기본 50은 공통 격자 32³에서
            in-plane 약 4토큰에 해당한다.
    """
    device = tokens.device
    gen = torch.Generator(device=device).manual_seed(seed)
    coords = token_coords(grid, device=str(device))
    pool = (
        torch.arange(tokens.shape[0], device=device)
        if subset is None
        else torch.nonzero(subset).flatten()
    )
    idx = pool[torch.randperm(len(pool), generator=gen, device=device)[:n_sample]]
    cos, dist = _cos_and_dist(tokens, coords, idx)

    out = {
        "LDS": lds(cos, dist, radius=max(grid) / 2),
        "CDS": cds(cos, dist),
        "RMSC": rmsc(tokens[idx]),
        **{f"seam_{k}": v for k, v in crop_seam(tokens, grid).items()},
        "cos_mean": float(cos.mean()),
        "n_tokens": int(len(pool)),
    }
    # SRSS 계열만 **복셀 단위** 좌표로 잰다. `r_near`가 토큰 단위면 격자가 다른 teacher끼리
    # 물리적으로 다른 크기의 이웃을 비교하게 된다 (SegVol 16×16×8은 토큰 하나가 32³ 복셀,
    # SPECTRE 32³은 16×16×8 복셀 — 같은 r_near=4가 2배 넓은 영역을 뜻한다).
    # 기준은 32³ 격자의 in-plane 4토큰 = 64 복셀.
    # 토큰 좌표 → **mm**: (복셀/토큰) × (mm/복셀). 이래야 격자·프레임이 달라도 같은 물리 이웃.
    scale = torch.tensor(
        [(v / g) * m for v, g, m in zip(voxel_shape, grid, mm_per_voxel)],
        device=device,
        dtype=coords.dtype,
    )
    coords_vox = coords * scale
    for name, mask in (organ_masks or {}).items():
        out[f"SRSS_{name}"] = srss(
            tokens, coords_vox, mask, r_near=srss_radius_mm, generator=gen
        )
        # 거리 통제판. 둘의 차이가 곧 "매끄러움이 SRSS를 얼마나 부풀렸는가"다.
        dm = srss_dm(tokens, coords_vox, mask, r_near=srss_radius_mm, generator=gen)
        # 널 대조군: 토큰을 **공간적으로 섞으면** 구조가 사라지므로 기댓값이 0이어야 한다.
        # feature 분포·차원·마스크는 그대로라, 0에서 벗어나면 그건 표현이 아니라 지표/격자
        # 쪽 편향이다. teacher마다 격자 크기가 달라 이 바닥값을 각자 재서 빼줘야 공정하다.
        perm = torch.randperm(tokens.shape[0], generator=gen, device=device)
        null = srss_dm(
            tokens[perm], coords_vox, mask, r_near=srss_radius_mm, generator=gen
        )
        out[f"SRSSdm_{name}"] = dm
        out[f"SRSSdmNull_{name}"] = null
        out[f"SRSSdmGap_{name}"] = dm - null
        out[f"nTok_{name}"] = int(
            mask.sum()
        )  # 표본 크기 — 거친 격자는 토큰이 훨씬 적다
    return out


__all__ = [
    "all_metrics",
    "cds",
    "srss_dm",
    "crop_seam",
    "lds",
    "rmsc",
    "spatial_norm",
    "srss",
    "token_coords",
]
