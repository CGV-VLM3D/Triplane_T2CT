"""공간 구조 지표의 성질 검증 — 특히 `srss` vs `srss_dm`의 매끄러움 교란.

U6에서 teacher 순위가 지표 정의에 좌우된다는 게 드러났다 (CNN teacher CT-FM이 SRSS/LDS에서
SPECTRE를 이기는데, cos-sim 그림은 "의미 있는 분리"가 아니라 그냥 매끄러운 덩어리로 보였다).
그래서 지표 자체가 무엇에 반응하는지를 **인공 입력**으로 고정한다.
"""

from __future__ import annotations

import torch

from tests.repa_probe._metrics import srss, srss_dm, token_coords

GRID = (16, 16, 16)
DIM = 32


def _ball_mask(grid: tuple[int, int, int], radius: float = 4.0) -> torch.Tensor:
    """격자 중앙의 구형 '장기' 마스크 `(T,)` bool."""
    coords = token_coords(grid)
    center = torch.tensor([(n - 1) / 2 for n in grid])
    return ((coords - center) ** 2).sum(-1).sqrt() <= radius


def _smooth_field(grid: tuple[int, int, int], seed: int = 0) -> torch.Tensor:
    """장기와 **무관한** 매끄러운 랜덤 필드 `(T, DIM)`.

    저해상도 노이즈를 trilinear 업샘플해 만든다 — 이웃 토큰은 자동으로 비슷하지만 장기 경계와는
    아무 관계가 없다. CNN teacher의 성질을 흉내낸 것.
    """
    g = torch.Generator().manual_seed(seed)
    low = torch.randn(1, DIM, 4, 4, 4, generator=g)
    up = torch.nn.functional.interpolate(
        low, size=grid, mode="trilinear", align_corners=False
    )
    return up[0].permute(1, 2, 3, 0).reshape(-1, DIM)


def _organ_field(mask: torch.Tensor, noise: float = 0.1, seed: int = 0) -> torch.Tensor:
    """장기 안/밖이 서로 직교하는 방향을 갖는 필드 `(T, DIM)` — '완벽한' 장기 인식 표현."""
    g = torch.Generator().manual_seed(seed)
    x = noise * torch.randn(len(mask), DIM, generator=g)
    x[mask, 0] += 1.0
    x[~mask, 1] += 1.0
    return x


def test_srss_is_fooled_by_smoothness_but_srss_dm_is_not() -> None:
    """이 probe 전체의 핵심 통제. 매끄럽기만 한 필드에서 두 지표가 갈려야 한다."""
    mask = _ball_mask(GRID)
    coords = token_coords(GRID)
    tokens = _smooth_field(GRID)

    plain = srss(tokens, coords, mask, generator=torch.Generator().manual_seed(0))
    matched = srss_dm(tokens, coords, mask, generator=torch.Generator().manual_seed(0))

    # 장기 정보가 전혀 없는데도 통제 안 된 SRSS는 크게 나온다 — 이게 교란의 실체다.
    assert plain > 0.15, plain
    # 거리를 맞추면 신호가 사라진다.
    assert abs(matched) < 0.05, matched
    assert matched < plain


def test_both_metrics_fire_on_a_genuinely_organ_aware_field() -> None:
    """거리 통제가 **진짜** 신호까지 죽이지는 않는지 — 없으면 지표가 무의미하다."""
    mask = _ball_mask(GRID)
    coords = token_coords(GRID)
    tokens = _organ_field(mask)

    plain = srss(tokens, coords, mask, generator=torch.Generator().manual_seed(0))
    matched = srss_dm(tokens, coords, mask, generator=torch.Generator().manual_seed(0))

    # noise=0.1이라 같은 장기 토큰끼리도 cosine이 1은 아니다 (실측 ~0.76).
    assert plain > 0.7, plain
    assert matched > 0.7, matched
    # 진짜 장기 인식 표현이면 두 정의가 거의 같은 값을 줘야 한다 — 거리 통제가 신호를 깎지 않는다.
    assert abs(plain - matched) < 0.15, (plain, matched)


def test_srss_dm_is_near_zero_on_a_constant_field() -> None:
    """상수 필드는 모든 쌍의 cosine이 1이라 어떤 분리도 없어야 한다."""
    mask = _ball_mask(GRID)
    coords = token_coords(GRID)
    tokens = torch.ones(len(mask), DIM)
    assert abs(srss_dm(tokens, coords, mask)) < 1e-4


def test_srss_dm_returns_nan_when_an_organ_is_absent() -> None:
    """마스크가 비면 조용히 0을 주지 말고 NaN이어야 한다 (집계에서 걸러진다)."""
    coords = token_coords(GRID)
    empty = torch.zeros(coords.shape[0], dtype=torch.bool)
    assert torch.isnan(torch.tensor(srss_dm(_smooth_field(GRID), coords, empty)))
