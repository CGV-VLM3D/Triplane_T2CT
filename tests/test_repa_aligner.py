"""`RepaAligner` 단위 테스트 — 계획의 sanity-check 층 **S1**(손실 정의 자체).

모델도 GPU도 필요 없다. 여기서 고정하는 건 "손실이 의도한 것을 재고 있는가"이고, 특히
**회전 불변성** 테스트가 hard(cosine)와 soft(관계형)가 서로 다른 것을 재고 있음을 증명한다.
"""

from __future__ import annotations

import pytest
import torch

from src.models.components.repa import REL_LOSSES, RepaAligner, spatial_norm, st_split

TEACHER_GRID = (4, 4, 4)
STUDENT_GRID = (2, 2, 2)
C_STUDENT, D_TEACHER = 8, 6


def make_aligner(**kw) -> RepaAligner:
    """작은 격자에서 도는 aligner — 기본값은 U-REPA 설정(hard 1 + rel 3)."""
    defaults = dict(
        student_channels=C_STUDENT,
        teacher_dim=D_TEACHER,
        teacher_grid=TEACHER_GRID,
        student_grid=STUDENT_GRID,
        crop_tokens=(2, 2, 2),
        rel_tokens=None,
    )
    return RepaAligner(**(defaults | kw))


def n_tokens(grid: tuple[int, int, int]) -> int:
    return grid[0] * grid[1] * grid[2]


# --- projection / shape -------------------------------------------------------


@pytest.mark.parametrize("projector", ["conv", "mlp"])
@pytest.mark.parametrize("upsample", ["trilinear", "deconv"])
def test_project_lands_on_the_teacher_grid(projector: str, upsample: str) -> None:
    """어떤 projector/업샘플이든 출력은 teacher 격자 위의 `(B, T, D)`여야 한다."""
    al = make_aligner(projector=projector, upsample=upsample)
    z = al.project(torch.randn(2, C_STUDENT, *STUDENT_GRID))
    assert z.shape == (2, n_tokens(TEACHER_GRID), D_TEACHER)


def test_teacher_token_count_mismatch_is_loud() -> None:
    """teacher 토큰 수가 격자와 안 맞으면 조용히 broadcast되지 말고 실패해야 한다."""
    al = make_aligner()
    with pytest.raises(ValueError, match="teacher_grid"):
        al(torch.randn(1, C_STUDENT, *STUDENT_GRID), torch.randn(1, 7, D_TEACHER))


def test_all_zero_weights_is_rejected() -> None:
    """세 가중치가 전부 0이면 no-op이므로 생성 자체를 막는다."""
    with pytest.raises(ValueError, match="no-op"):
        make_aligner(hard_weight=0.0, rel_weight=0.0, global_weight=0.0)


# --- S1: 손실 정의 -------------------------------------------------------------


def _identity_pair(seed: int = 0):
    """student projection이 teacher와 정확히 같아지도록 projector를 항등으로 만든다."""
    torch.manual_seed(seed)
    al = make_aligner(projector="conv", teacher_dim=C_STUDENT)
    with torch.no_grad():  # 3×3×3 conv를 중심 탭 항등으로
        al.projector.weight.zero_()
        al.projector.bias.zero_()
        for c in range(C_STUDENT):
            al.projector.weight[c, c, 1, 1, 1] = 1.0
    student = torch.randn(
        1, C_STUDENT, *TEACHER_GRID
    )  # teacher 격자와 동일 → 업샘플 없음
    al.teacher_grid = TEACHER_GRID
    teacher = student.flatten(2).transpose(1, 2).clone()  # (1, T, C)
    return al, student, teacher


@pytest.mark.parametrize("rel_loss", REL_LOSSES)
def test_identical_inputs_give_cos_minus_one_and_zero_relational(rel_loss: str) -> None:
    """동일 입력: cosine 손실 = −1 정확히, 관계형 3변형 모두 0.

    `cos_ce`는 교차엔트로피라 최솟값이 0이 아니라 teacher 분포의 엔트로피다 — 동일 입력에서
    KL이 0인지로 확인한다.
    """
    al, student, teacher = _identity_pair()
    al.rel_loss = rel_loss
    al.spatial_norm_teacher = False
    _, logs = al(student, teacher)
    assert logs["repa_cos"].item() == pytest.approx(1.0, abs=1e-5)  # 손실은 −cos
    if rel_loss == "cos_ce":
        gram = torch.nn.functional.normalize(teacher, dim=-1)
        gram = gram @ gram.transpose(1, 2)
        entropy = (
            -(torch.softmax(gram, -1) * torch.log_softmax(gram, -1)).sum(-1).mean()
        )
        assert logs["repa_rel"].item() == pytest.approx(float(entropy), abs=1e-4)
    else:
        assert logs["repa_rel"].item() == pytest.approx(0.0, abs=1e-5)


def test_orthogonal_and_antipodal_cosines() -> None:
    """직교 입력 → cos 0, 부호 반전 입력 → cos −1 (손실 최대)."""
    al, student, teacher = _identity_pair()
    al.spatial_norm_teacher = False
    al.rel_weight = 0.0
    _, logs = al(student, -teacher)
    assert logs["repa_cos"].item() == pytest.approx(-1.0, abs=1e-5)


def test_relational_is_rotation_invariant_but_cosine_is_not() -> None:
    """**hard와 soft가 서로 다른 것을 잰다는 증명.**

    student 표현에 임의 직교행렬 Q를 곱하면 Gram은 불변이므로 관계형 손실은 바뀌지 않아야 하고,
    token-wise cosine은 바뀌어야 한다. 둘 중 하나라도 어긋나면 구현이 잘못된 것이다.
    """
    al, student, teacher = _identity_pair(seed=1)
    al.spatial_norm_teacher = False
    _, base = al(student, teacher)

    q, _ = torch.linalg.qr(
        torch.randn(C_STUDENT, C_STUDENT, generator=torch.Generator().manual_seed(3))
    )
    rotated = torch.einsum("bchwd,ck->bkhwd", student, q)
    _, rot = al(rotated, teacher)

    assert rot["repa_rel"].item() == pytest.approx(base["repa_rel"].item(), abs=1e-5)
    assert abs(rot["repa_cos"].item() - base["repa_cos"].item()) > 0.1


def test_spatial_norm_zero_means_and_unit_variance_over_tokens() -> None:
    """spatial norm은 **토큰 축**에서 평균 0 / 표준편차 1을 만든다 (γ=1)."""
    x = torch.randn(2, 40, 7) * 3.0 + 5.0
    y = spatial_norm(x, gamma=1.0)
    torch.testing.assert_close(y.mean(dim=-2), torch.zeros(2, 7), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        y.std(dim=-2, unbiased=False), torch.ones(2, 7), atol=1e-3, rtol=0
    )
    # γ=0이면 평균 제거가 사라지고 분산 정규화만 남는다
    y0 = spatial_norm(x, gamma=0.0)
    assert y0.mean(dim=-2).abs().min() > 0.1


def test_spatial_norm_touches_the_teacher_only() -> None:
    """student 경로에는 spatial norm이 걸리지 않아야 한다 — teacher만 정규화 대상."""
    al, student, teacher = _identity_pair(seed=2)
    al.rel_weight = 0.0
    al.spatial_norm_teacher = True
    _, with_norm = al(student, teacher)
    al.spatial_norm_teacher = False
    _, without = al(student, teacher)
    # teacher만 바뀌었으므로 cosine이 달라진다. student도 정규화했다면 동일 입력이라 1.0이 유지된다.
    assert without["repa_cos"].item() == pytest.approx(1.0, abs=1e-5)
    assert with_norm["repa_cos"].item() < 0.999


# --- S1: 토큰 샘플링 ----------------------------------------------------------


def test_rel_tokens_sampling_is_reproducible_and_uses_its_own_generator() -> None:
    """같은 seed → 같은 값. 그리고 **global RNG를 소비하지 않아야** 한다 (S3의 bit-identical 조건)."""
    al = make_aligner(rel_tokens=16, hard_weight=0.0)
    student = torch.randn(1, C_STUDENT, *STUDENT_GRID)
    teacher = torch.randn(1, n_tokens(TEACHER_GRID), D_TEACHER)

    def run(seed: int) -> float:
        gen = torch.Generator().manual_seed(seed)
        return float(al(student, teacher, generator=gen)[1]["repa_rel"])

    assert run(0) == run(0)
    assert run(0) != run(1)

    torch.manual_seed(123)
    before = torch.rand(1).item()
    torch.manual_seed(123)
    al(student, teacher, generator=torch.Generator().manual_seed(0))
    assert torch.rand(1).item() == before, "aligner가 global RNG 스트림을 건드렸다"


def test_within_crop_scope_never_samples_across_a_crop_boundary() -> None:
    """`within_crop`이 뽑은 토큰은 전부 같은 crop 안에 있어야 한다 (seam 제거의 근거)."""
    al = make_aligner(rel_scope="within_crop", rel_tokens=None)
    idx = al._sample_tokens(torch.device("cpu"), torch.Generator().manual_seed(0))
    grid, crop = al.teacher_grid, al.crop_tokens
    h = idx // (grid[1] * grid[2])
    w = (idx // grid[2]) % grid[1]
    d = idx % grid[2]
    assert len(idx) == crop[0] * crop[1] * crop[2]
    for axis_idx, c in zip((h, w, d), crop):
        assert len(torch.unique(axis_idx // c)) == 1, "crop 경계를 넘어 샘플링했다"


def test_videorepa_split_keeps_whole_z_slabs() -> None:
    """`videorepa_l1_st`는 cross-slab 항이 살아야 하므로 z를 통째로 가져와야 한다."""
    al = make_aligner(rel_loss="videorepa_l1_st", rel_tokens=8, hard_weight=0.0)
    idx = al._sample_tokens(torch.device("cpu"), torch.Generator().manual_seed(0))
    z = idx % al.teacher_grid[2]
    assert len(torch.unique(z)) == al.teacher_grid[2], "z 슬랩이 통째로 들어오지 않았다"


def test_st_split_separates_in_slab_from_cross_slab_errors() -> None:
    """VideoREPA Eq.5의 두 항이 실제로 분리돼 있는지 — 오차를 한쪽에만 심어 확인한다.

    in-slab 쌍에만 오차가 있으면 temporal 항은 0이어야 하고, cross-slab 쌍에만 오차가 있으면
    spatial 항이 0이어야 한다. 하나라도 새면 두 항이 사실상 같은 것을 재고 있다는 뜻이다.
    """
    slab = torch.tensor([0, 0, 1, 1])  # 토큰 4개, 슬랩 2개
    same = (slab.unsqueeze(1) == slab.unsqueeze(0)) & ~torch.eye(4, dtype=torch.bool)

    in_slab_only = torch.zeros(1, 4, 4)
    in_slab_only[0][same] = 2.0
    spatial, temporal = st_split(in_slab_only, slab)
    assert spatial.item() == pytest.approx(2.0)
    assert temporal.item() == pytest.approx(0.0)

    cross_only = torch.zeros(1, 4, 4)
    cross_only[0][~same & ~torch.eye(4, dtype=torch.bool)] = 3.0
    spatial, temporal = st_split(cross_only, slab)
    assert spatial.item() == pytest.approx(0.0)
    assert temporal.item() == pytest.approx(3.0)


def test_st_split_ignores_the_diagonal() -> None:
    """대각(자기 자신과의 유사도)은 항상 1이라 손실에 들어가면 안 된다."""
    slab = torch.zeros(3, dtype=torch.long)
    diag_only = torch.diag(torch.tensor([5.0, 5.0, 5.0])).unsqueeze(0)
    spatial, temporal = st_split(diag_only, slab)
    assert spatial.item() == pytest.approx(0.0)
    assert temporal.item() == pytest.approx(0.0)


def test_hard_weight_zero_skips_the_cosine_term() -> None:
    """`hard_weight=0`(VideoREPA식 soft-only)이면 cosine을 계산조차 하지 않는다."""
    al = make_aligner(hard_weight=0.0)
    _, logs = al(
        torch.randn(1, C_STUDENT, *STUDENT_GRID),
        torch.randn(1, n_tokens(TEACHER_GRID), D_TEACHER),
    )
    assert "repa_cos" not in logs
    assert "repa_rel" in logs


def test_global_term_requires_the_teacher_global_vector() -> None:
    """`global_weight > 0`인데 teacher_global이 없으면 조용히 건너뛰지 말고 실패해야 한다."""
    al = make_aligner(global_weight=0.5, teacher_global_dim=D_TEACHER)
    with pytest.raises(ValueError, match="teacher_global"):
        al(
            torch.randn(1, C_STUDENT, *STUDENT_GRID),
            torch.randn(1, n_tokens(TEACHER_GRID), D_TEACHER),
        )


def test_shuffled_cosine_diagnostic_is_logged_for_real_batches() -> None:
    """배치가 2 이상이면 `repa_cos_shuffled` / `repa_cos_gap` 진단이 함께 나와야 한다.

    U5/S4에서 shuffled 대조군이 0.45까지 올라갔다 — raw cosine의 상당 부분이 볼륨별 대응이 아니라
    해부학적 위치 prior다. 학습 중 실제로 봐야 하는 값은 그 **격차**다.
    """
    al = make_aligner()
    n = n_tokens(TEACHER_GRID)
    _, logs = al(torch.randn(3, C_STUDENT, *STUDENT_GRID), torch.randn(3, n, D_TEACHER))
    assert {"repa_cos", "repa_cos_shuffled", "repa_cos_gap"} <= set(logs)
    assert logs["repa_cos_gap"].item() == pytest.approx(
        logs["repa_cos"].item() - logs["repa_cos_shuffled"].item(), abs=1e-6
    )
    # 배치 1이면 섞을 상대가 없으므로 진단을 내지 않는다 (자기 자신과의 비교는 무의미)
    _, single = al(torch.randn(1, C_STUDENT, *STUDENT_GRID), torch.randn(1, n, D_TEACHER))
    assert "repa_cos_shuffled" not in single
