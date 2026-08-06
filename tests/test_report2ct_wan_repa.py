"""report2ct_wan + REPA 배선 테스트 — 계획의 sanity-check 층 **S2**(grad 경로)와 **S3**(무해성).

이 두 층이 이 작업의 실질적 방어선이다. REPA의 실패는 크래시가 아니라 **조용한 무동작**으로 나타난다:
tap 텐서를 실수로 detach하면 손실은 정상적으로 내려가는데 UNet은 전혀 학습되지 않고, projector를
optimizer에 안 넣으면 정렬 손실이 랜덤 초기값에서 평평하게 머문다. 둘 다 FID를 돌려보기 전까지 증상이 없다.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.models.components.repa import RepaAligner
from src.models.report2ct_module import Report2CTModule

GRID = (16, 16, 16)  # 작은 latent 격자 — CPU forward를 빠르게
C = 16  # Wan latent 채널
STUDENT_GRID = (2, 2, 2)  # 16³ latent / 8 (stride-2 resblock 3개)
TEACHER_GRID = (4, 4, 4)
TEACHER_DIM = 24  # 실제 1080이지만 테스트에서는 작게


def _build_unet() -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=C,
        out_channels=C,
        num_channels=[64, 128, 256, 512],
        num_res_blocks=2,
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        cross_attention_dim=2560,
        num_class_embeds=128,
        include_fc=True,
        include_spacing_input=True,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        resblock_updown=True,
        with_conditioning=True,
        use_flash_attention=False,
    )


def _scheduler() -> RFlowScheduler:
    return RFlowScheduler(
        num_train_timesteps=1000,
        scale=1.4,
        use_timestep_transform=True,
        use_discrete_timesteps=False,
        sample_method="uniform",
    )


def _aligner(**kw) -> RepaAligner:
    defaults = dict(
        student_channels=512,
        teacher_dim=TEACHER_DIM,
        student_grid=STUDENT_GRID,
        teacher_grid=TEACHER_GRID,
        crop_tokens=(2, 2, 2),
        rel_tokens=None,
    )
    return RepaAligner(**(defaults | kw))


def _build_module(repa: RepaAligner | None, **kw):
    module = Report2CTModule(
        unet=_build_unet(), noise_scheduler=_scheduler(), repa=repa, **kw
    )
    module._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=100, global_step=0
    )
    module.log = lambda *a, **k: None
    module.setup("fit")  # hook 등록
    module.train()
    return module


def _batch(with_teacher: bool) -> dict:
    torch.manual_seed(0)
    batch = {
        "image": torch.randn(1, C, *GRID),
        "spacing": torch.tensor([[80.0, 80.0, 130.0]]),
        "context_f": torch.randn(1, 2560),
        "context_i": torch.randn(1, 2560),
    }
    if with_teacher:
        n = TEACHER_GRID[0] * TEACHER_GRID[1] * TEACHER_GRID[2]
        batch["spectre"] = torch.randn(1, n, TEACHER_DIM)
    return batch


# --- S2: grad 경로 ------------------------------------------------------------


def _grad_norm(module: torch.nn.Module) -> float:
    return sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)


def _has_any_grad(module: torch.nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def test_repa_gradient_reaches_the_encoder_half_but_not_the_decoder() -> None:
    """정렬 항만 backward → 인코더 절반 + projector에 grad, 디코더 절반에는 grad 없음.

    REPA 계열 전부가 같은 구조다 (REPA: SiT-XL 28블록 중 앞 8개, U-REPA: 36 중 18).
    `middle_block`을 tap하면 gradient는 `conv_in`→`down_blocks`→`middle_block`까지만 도달하고
    `up_blocks`/`out`은 diffusion 손실만 받는다.
    """
    m = _build_module(_aligner())
    m._shared_forward(_batch(True))  # UNet forward → hook이 tap을 잡는다
    repa_loss, logs = m._last_repa
    assert repa_loss is not None and "repa_cos" in logs and "repa_rel" in logs
    repa_loss.backward()

    assert _grad_norm(m.repa) > 0, "projector에 grad가 없다"
    assert _grad_norm(m.unet.conv_in) > 0, "conv_in에 grad가 없다"
    assert _grad_norm(m.unet.down_blocks[0]) > 0, "down_blocks[0]에 grad가 없다"
    assert _grad_norm(m.unet.middle_block) > 0, "middle_block에 grad가 없다"
    assert not _has_any_grad(m.unet.up_blocks), (
        "up_blocks에 정렬 항의 grad가 새어 들어갔다"
    )
    assert not _has_any_grad(m.unet.out), "out에 정렬 항의 grad가 새어 들어갔다"


def test_detaching_the_tap_breaks_the_encoder_gradient() -> None:
    """**negative test** — tap을 detach하면 위 테스트의 핵심 단언이 실패해야 한다.

    이 테스트가 없으면 S2는 "통과하는 척"만 할 수 있다: detach된 구현에서도 projector grad는
    남으므로, 검사가 실제로 무언가를 잡는지 여기서 확인한다.
    """
    m = _build_module(_aligner())
    original = m._repa_loss

    def detached(batch):
        m._repa_tap_feat = m._repa_tap_feat.detach()  # 조용한 실패를 인공적으로 재현
        return original(batch)

    m._repa_loss = detached
    m._shared_forward(_batch(True))
    m._last_repa[0].backward()

    assert _grad_norm(m.repa) > 0, (
        "detach해도 projector는 학습된다 (그래서 조용한 실패다)"
    )
    assert not _has_any_grad(m.unet.middle_block), (
        "detach했는데 UNet에 grad가 흘렀다 — 테스트가 아무것도 잡지 못한다"
    )


def test_diffusion_gradient_reaches_every_block() -> None:
    """대조군 — diffusion 항은 UNet 전체에 grad를 준다 (tap 위치와 무관).

    ⚠ `DiffusionModelUNetMaisi`의 출력 conv는 `zero_module`로 0 초기화라, **step 0에서는
    diffusion 손실의 grad가 상류 전체에서 정확히 0**이다 ([[maisi-unet-zero-init-output-conv]]).
    그래서 여기서만 출력 conv를 non-zero로 만들고 잰다. 반대로 REPA 항은 이 conv를 통과하지
    않고 `middle_block`에서 바로 들어오므로 그 함정에 걸리지 않는다 — 위 grad-경로 테스트가
    출력 conv를 건드리지 않고도 통과하는 것이 그 증거다.
    """
    m = _build_module(_aligner())
    with torch.no_grad():  # out[-1]은 MONAI Convolution 래퍼 (zero_module로 0 초기화)
        for p in m.unet.out[-1].parameters():
            p.normal_(std=0.02)
    m._shared_forward(_batch(True)).backward()
    for name in ("conv_in", "middle_block"):
        assert _grad_norm(getattr(m.unet, name)) > 0, f"{name}에 diffusion grad가 없다"
    assert _has_any_grad(m.unet.up_blocks)
    assert _has_any_grad(m.unet.out)


def test_repa_gradient_survives_the_zero_init_output_conv() -> None:
    """REPA 항은 출력 conv의 0 초기화와 무관하게 step 0에도 인코더에 grad를 준다.

    diffusion 손실만으로는 step 0에서 `middle_block` grad가 0인데(위 테스트의 주석),
    정렬 항은 0이 아니다 — 두 경로가 서로 다른 곳으로 들어온다는 직접 증거.
    """
    m = _build_module(_aligner())
    m._shared_forward(
        _batch(True)
    ).backward()  # diffusion만, 출력 conv는 0 초기화 그대로
    assert _grad_norm(m.unet.middle_block) == 0.0

    m2 = _build_module(_aligner())
    m2._shared_forward(_batch(True))
    m2._last_repa[0].backward()
    assert _grad_norm(m2.unet.middle_block) > 0.0


def test_teacher_tokens_carry_no_gradient() -> None:
    """teacher는 precompute된 leaf여야 한다 — grad 경로가 생기면 안 된다."""
    m = _build_module(_aligner())
    batch = _batch(True)
    m._shared_forward(batch)
    assert not batch["spectre"].requires_grad
    assert batch["spectre"].grad_fn is None


def test_shared_forward_still_returns_a_bare_scalar() -> None:
    """`_shared_forward`의 반환은 스칼라 유지 — 기존 3개 테스트가 `loss.ndim == 0`을 단언한다."""
    m = _build_module(_aligner())
    loss = m._shared_forward(_batch(True))
    assert loss.ndim == 0


def test_tap_is_cleared_every_step() -> None:
    """스텝 사이에 tap을 비워야 그래프가 남지 않는다 (메모리 누수 방지)."""
    m = _build_module(_aligner())
    m._shared_forward(_batch(True))
    assert m._repa_tap_feat is None


def test_validation_does_not_score_repa() -> None:
    """val/loss에 정렬 항이 섞이면 checkpoint 콜백이 보는 값이 달라진다 — eval 모드에서는 건너뛴다."""
    m = _build_module(_aligner())
    m.eval()
    m._shared_forward(_batch(True))
    assert m._last_repa == (None, {})
    assert m._repa_tap_feat is None, "eval 경로에서도 tap은 비워져야 한다"


# --- S2: optimizer ------------------------------------------------------------


def _param_ids(module: torch.nn.Module) -> set[int]:
    return {id(p) for p in module.parameters()}


def test_projector_params_are_in_the_optimizer_exactly_once() -> None:
    """projector를 optimizer에 안 넣으면 정렬 손실이 랜덤 초기값에서 평평하게 머문다."""
    m = _build_module(_aligner())
    groups = m.configure_optimizers()["optimizer"].param_groups
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "파라미터가 중복 등록됐다"
    assert _param_ids(m.repa) <= set(ids)
    assert _param_ids(m.unet) <= set(ids)


def test_without_repa_the_optimizer_holds_only_unet_params() -> None:
    """REPA를 끄면 optimizer 구성이 기존과 동일해야 한다."""
    m = _build_module(None)
    groups = m.configure_optimizers()["optimizer"].param_groups
    ids = {id(p) for g in groups for p in g["params"]}
    assert ids == _param_ids(m.unet)


# --- S3: 꺼져 있으면 무해 -------------------------------------------------------


def _loss_sequence(module: Report2CTModule, n: int = 5, seed: int = 7) -> list[float]:
    """같은 seed에서 n 스텝의 loss 시퀀스 — RNG 소비까지 같아야 값이 일치한다."""
    torch.manual_seed(seed)
    out = []
    for _ in range(n):
        batch = {
            "image": torch.randn(1, C, *GRID),
            "spacing": torch.tensor([[80.0, 80.0, 130.0]]),
            "context_f": torch.randn(1, 2560),
            "context_i": torch.randn(1, 2560),
            "spectre": torch.randn(
                1, TEACHER_GRID[0] * TEACHER_GRID[1] * TEACHER_GRID[2], TEACHER_DIM
            ),
        }
        out.append(float(module.training_step(batch, 0)))
    return out


def test_repa_weight_zero_is_bit_identical_to_no_repa() -> None:
    """`repa_weight=0`과 `repa=None`의 loss 시퀀스가 **정확히** 같아야 한다.

    같으려면 (a) hook이 forward를 바꾸지 않고, (b) aligner의 토큰 샘플링이 global RNG를
    소비하지 않아야 한다. 둘 중 하나라도 어긋나면 값이 갈라진다.
    """
    torch.manual_seed(0)
    unet_state = _build_unet().state_dict()

    off = Report2CTModule(unet=_build_unet(), noise_scheduler=_scheduler(), repa=None)
    off.unet.load_state_dict(unet_state)
    off._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=100, global_step=0
    )
    off.log = lambda *a, **k: None
    off.setup("fit")
    off.train()

    zero = Report2CTModule(
        unet=_build_unet(),
        noise_scheduler=_scheduler(),
        repa=_aligner(),
        repa_weight=0.0,
    )
    zero.unet.load_state_dict(unet_state)
    zero._trainer = SimpleNamespace(
        world_size=1, estimated_stepping_batches=100, global_step=0
    )
    zero.log = lambda *a, **k: None
    zero.setup("fit")
    zero.train()

    assert _loss_sequence(off) == _loss_sequence(zero)


def test_enabling_repa_does_not_perturb_the_diffusion_loss() -> None:
    """REPA를 **켠** 상태에서도 diffusion 항 자체는 baseline과 정확히 같아야 한다.

    앞 테스트는 `repa_weight=0`이라 aligner가 아예 안 돈다. 여기서는 실제로 돌린 뒤
    `_shared_forward`가 낸 diffusion 손실만 떼어 비교한다 — 값이 갈라지면 hook이 forward를
    바꿨거나 aligner가 global RNG를 소비했다는 뜻이다.
    """
    torch.manual_seed(0)
    unet_state = _build_unet().state_dict()

    def build(repa):
        m = Report2CTModule(unet=_build_unet(), noise_scheduler=_scheduler(), repa=repa)
        m.unet.load_state_dict(unet_state)
        m._trainer = SimpleNamespace(
            world_size=1, estimated_stepping_batches=100, global_step=0
        )
        m.log = lambda *a, **k: None
        m.setup("fit")
        m.train()
        return m

    def diffusion_only(module):
        torch.manual_seed(11)
        out = []
        for _ in range(3):
            batch = {
                "image": torch.randn(1, C, *GRID),
                "spacing": torch.tensor([[80.0, 80.0, 130.0]]),
                "context_f": torch.randn(1, 2560),
                "context_i": torch.randn(1, 2560),
                "spectre": torch.randn(
                    1, TEACHER_GRID[0] * TEACHER_GRID[1] * TEACHER_GRID[2], TEACHER_DIM
                ),
            }
            out.append(float(module._shared_forward(batch)))
        return out

    assert diffusion_only(build(None)) == diffusion_only(build(_aligner()))


def test_repa_stop_step_switches_the_term_off() -> None:
    """HASTE one-shot termination — `repa_stop_step` 이후에는 정렬 항이 사라져야 한다."""
    m = _build_module(_aligner(), repa_stop_step=3)
    m._trainer.global_step = 0
    assert m._repa_active()
    m._trainer.global_step = 3
    assert not m._repa_active()
    m._shared_forward(_batch(True))
    assert m._last_repa == (None, {})


def test_missing_teacher_key_fails_loudly() -> None:
    """teacher 없이 REPA를 켜면 조용히 건너뛰지 말고 실패해야 한다."""
    m = _build_module(_aligner())
    with pytest.raises(KeyError, match="spectre"):
        m._shared_forward(_batch(False))


def test_init_from_ckpt_loads_weights_without_optimizer_state() -> None:
    """finetune 경로 — `init_from_ckpt`는 UNet 가중치 + scale_factor만 가져온다.

    Lightning의 `ckpt_path=`는 optimizer와 LR 스케줄까지 복원한다. 300 epoch을 다 돈 런의
    PolynomialLR은 이미 ~0이라, 그대로 이어받으면 사실상 학습이 되지 않는다.
    """
    import tempfile

    torch.manual_seed(0)
    donor = Report2CTModule(unet=_build_unet(), noise_scheduler=_scheduler())
    with torch.no_grad():
        donor.scale_factor.fill_(1.2345)
        for p in donor.unet.conv_in.parameters():
            p.normal_(std=0.1)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as fh:
        torch.save({"state_dict": donor.state_dict()}, fh.name)
        loaded = Report2CTModule(
            unet=_build_unet(), noise_scheduler=_scheduler(), init_from_ckpt=fh.name
        )

    assert float(loaded.scale_factor) == pytest.approx(1.2345)
    assert loaded._scale_factor_initialized, "scale_factor를 첫 배치에서 다시 계산하면 안 된다"
    for a, b in zip(donor.unet.conv_in.parameters(), loaded.unet.conv_in.parameters()):
        torch.testing.assert_close(a, b)
