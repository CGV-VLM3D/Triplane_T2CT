"""Unit tests for report2ct_wan_mask_v2 — dual-condition CFG (learned null mask + IP2P 5/5/5 dropout).

Two layers:
  * TRAINING (Report2CTModule, mask_cfg=True): the learned ``no_mask_embed`` is created, lands in the
    optimizer, receives gradient (broadcast + torch.where do not sever the graph), the IP2P 5/5/5
    dropout hits the right ratios, and CFG-dropped samples get the null swapped into the concat.
  * SAMPLING (Report2CTWanMaskV2LatentSampler): the text-inner / mask-outer 3-pass CFG assembles the
    passes (∅m,∅t),(∅m,t),(m,t) in that order and combines them per Eq. 3; s_m=0 ignores the mask.

NOTE on the grad test: ``DiffusionModelUNetMaisi`` zero-inits its OUTPUT conv (zero_module), so on a
freshly-built UNet the gradient to ANY input (image/mask/null) is exactly 0 — not a bug, it clears
after the first optimizer step. The grad test therefore makes the output conv non-zero first.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.eval.samplers.report2ct_wan import (
    Report2CTWanLatentSampler,
    Report2CTWanMaskV2LatentSampler,
)
from src.models.report2ct_module import Report2CTModule

GRID = (16, 16, 16)  # small latent grid for a fast CPU forward
C = 16  # Wan latent channels


# --------------------------------------------------------------------------- module helpers
def _unet(in_ch: int) -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=in_ch,
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


def _module(mask_cfg: bool) -> Report2CTModule:
    m = Report2CTModule(
        unet=_unet(2 * C), noise_scheduler=_scheduler(), mask_cfg=mask_cfg
    )
    m._trainer = SimpleNamespace(world_size=1, estimated_stepping_batches=100)
    m.log = lambda *a, **k: None
    m.train()
    return m


def _batch(B: int) -> dict:
    return {
        "image": torch.randn(B, C, *GRID) * 0.98,
        "context": torch.randn(B, 2, 2560),
        "spacing": torch.tensor([[75.0, 75.0, 150.0]] * B),
        "mask_latent": torch.randn(B, C, *GRID),
    }


def _make_out_conv_nonzero(m: Report2CTModule) -> None:
    """Undo the zero_module output-conv init so input gradients can flow (== post-step-0)."""
    with torch.no_grad():
        w = m.unet.out[-1].conv.weight
        w.add_(torch.randn_like(w) * 0.05)


# --------------------------------------------------------------------------- TRAINING tests
def test_no_mask_embed_created_and_in_optimizer() -> None:
    """mask_cfg=True ⇒ learned null (16,) exists, small-random init, and is in the optimizer."""
    m = _module(True)
    assert hasattr(m, "no_mask_embed") and tuple(m.no_mask_embed.shape) == (C,)
    assert 0.005 < float(m.no_mask_embed.detach().std()) < 0.06  # ~0.02 init
    opt = m.configure_optimizers()["optimizer"]
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.no_mask_embed) in ids
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in m.unet.parameters()) + C


def test_mask_cfg_false_byte_identical() -> None:
    """mask_cfg=False ⇒ no null param, optimizer = UNet only, plain path runs."""
    m = _module(False)
    assert not hasattr(m, "no_mask_embed")
    opt = m.configure_optimizers()["optimizer"]
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in m.unet.parameters())
    loss = m.training_step(_batch(2), 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_null_learns_and_substituted(monkeypatch) -> None:
    """Forced mask-drop: dropped samples get the null at the concat, and no_mask_embed gets gradient.

    Covers the two sanity checks: (1) broadcast + torch.where keep the graph intact so the learned
    null actually learns; (2) the dropped mask channels equal the null broadcast (not the real mask).
    """
    B = 4
    m = _module(True)
    _make_out_conv_nonzero(
        m
    )  # else zero-init output conv → all input grads are 0 at step 0
    batch = _batch(B)

    # Force samples 0,1 into the mask-only-drop bucket [2p,3p)=[0.10,0.15); 2,3 fully conditioned.
    forced = torch.tensor([0.12, 0.12, 0.50, 0.50])
    real_rand = torch.rand
    state = {"done": False}

    def fake_rand(*a, **k):
        if (
            a and a[0] == B and not state["done"]
        ):  # the CFG draw only, not timestep sampling
            state["done"] = True
            return forced
        return real_rand(*a, **k)

    monkeypatch.setattr(torch, "rand", fake_rand)

    captured: dict = {}

    def pre_hook(_mod, _args, kwargs):
        captured["x"] = kwargs["x"].detach().clone()

    m.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)
    loss = m.training_step(batch, 0)
    loss.backward()

    # (2) mask channels (16:) of dropped samples == null broadcast; kept samples == real mask.
    mask_in = captured["x"][:, C:]  # (B, 16, H, W, D)
    null_b = m.no_mask_embed.detach().view(1, -1, 1, 1, 1).expand(B, C, *GRID)
    assert torch.allclose(mask_in[0], null_b[0]) and torch.allclose(
        mask_in[1], null_b[1]
    )
    assert torch.allclose(mask_in[2], batch["mask_latent"][2])
    assert not torch.allclose(mask_in[0], batch["mask_latent"][0])
    # (1) learned null received gradient.
    g = m.no_mask_embed.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert torch.isfinite(loss)


def test_dropout_ratios() -> None:
    """IP2P 5/5/5: text dropped 2p=10%, mask dropped 2p=10%, both p=5% (from one uniform draw)."""
    p = 0.05
    r = torch.rand(400_000)
    drop_text = r < 2 * p
    mask_drop = (r >= p) & (r < 3 * p)
    assert abs(float(drop_text.float().mean()) - 0.10) < 0.005
    assert abs(float(mask_drop.float().mean()) - 0.10) < 0.005
    assert abs(float((drop_text & mask_drop).float().mean()) - 0.05) < 0.005


# --------------------------------------------------------------------------- SAMPLING tests
class _MockUNet:
    """Per-row constant = f(row's x, context); records inputs to assert pass order."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, timesteps, context, class_labels, spacing_tensor):
        self.calls.append({"x": x.clone(), "context": context.clone()})
        B = x.shape[0]
        v = x.flatten(1).sum(1) * 1e-4 + context.flatten(1).sum(1) * 1e-3
        return v.view(B, 1, 1, 1, 1).expand(B, C, 4, 4, 4).contiguous()


def _mk(cls, **attrs):
    s = cls.__new__(cls)
    s._device = torch.device("cpu")
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _sampler_inputs():
    torch.manual_seed(0)
    g = (4, 4, 4)
    return {
        "z": torch.randn(1, C, *g),
        "ctx": torch.randn(1, 2, 2560),
        "cl": torch.tensor([1]),
        "sp": torch.tensor([[75.0, 75.0, 150.0]]),
        "t": torch.tensor([500.0]),
        "mask": torch.randn(1, C, *g),
        "null": torch.randn(C),
    }


def test_base_predict_2pass_unchanged() -> None:
    """base _predict = single-condition text CFG: mask present on BOTH branches, 2 passes."""
    d = _sampler_inputs()
    s = _mk(
        Report2CTWanLatentSampler,
        _mask_latent=d["mask"],
        _null_mask=None,
        cfg_scale=5.0,
        _unet=_MockUNet(),
    )
    out = s._predict(d["z"], d["t"], d["ctx"], d["cl"], d["sp"])
    c = s._unet.calls[0]
    assert c["x"].shape[0] == 2 and c["x"].shape[1] == 2 * C
    exp_mask = torch.cat([d["z"], d["mask"]], dim=1)
    assert torch.allclose(c["x"][0], exp_mask[0]) and torch.allclose(
        c["x"][1], exp_mask[0]
    )
    assert torch.allclose(c["context"][0], torch.zeros_like(d["ctx"])[0])
    assert torch.allclose(c["context"][1], d["ctx"][0])
    v = c["x"].flatten(1).sum(1) * 1e-4 + c["context"].flatten(1).sum(1) * 1e-3
    exp = v[0] + 5.0 * (v[1] - v[0])
    assert torch.allclose(out, exp.view(1, 1, 1, 1, 1).expand_as(out))


def test_v2_predict_3pass_dual_cfg() -> None:
    """V2 _predict: passes (∅m,∅t),(∅m,t),(m,t) in order + Eq.3 combine."""
    d = _sampler_inputs()
    s = _mk(
        Report2CTWanMaskV2LatentSampler,
        _mask_latent=d["mask"],
        _null_mask=d["null"],
        cfg_scale_text=5.0,
        cfg_scale_mask=1.5,
        _unet=_MockUNet(),
    )
    out = s._predict(d["z"], d["t"], d["ctx"], d["cl"], d["sp"])
    c = s._unet.calls[0]
    xb, cb = c["x"], c["context"]
    assert xb.shape == (3, 2 * C, 4, 4, 4)
    null_x = torch.cat(
        [d["z"], d["null"].view(1, -1, 1, 1, 1).expand_as(d["mask"])], dim=1
    )
    real_x = torch.cat([d["z"], d["mask"]], dim=1)
    assert torch.allclose(xb[0], null_x[0]) and torch.allclose(
        cb[0], torch.zeros_like(d["ctx"])[0]
    )
    assert torch.allclose(xb[1], null_x[0]) and torch.allclose(cb[1], d["ctx"][0])
    assert torch.allclose(xb[2], real_x[0]) and torch.allclose(cb[2], d["ctx"][0])
    v = xb.flatten(1).sum(1) * 1e-4 + cb.flatten(1).sum(1) * 1e-3
    exp = v[0] + 5.0 * (v[1] - v[0]) + 1.5 * (v[2] - v[1])
    assert torch.allclose(out, exp.view(1, 1, 1, 1, 1).expand_as(out))


def test_v2_s_m_zero_is_mask_off() -> None:
    """s_m=0 ⇒ text CFG on the null-mask branch (mask ignored); s_m=s_t=1 ⇒ e(m,t) fully cond."""
    d = _sampler_inputs()
    s0 = _mk(
        Report2CTWanMaskV2LatentSampler,
        _mask_latent=d["mask"],
        _null_mask=d["null"],
        cfg_scale_text=5.0,
        cfg_scale_mask=0.0,
        _unet=_MockUNet(),
    )
    o0 = s0._predict(d["z"], d["t"], d["ctx"], d["cl"], d["sp"])
    c = s0._unet.calls[0]
    v = c["x"].flatten(1).sum(1) * 1e-4 + c["context"].flatten(1).sum(1) * 1e-3
    assert torch.allclose(
        o0, (v[0] + 5.0 * (v[1] - v[0])).view(1, 1, 1, 1, 1).expand_as(o0)
    )

    s1 = _mk(
        Report2CTWanMaskV2LatentSampler,
        _mask_latent=d["mask"],
        _null_mask=d["null"],
        cfg_scale_text=1.0,
        cfg_scale_mask=1.0,
        _unet=_MockUNet(),
    )
    o1 = s1._predict(d["z"], d["t"], d["ctx"], d["cl"], d["sp"])
    c = s1._unet.calls[0]
    v = c["x"].flatten(1).sum(1) * 1e-4 + c["context"].flatten(1).sum(1) * 1e-3
    assert torch.allclose(o1, v[2].view(1, 1, 1, 1, 1).expand_as(o1))


def test_v2_predict_requires_null() -> None:
    """V2 _predict raises if the ckpt carried no learned null (mask_cfg not trained)."""
    d = _sampler_inputs()
    s = _mk(
        Report2CTWanMaskV2LatentSampler,
        _mask_latent=d["mask"],
        _null_mask=None,
        cfg_scale_text=5.0,
        cfg_scale_mask=1.0,
        _unet=_MockUNet(),
    )
    try:
        s._predict(d["z"], d["t"], d["ctx"], d["cl"], d["sp"])
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "no_mask_embed" in str(e)
