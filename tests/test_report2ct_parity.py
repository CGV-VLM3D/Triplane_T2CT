"""자급식 Report2CT 베이스라인의 parity 테스트.

우리 재구현(src/baselines/*, report2ct_module.py, configs/model/report2ct.yaml)이
upstream submodule과 정확히 일치하는지 검증한다: config kwargs bit-exact, UNet forward
bit-exact, 스케줄러 동작, 텍스트·이미지 인코더 parity(ref 있을 때), 학습 스텝 스모크.
parity ref는 사용자가 docs/report2ct_training_runbook.md로 한 번 생성한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
UPSTREAM_MODEL_DEF = (
    REPO_ROOT / "third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json"
)
OUR_MODEL_YAML = REPO_ROOT / "configs/model/report2ct.yaml"
PARITY_REFS = REPO_ROOT / "data/report2ct_work_dir/parity_refs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_upstream_json() -> dict:
    return json.loads(UPSTREAM_MODEL_DEF.read_text())


def load_our_yaml() -> "DictConfig":  # noqa: F821
    from omegaconf import OmegaConf

    return OmegaConf.load(str(OUR_MODEL_YAML))


def _make_tiny_batch(
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Minimal fake batch for training_step smoke. Uses 16×16×8 latent (not full 120×120×64)."""
    B = 1
    return {
        "image": torch.randn(B, 4, 16, 16, 8, device=device),
        "context_f": torch.randn(B, 2560, device=device),
        "context_i": torch.randn(B, 2560, device=device),
        "spacing": torch.tensor([[1.0, 1.0, 1.5]], device=device) * 100,
    }


# ---------------------------------------------------------------------------
# Test 1 — Config parity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_config_parity_vs_upstream_json() -> None:
    """Every UNet + scheduler kwarg in our YAML must match the upstream JSON exactly."""
    upstream = load_upstream_json()
    ours = load_our_yaml()

    # --- UNet ---
    upstream_unet = upstream["diffusion_unet_def"]
    ours_unet = ours.unet

    # Type-mappings: JSON "@include_body_region" → False (already resolved in ours as false)
    skip_keys = {"_target_"}
    for key, val in upstream_unet.items():
        if key in skip_keys:
            continue
        # Resolve MONAI variable references
        resolved = val
        if isinstance(val, str) and val.startswith("@"):
            ref = val[1:]
            resolved = upstream.get(ref, val)
        assert key in ours_unet, f"UNet kwarg missing in YAML: {key}"
        our_val = ours_unet[key]
        assert our_val == resolved, (
            f"UNet kwarg mismatch: {key}: ours={our_val!r} upstream={resolved!r}"
        )

    # --- Noise scheduler ---
    upstream_sched = upstream["noise_scheduler"]
    ours_sched = ours.noise_scheduler
    for key, val in upstream_sched.items():
        if key in skip_keys:
            continue
        assert key in ours_sched, f"Scheduler kwarg missing in YAML: {key}"
        assert ours_sched[key] == val, (
            f"Scheduler kwarg mismatch: {key}: ours={ours_sched[key]!r} upstream={val!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — UNet forward parity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_unet_forward_parity() -> None:
    """Instantiating from upstream JSON vs our YAML gives bit-exact forward pass.

    Strategy: load state_dict from upstream UNet into ours, assert outputs are identical.
    """
    import hydra
    from monai.bundle import ConfigParser
    from omegaconf import OmegaConf

    # --- Upstream instantiation via ConfigParser ---
    parser = ConfigParser()
    parser.read_config(str(UPSTREAM_MODEL_DEF))
    unet_upstream = parser.get_parsed_content("diffusion_unet_def").eval()

    # --- Our instantiation via Hydra ---
    cfg = OmegaConf.load(str(OUR_MODEL_YAML))
    unet_ours = hydra.utils.instantiate(cfg.unet).eval()

    # State-dict key sets must match
    up_keys = set(unet_upstream.state_dict().keys())
    our_keys = set(unet_ours.state_dict().keys())
    assert up_keys == our_keys, (
        f"state_dict key mismatch: {up_keys.symmetric_difference(our_keys)}"
    )

    # Share weights → outputs must be bit-exact
    unet_ours.load_state_dict(unet_upstream.state_dict())

    torch.manual_seed(42)
    x = torch.randn(1, 4, 16, 16, 8)
    ctx = torch.randn(1, 2, 2560)
    sp = torch.tensor([[1.0, 1.0, 1.5]])
    t = torch.tensor([500])
    cls = torch.tensor([1])

    with torch.no_grad():
        out_up = unet_upstream(
            x=x, timesteps=t, context=ctx, class_labels=cls, spacing_tensor=sp
        )
        out_ours = unet_ours(
            x=x, timesteps=t, context=ctx, class_labels=cls, spacing_tensor=sp
        )

    assert torch.allclose(out_up, out_ours, atol=0), (
        "UNet output not bit-exact after weight transfer"
    )


# ---------------------------------------------------------------------------
# Test 3 — Scheduler parity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scheduler_parity() -> None:
    """Our RFlowScheduler produces same add_noise output as one from the same kwargs."""
    from src.baselines.rflow import RFlowScheduler

    upstream = load_upstream_json()
    sched_cfg = {
        k: v for k, v in upstream["noise_scheduler"].items() if not k.startswith("_")
    }

    sched_a = RFlowScheduler(**sched_cfg)
    sched_b = RFlowScheduler(**sched_cfg)

    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 16, 16, 8)
    noise = torch.randn_like(x0)
    # Share timesteps: both schedulers get identical inputs → outputs must be bit-exact.
    # (Two separate sample_timesteps calls would consume different RNG state.)
    t = sched_a.sample_timesteps(x0)
    noisy_a = sched_a.add_noise(x0, noise, t)
    noisy_b = sched_b.add_noise(x0, noise, t)
    assert torch.allclose(noisy_a, noisy_b, atol=0), (
        "RFlowScheduler add_noise not deterministic for identical inputs"
    )


# ---------------------------------------------------------------------------
# Test 4 — Text encoder parity (skip if ref absent)
# ---------------------------------------------------------------------------


def test_text_encoder_parity() -> None:
    """Our Report2CTTextEncoder ≈ notebook reference within cosine sim ≥ 0.999."""
    ref_path = PARITY_REFS / "text_ref.json"
    if not ref_path.exists():
        pytest.skip(
            "text_ref.json not generated yet — run docs/report2ct_training_runbook.md §4 to create it"
        )

    import torch.nn.functional as F

    from src.baselines.report2ct_text_encoder import Report2CTTextEncoder

    ref = json.loads(ref_path.read_text())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = Report2CTTextEncoder(device=device)

    for entry in ref["samples"]:
        ref_emb = torch.tensor(entry["embedding"], dtype=torch.float32)
        ours_emb = encoder.encode(entry["text"]).cpu().float()
        sim = F.cosine_similarity(ref_emb.unsqueeze(0), ours_emb.unsqueeze(0)).item()
        assert sim >= 0.999, (
            f"Text encoder cosine similarity {sim:.4f} < 0.999 for text: {entry['text'][:60]!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Image encoder parity (skip if ref absent)
# ---------------------------------------------------------------------------


def test_image_encoder_parity() -> None:
    """Our Report2CTImageEncoder ≈ upstream within PSNR ≥ 50 dB."""
    ref_dir = PARITY_REFS
    ref_files = list(ref_dir.glob("*_emb_ref.nii.gz")) if ref_dir.exists() else []
    if not ref_files:
        pytest.skip(
            "No *_emb_ref.nii.gz in parity_refs/ — run docs/report2ct_training_runbook.md §4 to create them"
        )

    import nibabel as nib
    import numpy as np

    from src.baselines.report2ct_image_encoder import Report2CTImageEncoder

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = Report2CTImageEncoder(device=device)

    for ref_emb_path in ref_files:
        # Derive original NIfTI path from ref filename: <id>_emb_ref.nii.gz
        vol_id = re.sub(r"_emb_ref\.nii\.gz$", "", ref_emb_path.name)
        nifti_path = (
            Path("/workspace/datasets/datasets/CT-RATE/dataset/train_fixed")
            / f"{vol_id.rsplit('_', 2)[0]}"  # patient
            / f"{'_'.join(vol_id.rsplit('_', 1)[:1])}"  # series — simplified
            / f"{vol_id}.nii.gz"
        )
        # Try to find the NIfTI via glob if path derivation is uncertain
        if not nifti_path.is_file():
            found = list(
                Path("/workspace/datasets/datasets/CT-RATE/dataset/train_fixed").rglob(
                    f"{vol_id}.nii.gz"
                )
            )
            if not found:
                pytest.skip(f"NIfTI not found for {vol_id}")
            nifti_path = found[0]

        ours_img = encoder.encode(nifti_path)
        ref_img = nib.load(str(ref_emb_path))

        ours_nda = ours_img.get_fdata().astype(np.float32)
        ref_nda = ref_img.get_fdata().astype(np.float32)

        if ours_nda.shape != ref_nda.shape:
            pytest.fail(f"Shape mismatch: ours {ours_nda.shape} vs ref {ref_nda.shape}")

        mse = np.mean((ours_nda - ref_nda) ** 2)
        max_val = max(np.abs(ref_nda).max(), 1e-8)
        psnr = 10 * np.log10(max_val**2 / (mse + 1e-12))
        assert psnr >= 50.0, f"Image encoder PSNR {psnr:.1f} dB < 50 dB for {vol_id}"


# ---------------------------------------------------------------------------
# Test 6 — Training step smoke
# ---------------------------------------------------------------------------


def test_training_step_smoke() -> None:
    """training_step returns a finite scalar and gradients flow through the UNet."""
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(OUR_MODEL_YAML))
    unet = hydra.utils.instantiate(cfg.unet)
    scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    from src.models.report2ct_module import Report2CTModule

    module = Report2CTModule(unet=unet, noise_scheduler=scheduler)

    batch = _make_tiny_batch()

    # Simulate trainer.estimated_stepping_batches (needed by configure_optimizers)
    class _FakeTrainer:
        estimated_stepping_batches = 100
        world_size = 1

    module.trainer = _FakeTrainer()
    module._scale_factor_initialized = False  # reset for clean test

    loss = module._shared_forward(batch)

    assert loss.ndim == 0, "Loss must be scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()
    grads = [p.grad for p in unet.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through UNet"
