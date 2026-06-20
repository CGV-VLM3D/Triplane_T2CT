"""Text2CT 어댑터 스모크 테스트.

가중치 없는 lazy 생성, 체크포인트 경로, native anisotropic spacing(0.75,0.75,3.0),
그리고 vendored RFlowScheduler가 우리 MONAI 1.4에서 import·step되고 shim이 monai
네임스페이스를 같은 클래스로 해석하는지 확인한다. 가중치가 있으면 report→CT 추론까지.
"""

from __future__ import annotations

import pytest
import torch

from src.baselines.text2ct_adapter import (
    AUTOENCODER_CKPT,
    CLIP_CKPT,
    UNET_CKPT,
    Text2CTAdapter,
    weights_present,
)


def test_adapter_constructs_without_build() -> None:
    """Adapter must import + instantiate without weights/GPU and NOT trigger a build."""
    adapter = Text2CTAdapter(load_weights=False)
    assert adapter.output_size == (512, 512, 128)
    assert adapter._unet is None  # lazy — no `_ensure_built` yet


def test_output_spacing_is_native_anisotropic() -> None:
    """output_spacing exposes Text2CT's native (0.75, 0.75, 3.0) so inference.py writes a
    truthful affine instead of eye(4). Array-axis order (H, W, D); slice (D) is the 3.0 mm."""
    adapter = Text2CTAdapter(load_weights=False)
    assert adapter.output_spacing == (0.75, 0.75, 3.0)


def test_default_checkpoint_paths_under_workspace_data() -> None:
    """All three checkpoints must resolve under /workspace/data/checkpoints/text2ct/."""
    for p in (AUTOENCODER_CKPT, UNET_CKPT, CLIP_CKPT):
        assert str(p).startswith("/workspace/data/checkpoints/text2ct/")


def test_vendored_rflow_scheduler_runs_under_monai_14() -> None:
    """The vendored RFlowScheduler must import and step on CPU under our MONAI 1.4."""
    from src.baselines._vendored.rectified_flow import RFlowScheduler

    sch = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
        sample_method="uniform",
        scale=1.4,
    )
    noise = torch.randn(1, 4, 16, 16, 8)
    sch.set_timesteps(
        num_inference_steps=30,
        input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
    )
    assert len(sch.timesteps) == 30
    all_next = torch.cat(
        (sch.timesteps[1:], torch.tensor([0], dtype=sch.timesteps.dtype))
    )
    img = noise
    for t, next_t in zip(sch.timesteps, all_next):
        img, _ = sch.step(torch.randn_like(img), t, img, next_t)
    assert tuple(img.shape) == (1, 4, 16, 16, 8)
    assert torch.isfinite(img).all()


def test_rflow_shim_resolves_to_same_class() -> None:
    """After the shim, `monai.networks.schedulers.RFlowScheduler` is our vendored class.

    This is what makes diff_model_demo.py:18's import and run_inference's isinstance()
    check resolve to the same class object the config `_target_` builds.
    """
    from src.baselines.text2ct_adapter import _install_rflow_shim

    _install_rflow_shim()
    import monai.networks.schedulers as ms

    from src.baselines._vendored.rectified_flow import RFlowScheduler

    assert ms.RFlowScheduler is RFlowScheduler


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not weights_present(),
    reason="Text2CT weights not downloaded; see docs/text2ct_runbook.md",
)
@pytest.mark.skipif(
    not _cuda_available(),
    reason="Text2CT is CUDA-only (FrozenCLIP3D.cuda() at clip.py:208 + UNet flash attention)",
)
def test_requires_weights_report_to_ct_inference() -> None:
    """requires_weights — full report → CT smoke (2 steps for speed).

    Named with ``requires_weights`` so ``pytest -k requires_weights`` selects it and
    ``-k "not requires_weights"`` excludes it (CI-safe), matching the documented convention.
    """
    adapter = Text2CTAdapter(
        device_str="cuda:0", load_weights=True, num_inference_steps=2
    )
    vol = adapter.inference(
        "Findings: Mild cardiomegaly. No focal consolidation. "
        "Impression: Mild cardiomegaly without consolidation."
    )
    assert tuple(vol.shape) == (1, 1, 512, 512, 128)
    assert vol.dtype == torch.int16
    assert int(vol.min()) >= -1000 and int(vol.max()) <= 1000
    assert not torch.isnan(vol.float()).any()
