"""CT-CLIP 백본 어댑터 스모크 테스트.

가중치 없이도 어댑터가 import·생성되는지, 기본 체크포인트 경로가
/workspace/data/checkpoints/ctclip/ 아래인지 확인한다. 실제 가중치가 있으면
cuda:0에서 encode_image / encode_text가 NaN 없이 동작하는지까지 검증한다.
"""

from __future__ import annotations

import pytest

from src.baselines.ctclip_adapter import (
    DEFAULT_CKPT,
    CTCLIPBackbone,
)


def test_adapter_constructs_without_weights() -> None:
    """Construct (lazy build) — adapter must import + instantiate even without ckpts."""
    backbone = CTCLIPBackbone(load_weights=False)
    assert backbone.image_dim == 512
    assert backbone.text_dim == 512
    # No build triggered yet — `_clip` should still be None.
    assert backbone._clip is None


def test_default_checkpoint_path_points_to_workspace_data() -> None:
    """CT-CLIP ckpt must land under /workspace/data/checkpoints/ctclip/."""
    assert str(DEFAULT_CKPT).startswith("/workspace/data/checkpoints/ctclip/")


def _cuda_available() -> bool:
    import torch  # noqa: PLC0415

    return torch.cuda.is_available()


@pytest.mark.skipif(
    not DEFAULT_CKPT.is_file(),
    reason=f"CT-CLIP_v2.pt not downloaded yet at {DEFAULT_CKPT}",
)
@pytest.mark.skipif(
    not _cuda_available(),
    reason=(
        "CT-CLIP upstream (third_party/ct_clip/transformer_maskgit/ctvit.py) hardcodes "
        "`device=torch.device('cuda')` in 7 places, so it only runs on cuda:0. "
        "Use CUDA_VISIBLE_DEVICES=N to pin a different physical GPU."
    ),
)
def test_encode_image_and_text_with_real_weights() -> None:
    """requires_weights — real encode_image / encode_text smoke on cuda:0.

    NOTE: CT-CLIP upstream only runs on cuda:0. To run on a different physical
    GPU, set CUDA_VISIBLE_DEVICES=N (which remaps that GPU to cuda:0) — do not
    pass `device_str='cuda:1'` directly.
    """
    import torch

    backbone = CTCLIPBackbone(load_weights=True, device_str="cuda:0")
    # CT-CLIP_v2 visual encoder expects (B, 1, D, H, W) with D divisible by
    # temporal_patch_size=10 and H == W == image_size (480).
    vol = torch.zeros(1, 1, 240, 480, 480, device="cuda:0")
    image_feat = backbone.encode_image(vol)
    assert image_feat.shape == (1, backbone.image_dim)
    assert not torch.isnan(image_feat).any()

    toks = backbone.tokenize("lung opacity in the right lower lobe")
    text_feat = backbone.encode_text(toks["input_ids"], toks["attention_mask"])
    assert text_feat.shape == (1, backbone.text_dim)
    assert not torch.isnan(text_feat).any()
