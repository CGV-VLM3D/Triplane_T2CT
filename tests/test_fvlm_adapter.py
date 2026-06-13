"""fVLM 백본 어댑터 스모크 테스트.

fVLM은 anatomy-aware 모델이라 encode_image/text 통일 계약이 없다(자세한 내용은
src/baselines/fvlm_adapter.py 와 .claude/rules/fvlm.md). 여기서는 가중치 없이
생성되는지, 체크포인트 경로 규약, 그리고 가중치가 있을 때 래핑된 모델이
upstream(eval.py) API를 노출하는지만 확인한다.
"""

from __future__ import annotations

import pytest
import torch

from src.baselines.fvlm_adapter import DEFAULT_CKPT, FVLMBackbone


def test_adapter_constructs_without_weights() -> None:
    """가중치(ckpt) 없이도 import + 생성되고, 빌드는 아직 트리거되지 않는다."""
    backbone = FVLMBackbone(load_weights=False)
    assert backbone._model is None


def test_default_checkpoint_path_points_to_workspace_data() -> None:
    assert str(DEFAULT_CKPT).startswith("/workspace/data/checkpoints/fvlm/")


def test_encode_organs_returns_present_organ_features() -> None:
    """encode_organs (Unit 3): 존재 장기만 (256,) L2-norm 피처, absent 는 skip.

    가중치 불필요 — 랜덤 init 으로 계약(반환 키/shape/정규화)만 검증한다.
    eval.py:284-338 의 장기별 center_crop → forward_test_win 경로를 충실 재현하는지,
    organ-absent skip(eval.py:295)이 동작하는지 확인. crop_size 는 CPU 부담을 줄이려고
    작게(한 패치) 준다 — 윈도잉/피처 회수 로직은 크기와 무관.
    """
    backbone = FVLMBackbone(load_weights=False, device_str="cpu")
    try:
        _ = backbone.model  # 지연 빌드 (HF 텍스트 인코더 캐시 필요)
    except Exception as e:  # noqa: BLE001 — 캐시/네트워크 없으면 skip
        pytest.skip(f"fVLM build unavailable (HF text encoder?): {e}")

    torch.manual_seed(0)
    image = torch.rand(1, 1, 32, 64, 96)  # (1, 1, D, H, W)
    mask = torch.zeros(1, 1, 32, 64, 96, dtype=torch.long)
    mask[0, 0, 8:24, 16:48, 24:72] = 1  # lung 존재
    mask[0, 0, 10:20, 20:40, 30:60] = 2  # heart 존재
    # esophagus(3) / aorta(4) 부재

    feats = backbone.encode_organs(image, mask, crop_size=(16, 16, 32))
    assert set(feats) == {"lung", "heart"}  # 존재 장기만
    for f in feats.values():
        assert tuple(f.shape) == (256,)
        assert abs(f.norm().item() - 1.0) < 1e-4  # L2-normalized


@pytest.mark.skipif(
    not DEFAULT_CKPT.is_file(),
    reason=f"fVLM checkpoint not downloaded yet at {DEFAULT_CKPT}",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for fVLM smoke")
def test_load_weights_and_expose_upstream_methods() -> None:
    """requires_weights — 체크포인트가 로드되고 래핑 모델이 eval 시점 API를 노출한다."""
    backbone = FVLMBackbone(load_weights=True, device_str="cuda:0")
    m = backbone.model  # .model 접근이 지연 빌드를 트리거.
    # third_party/fvlm/eval.py:278, 329 에서 호출하는 eval 진입점.
    assert hasattr(m, "prepare_text_feat")
    assert hasattr(m, "forward_test_win")
    # 장기별 projection (fVLM의 핵심 novelty).
    assert hasattr(m, "vision_projs")
    assert hasattr(m, "text_proj")
