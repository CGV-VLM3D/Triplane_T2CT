"""pytorch-grad-cam 래퍼 (모델 비종속).

여러 CAM 방식을 동일 시그니처로 노출(config `cam.method` 전환), contrast 점수를
CAM target 으로 표현, transformer 토큰 시퀀스를 공간 그리드로 reshape.

라이브러리 동작(검증함):
  - 5D(3D 볼륨) 네이티브 지원: get_target_width_height/get_cam_image 5D 분기,
    scale_cam_image 가 3D는 scipy.ndimage.zoom 으로 입력 크기까지 업샘플.
  - ActivationsAndGradients 가 reshape_transform 을 activation·gradient 양쪽에 적용.
  - forward: outputs = model(input); loss = Σ target(output) over (targets, outputs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_CAM_REPO = Path("/workspace/third_party/pytorch_grad_cam")
if str(_CAM_REPO) not in sys.path:
    sys.path.insert(0, str(_CAM_REPO))

from pytorch_grad_cam import (  # noqa: E402
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    HiResCAM,
    LayerCAM,
)

# 동일 생성자 시그니처 (model, target_layers, reshape_transform) 를 공유하고 **3D(5D) 볼륨을
# 지원**하는 방식만 등록. 기본은 layercam.
#   - 제외: AblationCAM/ScoreCAM (추가 인자 필요).
#   - 제외: GradCAM++/XGradCAM — 이 라이브러리 버전의 get_cam_weights 가 axis=(2,3) 으로
#     2D를 가정해 5D 활성에서 깨짐(scipy zoom rank mismatch). 3D 안전한 5종만 노출.
CAM_REGISTRY: dict[str, type] = {
    "layercam": LayerCAM,
    "gradcam": GradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "hirescam": HiResCAM,
}

# cam 계산에서 target -> 이거 좀 이것저것 실험해봐야할 수도 있음 pos - neg가 맞는가? 
# pos - neg = 이 병변이 있다는 증거 방향
# S =  I·t_pos − I·t_neg = I·(t_pos − t_neg)
class ContrastOutputTarget:
    """CAM target: S = <I, t_pos> − <I, t_neg>  (contrast pos−neg).

    라이브러리가 배치별 output 한 행에 대해 호출 → output 은 (dim,) 한 샘플의
    image latent. t_pos/t_neg 는 (dim,) 정규화 텍스트 latent.
    """

    def __init__(self, pos_vec: torch.Tensor, neg_vec: torch.Tensor) -> None:
        self.pos = pos_vec.detach()  # (dim,)
        self.neg = neg_vec.detach()  # (dim,)

    def __call__(self, output: torch.Tensor) -> torch.Tensor:  # output: (dim,)
        pos = self.pos.to(output.device)
        neg = self.neg.to(output.device)
        return (output * pos).sum() - (output * neg).sum()  # scalar


def make_reshape_transform(grid: tuple[int, int, int] | None = None):
    """target-layer activation/grad → (B, C, d, h, w) (CAM 입력 규약).

    - 5D (B, d, h, w, C)  [CTViT encoded tokens]: 그리드는 텐서가 이미 가짐 → permute 만.
    - 3D (B, N, C)        [ViT token 시퀀스]:   grid 필요 → reshape + permute.
    """

    def reshape(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 5:  # (B, d, h, w, C)
            return t.permute(0, 4, 1, 2, 3)  # (B, C, d, h, w)
        if t.dim() == 3:  # (B, N, C)
            assert grid is not None, "grid required for (B, N, C) activations"
            B, N, C = t.shape
            d, h, w = grid
            assert N == d * h * w, f"N={N} != prod(grid)={d * h * w}"
            return t.reshape(B, d, h, w, C).permute(0, 4, 1, 2, 3)  # (B, C, d, h, w)
        raise ValueError(f"unexpected activation ndim {t.dim()}")

    return reshape


def build_cam(method: str, model, target_layers: list, reshape_transform=None):
    """CAM_REGISTRY 에서 CAM 객체 생성."""
    key = method.lower()
    if key not in CAM_REGISTRY:
        raise ValueError(
            f"unknown cam method {method!r}; choices: {sorted(CAM_REGISTRY)}"
        )
    return CAM_REGISTRY[key](
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )
