"""fVLM saliency runner (fine-grained, anatomy-aware encoder arm).

전처리는 src.baselines.fvlm_preprocess 를 import 재사용. saliency 는 빌드된
BlipPretrain(backbone.model)로 organ 분기(blip_pretrain.py:491-526)를 grad-enabled
재구현 — forward_test_win 은 @torch.no_grad 라 사용 불가.

두 독립 config 축:
  - fvlm_crop: "organ" (장기 중심 center_crop_organ + divisible_pad_end — eval 충실)
               | "whole" (전체 전처리 볼륨 — 마스크로 crop 안 함).
  - fvlm_mask: True (organ 마스크가 덮는 ViT 토큰만) | False (전체 토큰 주입 — 붕괴).

arm 조합:
  - +mask (default A): crop=organ, mask=True  → eval 분포 일치, organ 집중.
  - −mask (A):         crop=whole, mask=False → 마스크 완전 제거, organ query 자력 국소화 → 붕괴.
  - +mask whole (옵션): crop=whole, mask=True  → 전체 볼륨에서 organ-token 만.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.fvlm_adapter import FVLMBackbone
from src.baselines.fvlm_preprocess import (
    FVLM_PATCH_SIZE,
    center_crop_organ,
    divisible_pad_end,
    load_ct_and_mask_for_local,
)
from tests.saliency_map.cam import (
    ContrastOutputTarget,
    build_cam,
    make_reshape_transform,
)
from tests.saliency_map.data import SaliencyCase, display_axcodes
from tests.saliency_map.prompts import PromptItem
from tests.saliency_map.runners.base import BaseRunner, ModelInput, SaliencyResult


def _token_grid(shape_dhw: tuple[int, int, int]) -> tuple[int, int, int]:
    """이미지 (D, H, W) → ViT 토큰 그리드 (D/16, H/16, W/32)."""
    pd, ph, pw = FVLM_PATCH_SIZE  # (16, 16, 32)
    d, h, w = shape_dhw
    assert d % pd == 0 and h % ph == 0 and w % pw == 0, (
        f"image {shape_dhw} not divisible by patch {FVLM_PATCH_SIZE}"
    )
    return (d // pd, h // ph, w // pw)


class _FVLMOrganWrapper(torch.nn.Module):
    """grad-enabled organ-feature forward (forward_test_win 의 @torch.no_grad 없는 버전).

    blip_pretrain.py:491-526 의 organ 분기를 재현: ViT 토큰 → (±mask) organ 토큰 선택
    → organ-query cross-attention → vision_projs → L2-norm I (1, embed_dim).
    (mask, organ_id, fvlm_mask) 는 케이스/장기마다 다르므로 생성 시 바인딩.
    pytorch-grad-cam 은 forward(image) 만 호출하고 visual_encoder.norm 출력에서 활성/grad 캡처.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mask_dhw: torch.Tensor,
        organ_id: int,
        fvlm_mask: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.organ_id = organ_id
        self.fvlm_mask = fvlm_mask
        # organ 이 덮는 ViT 토큰 선택 (max_pool3d → flatten bool), blip_pretrain.py:496-503.
        organ_mask = (mask_dhw == organ_id + 1).float()  # (D, H, W)
        ds = F.max_pool3d(
            organ_mask.unsqueeze(0), kernel_size=(16, 16, 32), stride=(16, 16, 32)
        )  # (1, d, h, w)
        self.tokens = ds[0].flatten() > 0  # (L,) bool — organ 커버 토큰, True False
        self.n_tokens = int(self.tokens.sum())  # organ 커버 토큰의 총 수

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # image: (1, 1, D, H, W)
        _, hidden = self.model.visual_encoder(image)  # hidden: 4 levels × (1, L, C)
        q = self.model.query_tokens[self.organ_id][None, None]  # (1, 1, C)
        if self.fvlm_mask:  # +mask: organ 커버 토큰만 (4 levels concat)
            kv = torch.cat([lvl[0][self.tokens] for lvl in hidden])[
                None
            ]  # (1, 4·ntok, C)
        else:  # −mask: 전체 토큰 주입 (organ query가 스스로 찾아야 함 → 붕괴)
            kv = torch.cat([lvl[0] for lvl in hidden])[None]  # (1, 4·L, C)
        feat, _ = self.model.attention(q, kv, kv)  # (1, 1, C) cross-attention
        feat = feat.squeeze(0)  # (1, C)
        return F.normalize(
            self.model.vision_projs[self.organ_id](feat),
            dim=-1,  # 특정 organ에 해당하는 normalized mebedding
        )  # (1, embed_dim=256)


class FVLMRunner(BaseRunner):
    name = "fvlm"

    def build(self) -> None:
        self.backbone = FVLMBackbone(device_str=self.cfg.device, load_weights=True)
        self.model = self.backbone.model  # lazy build + weight load (BlipPretrain)
        self.device = next(self.model.parameters()).device
        self.fvlm_mask = self.cfg.fvlm_mask  # 토큰 선택 (organ-only/전체)
        self.fvlm_crop = self.cfg.fvlm_crop  # 입력 crop (organ/whole)
        self.target_layers = [self.model.visual_encoder.norm]  # 최종 ViT 토큰 (B, L, C)

    def preprocess(self, case: SaliencyCase) -> ModelInput:
        # 전체 전처리 볼륨 (pad_only=False = preprocess.py 충실, ≥112×256×352 대칭패딩, 크롭 X).
        # 장기별 center_crop / divisible_pad 는 saliency 시점에 fvlm_crop 모드별로 적용.
        image, mask = load_ct_and_mask_for_local(
            case.ct_path, case.ts_mask_path, pad_only=False
        )  # image/mask: (1, 1, D, H, W); mask int {0..4}
        mi = ModelInput(
            scan_id=case.scan_id,
            display_volume=image[0, 0].numpy(),  # (D, H, W) 전체 볼륨 (참고용)
        )
        mi.image = image.to(self.device)  # (1, 1, D, H, W) base
        mi.mask = mask.to(self.device)  # (1, 1, D, H, W) base organ id
        mi.orient = display_axcodes(case.ct_path, (2, 1, 0))  # (D,H,W)=(Z,Y,X)
        return mi

    def _crop(
        self, mi: ModelInput, organ_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        """fvlm_crop 모드에 따라 (image, mask) → 패치정렬 입력 + 토큰 그리드."""
        if self.fvlm_crop == "organ":  # eval 충실: 장기 중심 crop
            img, msk = center_crop_organ(mi.image, mi.mask, organ_id)  # (1,1,d',h',w')
        else:  # whole: 전체 볼륨 (crop 안 함)
            img, msk = mi.image, mi.mask
        img, msk = divisible_pad_end(img, msk)  # ViT 패치 (16,16,32) 배수로 끝-패딩
        grid = _token_grid(img.shape[-3:])  # (d, h, w)
        return img, msk, grid

    def _encode_text(self, item: PromptItem) -> tuple[torch.Tensor, torch.Tensor]:
        """prepare_text_feat → (neg_feat, pos_feat) 각 (embed_dim,)."""
        key = (item.organ, item.name, item.neg, item.pos)
        feat = self.backbone.prepare_text_feat([key])[
            key
        ]  # (2, embed_dim) = [neg, pos]
        return feat[0], feat[1]

    def _upsample(self, m: torch.Tensor, size_dhw: tuple[int, int, int]) -> np.ndarray:
        """(d, h, w) 토큰 맵 → (D, H, W) trilinear upsample."""
        up = F.interpolate(
            m[None, None].float(), size=size_dhw, mode="trilinear", align_corners=False
        )  # (1, 1, D, H, W)
        return up[0, 0].cpu().numpy()  # (D, H, W)

    def saliency(
        self, model_input: ModelInput, item: PromptItem
    ) -> SaliencyResult | None:
        if item.organ is None:  # CT-CLIP 전용 프롬프트 (organ 태그 없음) → fVLM skip
            return None
        organ_id = self.model.organs.index(item.organ)
        if int((model_input.mask == organ_id + 1).sum()) == 0:  # organ-absent 가드
            return None

        img, msk, grid = self._crop(model_input, organ_id)  # (1,1,D,H,W), grid (d,h,w)
        wrapper = _FVLMOrganWrapper(self.model, msk[0, 0], organ_id, self.fvlm_mask)
        if (
            self.fvlm_mask and wrapper.n_tokens == 0
        ):  # +mask 인데 organ 커버 토큰 0 → skip
            return None

        neg_feat, pos_feat = self._encode_text(item)  # (embed_dim,) 각각

        # pred_prob: zero-shot 양성 확률 (forward_test_win logits = I @ feat.T / temp).
        with torch.no_grad():
            img_feat = wrapper(img)[0]  # (embed_dim,)
        cos_neg = (img_feat @ neg_feat).item()
        cos_pos = (img_feat @ pos_feat).item()
        score = cos_pos - cos_neg  # S = cos(I,t_pos) − cos(I,t_neg)
        logits = torch.tensor([cos_neg, cos_pos]) / self.model.temp.item()
        pred_prob = torch.softmax(logits, dim=0)[1].item()

        # --- LayerCAM (grad-enabled wrapper) ---
        method = self.cfg.cam
        with build_cam(
            method, wrapper, self.target_layers, make_reshape_transform(grid)
        ) as cam:
            grayscale = cam(
                input_tensor=img, targets=[ContrastOutputTarget(pos_feat, neg_feat)]
            )  # (1, D, H, W), [0,1]
        cam_map = grayscale[0]  # (D, H, W)

        # --- native token-text sim map (보조; fVLM 적합 — 논문 방식, gradient-free) ---
        native = None
        if self.cfg.native_map:
            with torch.no_grad():
                _, hidden = self.model.visual_encoder(img)  # 4 levels × (1, L, C)
                tok = hidden[-1][0]  # (L, C) 최종 레벨 토큰
                v = F.normalize(
                    self.model.vision_projs[organ_id](tok), dim=-1
                )  # (L, ed)
                nmap = (v @ (pos_feat - neg_feat)).reshape(grid)  # (d, h, w)
            native = self._upsample(nmap, img.shape[-3:])  # (D, H, W)

        return SaliencyResult(
            item_name=item.name,
            organ=item.organ,
            cam=cam_map,
            display=img[0, 0].detach().cpu().numpy(),  # (D, H, W) window 프레임
            score=score,
            pred_prob=pred_prob,
            native_map=native,
            grid_shape=grid,
            orient=model_input.orient,
        )
