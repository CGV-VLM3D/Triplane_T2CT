"""CT-CLIP saliency runner (global encoder arm).

전처리(480^3)는 CT-CLIP 추론 파이프라인을 미러하고, saliency는 빌드된 CTCLIP
모듈(backbone.clip)로 grad-enabled forward를 직접 돌린다 (encode_image 는
@torch.no_grad 라 사용 불가). CTViT 는 device='cuda' 하드코딩 → GPU 필수.

전처리는 모델별로 runner에 번들(plan): ctclip_preprocess 는 이 파일에 인라인 작성.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.ctclip_adapter import CTCLIPBackbone
from tests.saliency_map.cam import (
    ContrastOutputTarget,
    build_cam,
    make_reshape_transform,
)
from tests.saliency_map.data import SaliencyCase
from tests.saliency_map.prompts import PromptItem
from tests.saliency_map.runners.base import BaseRunner, ModelInput, SaliencyResult

# CT-CLIP 입력 규약 (data_inference_nii.py:108-133): 0.75/0.75/1.5 mm, 480x480x240.
_TARGET_SPACING = (1.5, 0.75, 0.75)  # (z, x, y) mm — resize_array current/target 순서
_TARGET_SHAPE = (480, 480, 240)  # (H, W, D) — center crop/pad 기준
_HU_MIN, _HU_MAX = -1000, 1000


def _resize_array(array: torch.Tensor, current: tuple, target: tuple) -> np.ndarray:
    """DUPLICATION INTENTIONAL — data_inference_nii.py:12-34 의 resize_array 그대로.

    array: (1, 1, Z, X, Y); current/target: (z, x, y) mm. trilinear 리샘플.
    """
    orig = array.shape[2:]  # (Z, X, Y)
    factors = [current[i] / target[i] for i in range(3)]
    new_shape = [int(orig[i] * factors[i]) for i in range(3)]  # (Z', X', Y')
    return (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )


def ctclip_preprocess(case: SaliencyCase) -> torch.Tensor:
    """CT-RATE 볼륨 → CT-CLIP 입력 (1, 1, 240, 480, 480).

    data_inference_nii.py:96-165 (nii_img_to_tensor) 의 공간/강도 처리를 미러하되,
    **metadata RescaleSlope/Intercept 는 적용하지 않는다** — valid_fixed 는 이미 HU 가
    구워져 있어 재적용하면 tissue 를 air 로 손상시킨다([[ctrate-fixed-hu-no-rescale]];
    upstream 의 slope·intercept 는 raw CT-RATE 전용). spacing 은 헤더와 동일함을 확인,
    case.meta 의 xy/z 사용. 강도: clip[-1000,1000] → /1000 ≈ [-1, 1]; pad 값 -1.
    """
    m = case.meta  # xy_spacing / z_spacing (slope/intercept 는 의도적으로 미사용)
    img = nib.load(
        str(case.ct_path)
    ).get_fdata()  # (X, Y, Z) — 이미 HU (재rescale 금지)
    img = np.clip(img, _HU_MIN, _HU_MAX)  # clip[-1000, 1000]
    img = img.transpose(2, 0, 1)  # (Z, X, Y)

    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, Z, X, Y)
    current = (m["z_spacing"], m["xy_spacing"], m["xy_spacing"])  # (z, x, y) mm
    img = _resize_array(t, current, _TARGET_SPACING)  # (1, 1, Z', X', Y')
    img = img[0][0]  # (Z', X', Y')
    img = np.transpose(img, (1, 2, 0))  # (X', Y', Z') = (H, W, D)
    img = (img / 1000).astype(np.float32)  # [-1, 1] norm

    t = torch.tensor(img)  # (H, W, D)
    dh, dw, dd = _TARGET_SHAPE
    h, w, d = t.shape
    # 중앙 크롭 → 중앙 패딩(-1) 으로 정확히 (480, 480, 240).
    hs, ws, ds = max((h - dh) // 2, 0), max((w - dw) // 2, 0), max((d - dd) // 2, 0)
    t = t[hs : hs + dh, ws : ws + dw, ds : ds + dd]  # (≤H, ≤W, ≤D)
    ph, pw, pd = dh - t.shape[0], dw - t.shape[1], dd - t.shape[2]
    t = F.pad(
        t,
        (pd // 2, pd - pd // 2, pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
        value=-1,  # -1 padding
    )  # (480, 480, 240) = (H, W, D)
    t = t.permute(2, 0, 1)  # (D, H, W) = (240, 480, 480)
    return t.unsqueeze(0).unsqueeze(0)  # (1, 1, 240, 480, 480) = (B, 1, D, H, W)


class _CTCLIPImageWrapper(torch.nn.Module):
    """grad-enabled image encode (encode_image 의 @torch.no_grad 없는 버전).

    문제: CTViT 의 `return_encoded_tokens` 토큰은 VQ(vector quantization) **이후** 라
    requires_grad=False → 그 위 어떤 spatial 층으로도 gradient 가 못 올라가 Grad-CAM 불가.

    해법: straight-through estimator. pre-VQ 인코딩 토큰(grad OK)과 실제 post-VQ 토큰(값)을
    합쳐 `enc_st = enc_pre + (quant − enc_pre).detach()` — **값은 실제 post-VQ 와 동일**
    (score/pred_prob 보존), gradient 는 VQ 를 항등으로 통과해 pre-VQ spatial 로 흐름.
    `probe`(Identity) 출력을 CAM target 으로 hook.

    forward 본문은 ctclip_adapter.py:246-250(encode_image) 의 pool→project 와 동일.
    """

    def __init__(self, clip: torch.nn.Module) -> None:
        super().__init__()
        self.clip = clip
        self.probe = torch.nn.Identity()  # CAM target (straight-through 인코딩 토큰)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:  # vol: (B, 1, D, H, W)
        vt = self.clip.visual_transformer  # image encoder, CTViT ... third_party/ct_clip/transformer_maskgit/transformer_maskgit/ctvit.py 참고
        enc_pre = vt.encode(vt.to_patch_emb(vol))  # (B, T, 24, 24, 512) pre-VQ, grad OK
        with torch.no_grad():
            quant = vt(
                vol, return_encoded_tokens=True
            )  # (B, T, 24, 24, 512) 실제 post-VQ
        enc = (
            enc_pre + (quant - enc_pre).detach()
        )  # straight-through: 값=quant, grad→enc_pre
        enc = self.probe(enc)  # (B, T, 24, 24, 512)  <- CAM target layer 출력
        enc = torch.mean(enc, dim=1)  # temporal pool → (B, 24, 24, 512)
        enc = enc.reshape(enc.shape[0], -1)  # flatten → (B, 294912)
        latent = self.clip.to_visual_latent(enc)  # (B, 512)
        return F.normalize(latent, dim=-1)  # (B, 512) L2-norm


class CTCLIPRunner(BaseRunner):
    name = "ctclip"

    def build(self) -> None:
        device = self.cfg.device
        self.backbone = CTCLIPBackbone(device_str=device, load_weights=True)
        self.clip = self.backbone.clip  # lazy build + weight load (grad-enabled handle)
        self.device = next(
            self.clip.parameters()
        ).device  # next는 제너레이터에서 첫 번째 파라미터 텐서 하나만 꺼냄
        self.wrapper = _CTCLIPImageWrapper(self.clip)  # grad-enabled forward
        self.target_layers = [self.wrapper.probe]  # straight-through 토큰 (5D, grad OK)

    def preprocess(self, case: SaliencyCase) -> ModelInput:
        vol = ctclip_preprocess(case).to(self.device)  # (1, 1, 240, 480, 480)
        # display 배경: 전처리 볼륨 (D, H, W), HU/1000 ≈ [-1, 1].
        disp = vol[0, 0].detach().cpu().numpy()  # (240, 480, 480)
        mi = ModelInput(scan_id=case.scan_id, display_volume=disp)
        mi.vol = vol  # (1, 1, 240, 480, 480) — saliency 입력 (서브클래스 확장 필드)
        return mi

    def _encode_text(self, text: str) -> torch.Tensor:
        """text → (512,) L2-norm 텍스트 latent (grad 불필요 — 상수)."""
        tok = self.backbone.tokenize(text)
        feat = self.backbone.encode_text(tok["input_ids"], tok["attention_mask"])
        return feat[0]  # (512,)

    def _upsample(self, m: torch.Tensor) -> np.ndarray:
        """(T, 24, 24) 토큰 맵 → (240, 480, 480) trilinear upsample."""
        up = F.interpolate(
            m[None, None].float(),
            size=(240, 480, 480),
            mode="trilinear",
            align_corners=False,
        )  # (1, 1, 240, 480, 480)
        return up[0, 0].cpu().numpy()  # (240, 480, 480)

    def saliency(
        self, model_input: ModelInput, item: PromptItem
    ) -> SaliencyResult | None:
        vol = model_input.vol  # (1, 1, 240, 480, 480)
        t_pos = self._encode_text(item.pos)  # (512,)
        t_neg = self._encode_text(item.neg)  # (512,)

        # 한 번의 no_grad forward 로 I + grid + (native 재료) 모두 확보.
        with torch.no_grad():
            enc = self.clip.visual_transformer(
                vol, return_encoded_tokens=True
            )  # (B, T, 24, 24, 512)
            T = enc.shape[1]
            pooled = enc.mean(1).reshape(1, -1)  # (1, 294912)
            latent = self.clip.to_visual_latent(pooled)  # (1, 512)
            img_feat = F.normalize(latent, dim=-1)[0]  # (512,)

        cos_pos = (img_feat @ t_pos).item()
        cos_neg = (img_feat @ t_neg).item()
        score = cos_pos - cos_neg  # S = cos(I,t_pos) − cos(I,t_neg)
        # zero-shot 양성 확률 (cosine softmax; 온도 없이 — 정성 표기용).
        pred_prob = torch.softmax(torch.tensor([cos_neg, cos_pos]), dim=0)[1].item()

        # --- LayerCAM (grad-enabled wrapper) ---
        method = self.cfg.cam
        with build_cam(
            method, self.wrapper, self.target_layers, make_reshape_transform(None)
        ) as cam:
            grayscale = cam(
                input_tensor=vol, targets=[ContrastOutputTarget(t_pos, t_neg)]
            )  # (1, 240, 480, 480), [0,1]
        cam_map = grayscale[0]  # (240, 480, 480)

        # --- native token-text sim map (탐색용; config native_map) ---
        native = None
        if self.cfg.native_map:
            v = F.normalize(enc, dim=-1)  # (B, T, 24, 24, 512) 토큰별 정규화
            nmap = (v * (t_pos - t_neg)).sum(-1)[
                0
            ]  # (T, 24, 24) = cos(v,t_pos)−cos(v,t_neg)
            native = self._upsample(nmap)  # (240, 480, 480)

        return SaliencyResult(
            item_name=item.name,
            organ=None,  # CT-CLIP global — organ 라우팅 없음
            cam=cam_map,
            display=model_input.display_volume,  # (240, 480, 480) 전처리 볼륨
            score=score,
            pred_prob=pred_prob,
            native_map=native,
            grid_shape=(T, 24, 24),
        )
