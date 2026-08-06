"""마스크 VAE 모델 빌더 — fresh(학습가능) ``AutoencoderKlMaisi``.

이미지 latent와 격자를 맞추려고 MAISI 이미지 VAE와 **같은 아키텍처**(`num_channels [64,128,256]` → 4× 다운샘플,
`latent_channels 4`)를 쓰되, 마스크용으로 **in/out 채널만 override**한다:
  - in_channels  = 1    (raw pseudo-intensity 라벨)
  - out_channels = 118  (0..117 클래스 logit)
  - latent_channels = 4 (이미지 latent와 동일 → Stage 3 concat 시 in_ch 8)

아키텍처 kwargs는 손으로 재선언하지 않고 번들 `autoencoder_def`를 `ConfigParser`로 읽어 상속
([src/baselines/maisi.py](../../src/baselines/maisi.py) 방식). `load_frozen`은 freeze + 이미지 가중치 로드라
쓰지 않는다 — 여기선 **scratch 초기화 + 학습가능**.

⚠️ `norm_float16=True` (MaisiGroupNorm3D) → forward는 `autocast("cuda", fp16)` 안에서 호출해야 한다
(baseline-clone #7).
"""

from __future__ import annotations

import copy

import torch
from monai.bundle import ConfigParser

from src.baselines.maisi import MAISI_INFERENCE_CONFIG


def build_mask_vae(
    in_channels: int = 1,
    out_channels: int = 118,
    latent_channels: int = 4,
    norm_float16: bool = False,
    num_splits: int = 1,
) -> torch.nn.Module:
    """번들 `autoencoder_def`에서 in/out/latent만 바꿔 학습가능 마스크 VAE를 만든다.

    Args:
        in_channels: VAE 입력 채널 (raw = 1).
        out_channels: 재구성 클래스 수 (TS v2 배경 포함 118).
        latent_channels: latent 채널 (이미지 latent와 동일하게 4).

    Returns:
        ``AutoencoderKlMaisi`` (scratch 초기화, 모든 파라미터 ``requires_grad=True``).
        forward(x): ``x (B,in,·) → (recon_logits (B,out,·), z_mu (B,latent,·), z_sigma (B,latent,·))``.
    """
    parser = ConfigParser()
    parser.read_config(str(MAISI_INFERENCE_CONFIG))
    # 번들 def(@spatial_dims 등 참조 포함)를 복제 후 채널만 리터럴로 덮어씀.
    ae_def = copy.deepcopy(parser["autoencoder_def"])
    ae_def["in_channels"] = in_channels
    ae_def["out_channels"] = out_channels
    ae_def["latent_channels"] = latent_channels
    # scratch 학습엔 fp16 groupnorm이 불필요·불안정 → 기본 False (MAISI 추론 최적화는 True).
    ae_def["norm_float16"] = norm_float16
    # num_splits>1은 split-conv가 GPU→CPU 복사(Memcpy DtoH)로 ~수십× 느려짐([[maisi-encode-gpu-cat-speedup]]).
    # 작은 모델·64³ 패치엔 split 불필요 → 기본 1로 꺼서 대폭 가속.
    ae_def["num_splits"] = num_splits
    parser["mask_autoencoder_def"] = ae_def
    ae = parser.get_parsed_content(
        "mask_autoencoder_def"
    )  # AutoencoderKlMaisi 인스턴스화
    ae.train()
    return ae


__all__ = ["build_mask_vae"]
