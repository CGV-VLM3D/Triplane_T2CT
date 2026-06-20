"""CT-CLIP backbone adapter (3D CT ↔ radiology text alignment).

Wraps `third_party/ct_clip/` (paper code from ibrahimethemhamamci/CT-CLIP) plus
the pretrained `CT-CLIP_v2.pt` checkpoint without modifying the submodule
(P2: `third_party/` is read-only). Loaded weights provide the contrastive
alignment used by all three VLM3D 2026 eval dockers (ctgen / reportgen /
abnclass), so this is the de-facto reference encoder.

Surface (uniform across CT-CLIP / fVLM):
    image_dim, text_dim : int    -- documented latent dims (post-projection, L2-normalized)
    encode_image(vol)            -- (B, 1, D, H, W) -> (B, image_dim)
    encode_text(input_ids, mask) -- (B, L)          -> (B, text_dim)
    tokenize(text)               -- str | list[str] -> dict(input_ids, attention_mask)

Volume convention (mirrors third_party/ct_clip/scripts/run_zero_shot.py:11-21):
    image_size=480, patch_size=20  → 24 spatial patches per side
    temporal_patch_size=10         → depth must be a multiple of 10 after preprocessing
    channels=1 (grayscale CT)

Lazy build pattern (see generatect_adapter.py:228-240): adapter constructs
without weights so tests / config compose can pass before the runbook
download step. The first `encode_*` or `tokenize` call triggers
`_ensure_built()` which (a) builds the modules, (b) loads `CT-CLIP_v2.pt`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------
# Submodule paths — `transformer_maskgit/` (for CTViT) and `CT_CLIP/` (for the
# CTCLIP wrapper). Both ct_clip and generatect ship their own copies of a
# top-level `transformer_maskgit` package whose `CTViT.forward` signatures
# differ (ct_clip's accepts `return_encoded_tokens`, generatect's does not),
# so whichever loads first poisons `sys.modules` for the other. We therefore
# (a) do NOT mutate sys.path at module import, and (b) scrub + re-prioritize
# inside `_ensure_built` so the most recent caller wins. Same pattern as
# `fvlm_adapter._ensure_built` (which handles the analogous lavis-package
# collision for fVLM).
# ----------------------------------------------------------------------------
_CTCLIP_REPO: Final[Path] = Path("/workspace/third_party/ct_clip")
_CTCLIP_PKG_DIR: Final[Path] = _CTCLIP_REPO / "CT_CLIP"
_TM_PKG_DIR: Final[Path] = _CTCLIP_REPO / "transformer_maskgit"
_GENERATECT_TM_PKG_DIR: Final[Path] = Path(
    "/workspace/third_party/generatect/transformer_maskgit"
)

CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/ctclip")
DEFAULT_CKPT: Final[Path] = CKPT_DIR / "CT-CLIP_v2.pt"

# HF text encoder (also provides the BertTokenizer). Caches under HF_HOME.
HF_TEXT_ENCODER: Final[str] = "microsoft/BiomedVLP-CXR-BERT-specialized"

# DUPLICATION INTENTIONAL — these are the v2-pretrained kwargs the CT-CLIP
# authors used; transcribed 1:1 from third_party/ct_clip/scripts/run_zero_shot.py:11-34.
# If the upstream repo ever exposes these via a config file, switch to ConfigParser.
_CTVIT_KWARGS: Final[dict] = {
    "dim": 512,
    "codebook_size": 8192,
    "image_size": 480,
    "patch_size": 20,
    "temporal_patch_size": 10,
    "spatial_depth": 4,
    "temporal_depth": 4,
    "dim_head": 32,
    "heads": 8,
}
_CLIP_KWARGS: Final[dict] = {
    "dim_image": 294912,  # CTViT flattened patch dim — see run_zero_shot.py:26
    "dim_text": 768,  # BiomedVLP-CXR-BERT hidden — see run_zero_shot.py:27
    "dim_latent": 512,  # contrastive latent space — see run_zero_shot.py:28
    "extra_latent_projection": False,
    "use_mlm": False,
    "downsample_image_embeds": False,
    "use_all_token_embeds": False,
}


class CTCLIPBackbone(nn.Module):
    """CT-CLIP backbone with the uniform VLM-backbone contract.

    Args:
        ckpt_path: Path to CT-CLIP_v2.pt. If missing at construction time, the
            adapter still builds (lazy weights) so the module is importable in
            CI. Weights are loaded on the first `encode_*` / `tokenize` call.
        device_str: Where to place the model after build. Default 'cpu' so a
            unit test can construct without a GPU; task modules override at
            train/eval time.
        load_weights: If False, build the modules with random init (useful for
            architecture sanity tests). Default True.
    """

    image_dim: int = _CLIP_KWARGS["dim_latent"]
    text_dim: int = _CLIP_KWARGS["dim_latent"]

    def __init__(
        self,
        ckpt_path: str | Path = DEFAULT_CKPT,
        device_str: str = "cpu",
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        self._ckpt_path = Path(ckpt_path)
        self._device_str = device_str
        self._load_weights = load_weights
        self._clip: nn.Module | None = None
        self._tokenizer = None

    def _ensure_built(self) -> None:
        if self._clip is not None:
            return
        # ct_clip and generatect both ship a top-level `transformer_maskgit`
        # package with DIFFERENT `CTViT.forward` signatures (ct_clip accepts
        # `return_encoded_tokens=True`, generatect's older copy does not). If
        # generatect_adapter was built earlier in the same process, its
        # transformer_maskgit is cached in sys.modules and our `from
        # transformer_maskgit import CTViT` would resolve to its CTViT,
        # causing `TypeError: CTViT.forward() got an unexpected keyword
        # argument 'return_encoded_tokens'`. Scrub the cache, remove the
        # generatect path if present, and prepend our path. Same fix pattern
        # as `fvlm_adapter._ensure_built` for the lavis collision.
        for _name in [
            m
            for m in sys.modules
            if m == "transformer_maskgit" or m.startswith("transformer_maskgit.")
        ]:
            sys.modules.pop(_name, None)
        _genct_tm_str = str(_GENERATECT_TM_PKG_DIR)
        _genct_tm_had_path = _genct_tm_str in sys.path
        while _genct_tm_str in sys.path:
            sys.path.remove(_genct_tm_str)
        for _p in (_TM_PKG_DIR, _CTCLIP_PKG_DIR):
            while str(_p) in sys.path:
                sys.path.remove(str(_p))
            sys.path.insert(0, str(_p))

        try:
            # Imports deferred so adapter can be imported without the heavy deps
            # being installed (matches generatect_adapter._import_transformer_maskgit).
            from transformer_maskgit import CTViT  # noqa: PLC0415
            from ct_clip import CTCLIP  # noqa: PLC0415
            from transformers import BertModel, BertTokenizer  # noqa: PLC0415
        finally:
            # Restore generatect's path entry so a *later* GenerateCTAdapter
            # build in the same process can scrub-and-reimport its own version.
            if _genct_tm_had_path and _genct_tm_str not in sys.path:
                sys.path.append(_genct_tm_str)

        # NOTE: no monkey-patch here (unlike generatect_adapter). CT-CLIP upstream
        # hardcodes `device=torch.device('cuda')` in ~7 places across attention.py /
        # ctvit.py, and the only ContinuousPositionBias caller — `CTViT.encode`
        # (ctvit.py:292-293) — passes `device=cuda` *explicitly*. So the encoder runs
        # on cuda:0 only: a CPB patch would be a no-op on cuda:0 and still can't fix a
        # CPU run (Attention/AlibiPositionalBias at attention.py:135/171/195/219 fire
        # regardless). To target a different physical GPU use CUDA_VISIBLE_DEVICES=N
        # (remaps to cuda:0) — do NOT pass device_str='cuda:1'. (generatect_adapter
        # DOES patch CPB because its MaskGit path passes the real x.device
        # — MaskGITTransformer.py:177,184 — which does not apply to CT-CLIP.)
        tokenizer = BertTokenizer.from_pretrained(HF_TEXT_ENCODER, do_lower_case=True)
        text_encoder = BertModel.from_pretrained(HF_TEXT_ENCODER)
        text_encoder.resize_token_embeddings(len(tokenizer))
        image_encoder = CTViT(**_CTVIT_KWARGS)

        clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            **_CLIP_KWARGS,
        )

        if self._load_weights:
            if not self._ckpt_path.is_file():
                raise FileNotFoundError(
                    f"CT-CLIP checkpoint missing at {self._ckpt_path}. "
                    f"See docs/vlm_baselines_runbook.md for the download recipe."
                )
            # Don't call upstream `clip.load()` — it does strict=True, which fails on
            # benign HF-transformers drift (CT-CLIP_v2.pt was saved with an older
            # transformers that persisted `text_transformer.embeddings.position_ids`
            # as a buffer; current versions don't). Load with strict=False and log.
            state_dict = torch.load(
                str(self._ckpt_path), map_location="cpu", weights_only=False
            )
            missing, unexpected = clip.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                import logging  # noqa: PLC0415

                logging.getLogger(__name__).info(
                    "CT-CLIP load: %d missing, %d unexpected keys (e.g. %s)",
                    len(missing),
                    len(unexpected),
                    (unexpected[:1] or missing[:1] or ["none"])[0],
                )

        clip.eval()
        clip.to(self._device_str)
        self._clip = clip
        self._tokenizer = tokenizer

    # --- grad-enabled access (parity with FVLMBackbone.model) ----------------
    @property
    def clip(self) -> nn.Module:
        """Built CT-CLIP module (CTCLIP), lazily constructed + weight-loaded.

        Parity with `FVLMBackbone.model`. Exists so callers that need a
        *grad-enabled* forward (e.g. Grad-CAM / saliency) can drive the encoder
        submodules directly — the `encode_image` / `encode_text` methods are
        `@torch.no_grad()` and so cannot back-propagate to the input volume.
        Read-only; does not relax the no_grad guard on the encode_* path.
        """
        self._ensure_built()
        return self._clip

    # --- uniform contract ---------------------------------------------------
    # 실질적으로 아래의 코드만 보면
    def tokenize(self, text: str | list[str]) -> dict:
        """Tokenize text for the CT-CLIP text encoder.

        Args:
            text: a single report string, or a batch list of strings.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` tensors, each
            ``(B, 512)`` (max_length padding), on CPU.
        """
        self._ensure_built()
        out = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    @torch.no_grad()
    def encode_image(self, vol: torch.Tensor) -> torch.Tensor:
        """Encode a CT volume to the contrastive image latent, L2-normalized.

        Args:
            vol: CT volume, ``(B, 1, D, H, W)``. Must satisfy CTViT patch
                divisibility (image_size=480, patch_size=20, temporal_patch_size=10
                → ``D % 10 == 0``, ``H == W == 480``). Intensity convention HU/1000
                in ``[-1, 1]`` (same as data_inference_nii.py:115-128). Caller owns
                preprocessing; only the shape is checked here so an experimental
                normalization can still be passed through.

        Returns:
            Image latent, ``(B, image_dim)`` (image_dim = 512), L2-normalized.

        Mirrors third_party/ct_clip/CT_CLIP/ct_clip/ct_clip.py:715-740 — encoded
        tokens are temporally averaged then flattened to (B, 294912) before the
        contrastive projection. Calling `visual_transformer(vol)` (CTViT.forward)
        without `return_encoded_tokens=True` would invoke the decoder, whose
        weights are not in CT-CLIP_v2.pt and would AttributeError.
        """
        self._ensure_built()
        assert vol.ndim == 5 and vol.shape[1] == 1, (
            f"expected (B,1,D,H,W), got {tuple(vol.shape)}"
        )
        assert vol.shape[2] % 10 == 0 and vol.shape[3] == 480 and vol.shape[4] == 480, (
            f"CTViT requires D%10==0, H=W=480; got D={vol.shape[2]} "
            f"H={vol.shape[3]} W={vol.shape[4]}"
        )
        clip = self._clip
        vol = vol.to(next(clip.parameters()).device, non_blocking=True)
        enc = clip.visual_transformer(vol, return_encoded_tokens=True)  # (B,T,H,W,D)
        enc = torch.mean(enc, dim=1)  # temporal pool → (B,H,W,D)
        enc = enc.view(enc.shape[0], -1)  # (B, 294912)
        image_latent = clip.to_visual_latent(enc)  # (B, 512)
        return F.normalize(image_latent, dim=-1)

    @torch.no_grad()
    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode tokenized text to the contrastive text latent, L2-normalized.

        Args:
            input_ids: token ids, ``(B, L)`` (L = 512).
            attention_mask: padding mask, ``(B, L)``.

        Returns:
            Text latent, ``(B, text_dim)`` (text_dim = 512), L2-normalized.

        Mirrors third_party/ct_clip/CT_CLIP/ct_clip/ct_clip.py:762-765 — take
        the CLS hidden state from `last_hidden_state[:, 0]` (NOT pooler_output)
        and project through `to_text_latent`.
        """
        self._ensure_built()
        clip = self._clip
        device = next(clip.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        text_output = clip.text_transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_hidden = text_output.last_hidden_state[:, 0]  # class token output
        text_latent = clip.to_text_latent(cls_hidden)
        return F.normalize(text_latent, dim=-1)
