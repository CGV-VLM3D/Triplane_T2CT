"""SPECTRE backbone adapter (CT-only 3D ViT foundation model, frozen teacher).

Wraps `third_party/spectre/` (cclaess/SPECTRE, arXiv:2511.17209) plus the published
ViT-Large backbones without modifying the submodule (P2: `third_party/` is read-only).
Used as the **REPA alignment teacher** for `report2ct_wan`: its dense patch tokens are the
target that the diffusion U-Net's middle-stage feature is aligned to.

Surface (this adapter does NOT share CT-CLIP's `encode_image`/`encode_text` contract —
SPECTRE is a vision-only encoder whose product is a dense 3D token grid):
    dense_dim, global_dim : int                     -- token / scan-level dims (1080)
    window(vol_hu)                 -> (crops, grid)  -- HU windowing + crop tiling, no model
    encode_crops(crops, grid, ...) -> (dense, global) -- ONE backbone pass, both products
    encode_dense(vol_hu, ...)      -> (Gh, Gw, Gd, 1080)
    encode_global(vol_hu)          -> (1080,)

Volume convention (mirrors third_party/spectre/src/spectre/windowing.py):
    crop_size  = (128, 128, 64)  voxels, channel-first ``(C, H, W, D)``
    patch_size = (16, 16, 8)     → 8×8×8 = 512 patch tokens per crop (+1 CLS at index 0)
    RAS orientation, raw Hounsfield Units in, HU window [-1000, 1000] → [0, 1] applied
    internally by `window_scan`.

Two silent-bug guards live here (both verified in tests/repa_probe/u0_smoke/REPORT.md):
  * **Depth must already be a multiple of 64.** `window_scan` CENTER-CROPS to the largest
    whole multiple of the crop size (`windowing.py:largest_multiple_crop_size`, `(s // c) * c`);
    it only pads axes *smaller* than one crop. A 253-slice volume silently loses 61 slices
    (24 %). The caller owns the pad; `window()` raises instead of cropping.
  * **The CLS token must be dropped before reshaping to the grid.** `num_prefix_tokens == 1`
    for every preset; keeping it shifts every token's spatial correspondence by one.

Lazy build pattern (see ctclip_adapter.py:20-23): the adapter constructs without weights so
tests / config compose pass before the download step. The first `encode_*` / `window` call
triggers `_ensure_built()`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import torch
from torch import nn

# ----------------------------------------------------------------------------
# Submodule path. Unlike ct_clip/generatect there is no package-name collision to
# scrub, but we still insert lazily (not at module import) so importing this file
# is free and the path order can't surprise another adapter built earlier.
# ----------------------------------------------------------------------------
_SPECTRE_REPO: Final[Path] = Path("/workspace/third_party/spectre")
_SPECTRE_PKG_DIR: Final[Path] = _SPECTRE_REPO / "src"

CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/spectre")
#: SSL-only backbone (DINO + iBOT + KoLeo, no vision-language alignment).
DEFAULT_CKPT: Final[Path] = (
    CKPT_DIR / "spectre_backbone_vit_large_patch16_128_no_vla.pt"
)
#: SSL + vision-language (SigLIP on paired reports) backbone.
CKPT_VLA: Final[Path] = CKPT_DIR / "spectre_backbone_vit_large_patch16_128.pt"
#: Scan-level combiner (global ViT over crop embeddings); only needed for `encode_global`.
CKPT_COMBINER: Final[Path] = CKPT_DIR / "spectre_combiner_feature_vit_large.pt"

#: Preset whose architecture kwargs we build from. Read from `spectre.presets` rather than
#: transcribed, so there is no config-vs-hardcode drift (audit #3).
PRESET: Final[str] = "spectre-large"

CROP_SIZE: Final[tuple[int, int, int]] = (128, 128, 64)
PATCH_SIZE: Final[tuple[int, int, int]] = (16, 16, 8)
HU_RANGE: Final[tuple[float, float]] = (-1000.0, 1000.0)


def _install_hf_shim() -> None:
    """Give old `huggingface_hub` the `load_state_dict_from_file` symbol SPECTRE imports.

    `spectre/utils/modeling.py:14` imports it at module level; our pinned huggingface_hub is
    0.26.3 and the symbol landed in 0.30+. It is only *called* on the HF-URL branch
    (`modeling.py:77`), which local-path loads never take — so this exists purely to let the
    import succeed. We do not upgrade huggingface_hub: transformers 4.46 / diffusers 0.31 pin
    against it. Same spirit as the numpy-2.x shim in `src/eval/tasks/_fid_runner.py`.
    """
    import huggingface_hub  # noqa: PLC0415

    if hasattr(huggingface_hub, "load_state_dict_from_file"):
        return

    def load_state_dict_from_file(
        checkpoint_file, map_location=None, weights_only=True
    ):
        path = str(checkpoint_file)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415

            return load_file(path, device=str(map_location or "cpu"))
        return torch.load(
            path, map_location=map_location or "cpu", weights_only=weights_only
        )

    huggingface_hub.load_state_dict_from_file = load_state_dict_from_file


def import_spectre():
    """Make `third_party/spectre/src` importable and return the `spectre` package.

    Public because the adapter is not the only consumer: `tests/test_spectre_adapter.py` and
    the probes under `tests/repa_probe/` need `spectre.windowing` without paying for a 338M
    random-init build.
    """
    _install_hf_shim()
    pkg = str(_SPECTRE_PKG_DIR)
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    import spectre  # noqa: PLC0415

    return spectre


class SpectreBackbone(nn.Module):
    """Frozen SPECTRE image encoder producing a dense 3D token grid.

    Args:
        ckpt_path: Backbone `.pt`. `DEFAULT_CKPT` is the SSL-only variant; pass `CKPT_VLA`
            for the vision-language-aligned one. If missing at construction time the adapter
            still builds (lazy weights) so the module is importable in CI.
        device_str: Where to place the model after build. Default 'cpu' so a unit test can
            construct without a GPU; task modules override at train/eval time.
        load_weights: If False, build with random init (architecture sanity tests only).
        with_combiner: Also build the scan-level feature combiner. Required for
            `encode_global`; adds ~234 MB of weights and is off by default because the REPA
            target is the dense grid, not the global vector.
        combiner_ckpt_path: Combiner `.pt`, used only when `with_combiner`.
    """

    dense_dim: int = 1080
    global_dim: int = 1080
    crop_size: tuple[int, int, int] = CROP_SIZE
    patch_size: tuple[int, int, int] = PATCH_SIZE

    def __init__(
        self,
        ckpt_path: str | Path = DEFAULT_CKPT,
        device_str: str = "cpu",
        load_weights: bool = True,
        with_combiner: bool = False,
        combiner_ckpt_path: str | Path = CKPT_COMBINER,
    ) -> None:
        super().__init__()
        self._ckpt_path = Path(ckpt_path)
        self._combiner_ckpt_path = Path(combiner_ckpt_path)
        self._device_str = device_str
        self._load_weights = load_weights
        self._with_combiner = with_combiner
        self._model: nn.Module | None = None
        self._window_scan = None

    def _ensure_built(self) -> None:
        if self._model is not None:
            return
        spectre = import_spectre()
        SpectreImageFeatureExtractor = spectre.SpectreImageFeatureExtractor
        window_scan = spectre.window_scan
        preset = spectre.presets.get_preset(PRESET)
        kwargs: dict = {
            "backbone_name": preset.backbone,
            "backbone_kwargs": preset.backbone_kwargs,
        }
        if self._with_combiner:
            kwargs["feature_combiner_name"] = preset.feature_combiner
            kwargs["feature_combiner_kwargs"] = preset.feature_combiner_kwargs

        if self._load_weights:
            for path, what in ((self._ckpt_path, "backbone"),) + (
                ((self._combiner_ckpt_path, "combiner"),) if self._with_combiner else ()
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"SPECTRE {what} checkpoint missing at {path}. See "
                        f"docs/submodule_pins.md for the huggingface-cli download recipe."
                    )
            # Upstream loads with strict=True by default and the published checkpoints match
            # exactly ("All keys matched successfully"); we keep that strictness because a
            # mismatch here would silently mean a differently-initialised teacher.
            kwargs["backbone_checkpoint_path_or_url"] = str(self._ckpt_path)
            if self._with_combiner:
                kwargs["feature_combiner_checkpoint_path_or_url"] = str(
                    self._combiner_ckpt_path
                )

        model = SpectreImageFeatureExtractor(**kwargs)
        model.eval().requires_grad_(False).to(self._device_str)
        self._model = model
        self._window_scan = window_scan

    # --- grad-enabled access (parity with CTCLIPBackbone.clip) ----------------
    @property
    def model(self) -> nn.Module:
        """Built `SpectreImageFeatureExtractor`, lazily constructed + weight-loaded.

        Parity with `CTCLIPBackbone.clip`. Exists so callers needing a grad-enabled forward
        (attention hooks, saliency) can drive the submodules directly — the `encode_*`
        methods are `@torch.no_grad()`. Read-only; does not relax that guard.
        """
        self._ensure_built()
        return self._model

    # --- geometry -------------------------------------------------------------
    @classmethod
    def token_grid(cls, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Token-grid size for a volume of `spatial_shape` ``(H, W, D)``.

        E.g. ``(512, 512, 256)`` → ``(32, 32, 32)`` — exactly 2× coarser per axis than the
        Wan latent ``(64, 64, 64)`` built from the same volume.
        """
        return tuple(s // p for s, p in zip(spatial_shape, cls.patch_size))  # type: ignore[return-value]

    def window(self, vol_hu: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """HU-window and tile a whole volume into the crops the backbone consumes.

        Args:
            vol_hu: ``(1, H, W, D)`` channel-first, **raw Hounsfield Units** (audit #1 —
                SPECTRE windows `[-1000, 1000] → [0, 1]` itself; pre-scaling double-scales).
                Every axis must already be a whole multiple of `crop_size` (see below).

        Returns:
            ``(crops, grid)`` — crops ``(N, 1, 128, 128, 64)`` in ``[0, 1]`` and the crop grid
            ``(n_h, n_w, n_d)`` with ``N == n_h*n_w*n_d``. Crop ``n`` sits at
            ``n = (h*n_w + w)*n_d + d`` (depth varies fastest).

        Raises:
            ValueError: if an axis is not a multiple of `crop_size`. Upstream would silently
                center-crop it away (audit #4); padding is the caller's job so the loss is
                never invisible.
        """
        self._ensure_built()
        if vol_hu.ndim != 4 or vol_hu.shape[0] != 1:
            raise ValueError(f"expected (1, H, W, D), got {tuple(vol_hu.shape)}")
        bad = [
            f"{'HWD'[i]}={int(s)} % {c} = {int(s) % c}"
            for i, (s, c) in enumerate(zip(vol_hu.shape[1:], self.crop_size))
            if int(s) % c
        ]
        if bad:
            raise ValueError(
                f"every axis must be a whole multiple of crop_size {self.crop_size}, got "
                f"{tuple(vol_hu.shape[1:])} ({', '.join(bad)}). window_scan would CENTER-CROP "
                f"the remainder away instead of padding it — pad the volume first "
                f"(e.g. 253 → 256 in depth)."
            )
        return self._window_scan(vol_hu, self.crop_size, hu_range=HU_RANGE)

    # --- encoding -------------------------------------------------------------
    @torch.no_grad()
    def encode_crops(
        self,
        crops: torch.Tensor,
        grid: tuple[int, int, int],
        layer: int | None = None,
        want_global: bool = False,
        max_crops_per_forward: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the backbone **once** over pre-windowed crops and return both products.

        Args:
            crops: ``(N, 1, 128, 128, 64)`` from `window`, already in ``[0, 1]``.
            grid: crop grid ``(n_h, n_w, n_d)``.
            layer: 0-based block index to tap instead of the final normed output. `None`
                uses `forward_features` (the published representation).
            want_global: also return the scan-level embedding. Requires `with_combiner=True`.
            max_crops_per_forward: caps crops per backbone pass to bound memory.

        Returns:
            ``(dense, global_)`` — dense ``(n_h*8, n_w*8, n_d*8, 1080)`` on the crops' device,
            and either the scan-level vector ``(1080,)`` or `None`.
        """
        self._ensure_built()
        model = self._model
        device = next(model.parameters()).device
        crops = crops.to(device, non_blocking=True)

        chunks = []
        for chunk in torch.split(crops, max_crops_per_forward, dim=0):
            if layer is None:
                # (n, 1 + 512, F) — CLS at index 0
                chunks.append(model.backbone.forward_features(chunk))
            else:
                # forward_intermediates already strips prefix tokens → (n, 512, F)
                chunks.append(
                    model.backbone.forward_intermediates(
                        chunk,
                        indices=[layer],
                        return_prefix_tokens=False,
                        norm=True,
                        output_fmt="NLC",
                        intermediates_only=True,
                    )[0]
                )
        tokens = torch.cat(chunks, dim=0)

        n_prefix = int(model.backbone.num_prefix_tokens) if layer is None else 0
        dense = self._crops_to_grid(tokens[:, n_prefix:], grid)

        global_ = None
        if want_global:
            if model.feature_combiner is None:
                raise RuntimeError(
                    "encode_crops(want_global=True) needs the combiner — construct with "
                    "SpectreBackbone(..., with_combiner=True)."
                )
            if layer is not None:
                raise ValueError(
                    "the scan-level embedding is only defined on the final layer; "
                    "call with layer=None."
                )
            # Reuse upstream's pooling rather than re-deriving it: its docstring says it
            # "must stay bit-identical to combine_features" (spectre/model.py), and any
            # divergence would silently move the published embedding.
            pooled = model._pool_crop_tokens(tokens)  # (N, 2F)  noqa: SLF001
            combined = model.feature_combiner(pooled.unsqueeze(0), grid)  # (1, T', F')
            global_ = combined[0, 0]  # scan-level CLS — (F',)

        return dense, global_

    @torch.no_grad()
    def encode_dense(
        self,
        vol_hu: torch.Tensor,
        layer: int | None = None,
        pool_to: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Encode a whole volume to its dense token grid.

        Args:
            vol_hu: ``(1, H, W, D)`` raw HU; every axis a whole multiple of `crop_size`.
            layer: block index to tap, or `None` for the final normed output.
            pool_to: average-pool the grid to this size, e.g. ``(16, 16, 16)``. `None` keeps
                the native ``(H/16, W/16, D/8)``.

        Returns:
            ``(Gh, Gw, Gd, 1080)`` — ``(32, 32, 32, 1080)`` for a 512×512×256 volume.
        """
        crops, grid = self.window(vol_hu)
        dense, _ = self.encode_crops(crops, grid, layer=layer)
        return dense if pool_to is None else self.pool_grid(dense, pool_to)

    @torch.no_grad()
    def encode_global(self, vol_hu: torch.Tensor) -> torch.Tensor:
        """Encode a whole volume to its scan-level embedding, ``(1080,)``.

        Requires `with_combiner=True`. This is the vector SPECTRE aligns to text (SigLIP) in
        the VLA checkpoint — useful as a diagnostic, **not** as the REPA token target.
        """
        crops, grid = self.window(vol_hu)
        _, global_ = self.encode_crops(crops, grid, want_global=True)
        return global_

    # --- helpers --------------------------------------------------------------
    @classmethod
    def _crops_to_grid(
        cls, patch_tokens: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Reassemble per-crop patch tokens into one volume-wide 3D token grid.

        Both orderings are C-order with **depth varying fastest**, which is what makes this a
        plain reshape+permute:
          * crop index — `windowing.py:grid_patch`, ``n = (h*n_w + w)*n_d + d``
          * patch index within a crop — rope `patch_embed` emits ``output_fmt='NHWDC'`` and
            `vision_transformer.py:_pos_embed` flattens with ``x.view(B, -1, C)``
        Proven model-free by the index round-trip in tests/repa_probe/u0_smoke/run.py.

        Args:
            patch_tokens: ``(N, 512, F)`` — **CLS already removed**.
            grid: ``(n_h, n_w, n_d)``.

        Returns:
            ``(n_h*8, n_w*8, n_d*8, F)``.
        """
        n_h, n_w, n_d = grid
        p_h, p_w, p_d = (c // p for c, p in zip(cls.crop_size, cls.patch_size))
        feat = patch_tokens.shape[-1]
        if patch_tokens.shape[0] != n_h * n_w * n_d:
            raise ValueError(f"{patch_tokens.shape[0]} crops != grid {grid} product")
        if patch_tokens.shape[1] != p_h * p_w * p_d:
            raise ValueError(
                f"{patch_tokens.shape[1]} tokens per crop != {p_h * p_w * p_d}; the CLS "
                f"token was probably not removed"
            )
        x = patch_tokens.reshape(
            n_h, n_w, n_d, p_h, p_w, p_d, feat
        )  # (4,4,4, 8,8,8, F)
        x = x.permute(0, 3, 1, 4, 2, 5, 6)  # (4,8, 4,8, 4,8, F)
        return x.reshape(n_h * p_h, n_w * p_w, n_d * p_d, feat)  # (32,32,32, F)

    @staticmethod
    def pool_grid(dense: torch.Tensor, out_grid: tuple[int, int, int]) -> torch.Tensor:
        """Average-pool a token grid ``(Gh, Gw, Gd, F)`` down to `out_grid`.

        Used to trade spatial detail for training-time I/O: ``(32,32,32)`` is 70.8 MB/scan in
        fp16 while ``(16,16,16)`` is 8.85 MB. U-REPA Table 6 measures this trade-off as a real
        (if modest) cost, so which one to train on is decided by measurement, not default.
        """
        g = dense.shape[:3]
        if any(s % o for s, o in zip(g, out_grid)):
            raise ValueError(f"grid {tuple(g)} is not a whole multiple of {out_grid}")
        x = dense.permute(3, 0, 1, 2).unsqueeze(0)  # (1, F, Gh, Gw, Gd)
        x = torch.nn.functional.adaptive_avg_pool3d(x, out_grid)  # (1, F, oh, ow, od)
        return x.squeeze(0).permute(1, 2, 3, 0)  # (oh, ow, od, F)


__all__ = [
    "CKPT_COMBINER",
    "CKPT_DIR",
    "CKPT_VLA",
    "DEFAULT_CKPT",
    "SpectreBackbone",
    "import_spectre",
]
