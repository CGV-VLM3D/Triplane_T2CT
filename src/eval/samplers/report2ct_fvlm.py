"""fVLM-conditioned inference sampler for VLM3D evaluation.

Same Report2CT backbone, conditioner swapped: instead of encoding findings/impression
through 3 text encoders → (1, 2, 2560), this loads the **precomputed** frozen fVLM
per-organ embedding → (1, 4, 256). Using the precomputed `.pt` (rather than re-encoding
in the sampler) guarantees the eval conditioner is bit-identical to what training saw
(`scripts/precompute_fvlm_cond_embeddings.py` → `data/fVLM/cond_embeddings/<split>/`).

Everything else — RFlow denoising loop, sliding-window MAISI decode, HU save, CFG — is
inherited unchanged from `Report2CTSampler`. The only overrides are the UNet's
`cross_attention_dim` (256, via `_load_checkpoint`) and how a case maps to its context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.baselines.maisi import load_frozen
from src.data.fvlm_organ_report import scan_id_from_volume_name
from src.eval.samplers.base import EvalCase
from src.eval.samplers.report2ct import Report2CTSampler, _load_checkpoint

log = logging.getLogger(__name__)


class Report2CTFVLMSampler(Report2CTSampler):
    """Report2CT backbone + fVLM per-organ conditioning, loaded from precomputed `.pt`.

    Args:
        ckpt_path: Lightning .ckpt of a model trained with experiment=report2ct_fvlm.
        cond_dir: directory of precomputed fVLM cond embeddings (`<scan_id>.pt`, each
            a (4, 256) tensor) — typically data/fVLM/cond_embeddings/valid.
        (n_steps / modality_class_label / spacing_mm / cfg_scale / name as Report2CTSampler.)
    """

    # UNet cross-attention dim for the swapped conditioner (matches the trained model's
    # configs/model/report2ct_fvlm.yaml). Subclasses override to load a different-width
    # condition (e.g. report2ct_CLIP3D → 768 for the (1, 768) FrozenCLIP3D vector).
    cross_attention_dim: int = 256

    def __init__(
        self,
        ckpt_path: str,
        cond_dir: str,
        n_steps: int = 100,
        modality_class_label: int = 1,
        spacing_mm: list[float] | None = None,
        cfg_scale: float = 1.0,
        name: str | None = None,  # absorbed from Hydra eval/model config; not used
    ) -> None:
        """Store fVLM-specific config; all else delegated to ``Report2CTSampler.__init__``.

        Args:
            ckpt_path: Lightning ``.ckpt`` trained with ``experiment=report2ct_fvlm``
                (UNet cross_attention_dim=256 instead of 2560).
            cond_dir: directory of precomputed fVLM conditioning embeddings; each file
                is ``<scan_id>.pt`` holding a ``(4, 256)`` float tensor produced by
                ``scripts/precompute_fvlm_cond_embeddings.py``.
        """
        super().__init__(
            ckpt_path=ckpt_path,
            n_steps=n_steps,
            modality_class_label=modality_class_label,
            spacing_mm=spacing_mm,
            cfg_scale=cfg_scale,
            name=name,
        )
        self.cond_dir = Path(cond_dir)

    def _init(self, device: torch.device) -> None:
        """Load UNet (cross_attention_dim=256) + frozen MAISI VAE. No text encoder —
        the conditioner is read from precomputed `.pt` files in `_case_to_context`.
        """
        if self._unet is not None:
            return
        self._device = device

        # Disable FA3 on Blackwell (same as Report2CTModule.setup())
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        log.info("Loading fVLM-cond UNet from %s …", self.ckpt_path)
        self._unet, self._scale_factor = _load_checkpoint(
            self.ckpt_path, device, cross_attention_dim=self.cross_attention_dim
        )

        log.info("Loading frozen MAISI VAE …")
        self._autoencoder = load_frozen(device=device)

    def _case_to_context(self, case: EvalCase) -> torch.Tensor:
        """Load the precomputed fVLM per-organ embedding for one case → (1, 4, 256).

        The embedding is keyed by scan-study id (e.g. 'valid_1000_a'); EvalCase.scan_id
        carries the reconstruction index ('valid_1000_a_1'), so collapse it first.
        """
        scan_id = scan_id_from_volume_name(
            case.scan_id
        )  # 'valid_1000_a_1' → 'valid_1000_a'
        pt_path = self.cond_dir / f"{scan_id}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(
                f"fVLM cond embedding missing: {pt_path}. Run "
                f"scripts/precompute_fvlm_cond_embeddings.py for this split first."
            )
        ctx = torch.load(pt_path, weights_only=True).float()  # (4, 256)
        return ctx.unsqueeze(0).to(self._device)  # (1, 4, 256)


__all__ = ["Report2CTFVLMSampler"]
