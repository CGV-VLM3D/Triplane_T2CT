"""CLIP3D-conditioned inference sampler for VLM3D evaluation.

Same Report2CT backbone as report2ct_fvlm, with text2ct's FrozenCLIP3D condition. Inherits
the whole load/RFlow-denoise/MAISI-decode/HU-save path from `Report2CTFVLMSampler`: the
precomputed `.pt` is loaded per scan id and unsqueezed to a batch in `_case_to_context`,
which is shape-agnostic. The ONLY difference is the UNet `cross_attention_dim` — 768 for the
CLIP3D condition `(1, 768)` instead of fVLM's 256.

The condition is precomputed by `scripts/precompute_clip3d_cond_embeddings.py` →
`data/text2ct_clip3d/cond_embeddings/<split>/<scan_id>.pt` (each a `(1, 768)` tensor), keyed
by scan id exactly like fVLM, so `cond_dir` / `_case_to_context` are reused unchanged. Spacing
stays Report2CT's `[1.0, 1.0, 1.5]` (the latents are Report2CT `_emb.nii.gz`), set via config.
"""

from __future__ import annotations

from src.eval.samplers.report2ct_fvlm import Report2CTFVLMSampler


class Report2CTCLIP3DSampler(Report2CTFVLMSampler):
    """Report2CT backbone + text2ct FrozenCLIP3D conditioning, loaded from precomputed `.pt`.

    Identical to `Report2CTFVLMSampler` except the UNet cross-attention width: the CLIP3D
    condition is `(1, 768)`, so `cross_attention_dim = 768`. Pass `cond_dir` pointing at the
    CLIP3D embeddings (`data/text2ct_clip3d/cond_embeddings/valid`).
    """

    cross_attention_dim = (
        768  # FrozenCLIP3D condition dim (configs/model/report2ct_clip3d.yaml)
    )


__all__ = ["Report2CTCLIP3DSampler"]
