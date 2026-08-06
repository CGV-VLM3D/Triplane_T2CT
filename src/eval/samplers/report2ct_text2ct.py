"""report2ct_text2ct inference sampler for VLM3D evaluation.

The `report2ct_text2ct` experiment trains the Report2CT backbone (report2ct code) on
**text2ct's** VAE latents + FrozenCLIP3D condition — an A/B to check whether Report2CT's
training loop reproduces text2ct's result. So generation must match text2ct's latent
geometry, NOT Report2CT's:

  * text2ct latent = ``(4, 128, 128, 32)`` → MAISI-decode → ``512×512×128`` volume
  * Report2CT latent = ``(4, 120, 120, 64)`` → ``480×480×256`` (what the parent samples)

The MAISI VAE is identical (same ``autoencoder.pt``; text2ct's ``autoencoder_epoch273.pt``
is byte-identical), so only the sampled latent shape (and the saved/conditioned spacing,
set to text2ct's ``0.75/0.75/3.0`` via the eval config) differ from the CLIP3D parent.
"""

from __future__ import annotations

from src.eval.samplers.report2ct_clip3d import Report2CTCLIP3DSampler


class Report2CTText2CTSampler(Report2CTCLIP3DSampler):
    """Report2CT backbone + FrozenCLIP3D cond, sampling text2ct's ``(4,128,128,32)`` latent.

    Identical to ``Report2CTCLIP3DSampler`` (CLIP3D 768-d condition from precomputed ``.pt``)
    except the RFlow loop samples noise at text2ct's latent geometry so the MAISI decode
    yields text2ct's native ``512×512×128`` volume. Pass ``spacing_mm=[0.75, 0.75, 3.0]``
    via config to match text2ct's conditioning/affine.
    """

    _latent_shape = (4, 128, 128, 32)  # text2ct VAE latent (C, H, W, D)


__all__ = ["Report2CTText2CTSampler"]
