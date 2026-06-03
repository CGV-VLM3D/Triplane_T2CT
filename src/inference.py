from typing import Any, Dict

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# Same convention as src/train.py and src/eval.py: rootutils sets PROJECT_ROOT so
# configs/paths/default.yaml resolves correctly, and adds the project root to
# PYTHONPATH so `from src.baselines... import ...` works without `pip install -e`.
# ------------------------------------------------------------------------------------ #

from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def inference(cfg: DictConfig) -> Dict[str, Any]:
    """Run text -> CT volume inference for every prompt in cfg.prompts.

    The model is constructed via Hydra `instantiate(cfg.model)`, so any adapter
    that exposes a ``.inference(prompt, **kwargs) -> torch.Tensor`` method
    (shape ``(1, 1, D, H, W)``) can drop into this entrypoint with its own
    ``configs/model/<name>.yaml``. GenerateCT is the first concrete model;
    Report2CT / our own model will plug in the same way.
    """
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving outputs under {out}")

    for entry in cfg.prompts:
        log.info(f"Generating: {entry.name}")
        vol = model.inference(entry.text, **cfg.sampler)
        arr = vol.squeeze().detach().cpu().numpy()
        # Truthful per-model affine so the saved volume carries its real voxel geometry.
        # `output_spacing` is in the squeezed-array axis order, so a diagonal affine
        # reproduces each upstream's `diag(spacing)` save. Adapters that don't expose it
        # fall back to identity (geometry unspecified — fine for visualization only).
        sp = getattr(model, "output_spacing", None)
        if sp is not None:
            affine = np.diag([float(s) for s in sp] + [1.0])
        else:
            log.warning(
                "%s exposes no output_spacing; saving %s with identity affine "
                "(geometry unspecified).",
                type(model).__name__,
                entry.name,
            )
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(arr, affine), str(out / f"{entry.name}.nii.gz"))
        if cfg.save_text_companion:
            (out / f"{entry.name}.txt").write_text(entry.text, encoding="utf-8")
        log.info(f"  saved {entry.name}.nii.gz  shape={tuple(arr.shape)}")

    return {"num_samples": len(cfg.prompts)}


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    inference(cfg)


if __name__ == "__main__":
    main()
