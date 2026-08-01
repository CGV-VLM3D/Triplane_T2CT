"""Report2CT-Wan + frozen SPECTRE teacher tokens LightningDataModule (REPA).

Thin subclass of :class:`~src.data.report2ct_datamodule.Report2CTDataModule` for the
``report2ct_wan_repa`` variant. It emits everything the plain-wan module does — ``image``
(Wan image latent ``(16, 64, 64, 64)``), ``spacing``, ``context_f``/``context_i`` (2560-d
findings+impression) — PLUS ``spectre`` ``(T, 1080)``, the precomputed dense token grid of a
frozen SPECTRE ViT-L (``scripts/precompute_spectre_features.py``).

The teacher path is **derived from the image path** rather than carried in the datalist: both
come out of the same precompute run and share the volume id, so deriving removes a way for them
to drift apart and lets the existing ``datalist_wan_2560.json`` be reused untouched.

Grid choice is `spectre_dir`: ``spectre_ssl_16`` gives ``(4096, 1080)`` (8.85 MB/scan — the 44 GB
train set stays resident in page cache) and ``spectre_ssl_32`` gives ``(32768, 1080)``
(70.8 MB/scan, the paper-faithful target). ``RepaAligner.teacher_grid`` must match.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, CopyItemsd, Lambdad

from src.data.report2ct_datamodule import Report2CTDataModule, _build_transforms

IMAGE_SUFFIX = "_emb.nii.gz"


def _teacher_from_image_path(image_path: str, spectre_dir: Path) -> torch.Tensor:
    """``…/<id>_emb.nii.gz`` → the matching teacher grid ``(T, 1080)`` float32.

    Module-level (bound to `spectre_dir` via `functools.partial`) rather than a method: MONAI
    pickles the transform to every dataloader worker, and a bound method would drag the whole
    DataModule — CacheDatasets included — along with it.

    Raises:
        FileNotFoundError: if the volume has no precomputed teacher. Skipping it silently would
            quietly change the training set, so this is fatal.
    """
    vol_id = Path(image_path).name[: -len(IMAGE_SUFFIX)]
    path = spectre_dir / f"{vol_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"no SPECTRE teacher features for {vol_id} at {path} — finish "
            f"scripts/precompute_spectre_features.py before training with REPA."
        )
    return torch.from_numpy(np.load(path)).float()


class Report2CTWanRepaDataModule(Report2CTDataModule):
    """Report2CT-Wan datamodule that additionally loads frozen SPECTRE teacher tokens."""

    def __init__(self, *args, spectre_dir: str | Path, **kwargs) -> None:
        """
        Args:
            spectre_dir: Directory of ``<volume_id>.npy`` teacher grids, e.g.
                ``/workspace/data/report2ct_wan/spectre_ssl_16``.
        """
        super().__init__(*args, **kwargs)
        self.spectre_dir = Path(spectre_dir)

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets, then check every entry has a teacher grid before training starts.

        Without this the first missing file surfaces minutes later inside a dataloader worker,
        after the 233M UNet has already been built. It also catches a mismatch that is easy to
        miss: the datalist is the only thing that decides which volumes are trained on, and it
        has drifted from ``data/ctrate_toy_v2/*/ids.json`` before (a stale wan datalist once
        disagreed with the canonical ``valid_v2`` list on 603 of 1304 scans). Precomputing
        "all of valid_v2" is therefore *not* the same as covering this datalist — check the
        datalist itself.
        """
        super().setup(stage)
        missing = [
            vol_id
            for entries in (self.train_files, self.val_files)
            for entry in entries
            for vol_id in [Path(entry["image"]).name[: -len(IMAGE_SUFFIX)]]
            if not (self.spectre_dir / f"{vol_id}.npy").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.train_files) + len(self.val_files)} datalist "
                f"entries have no teacher grid in {self.spectre_dir} (e.g. {missing[:3]}). "
                f"Run scripts/precompute_spectre_features.py over the DATALIST's ids — note "
                f"they are not identical to data/ctrate_toy_v2/*/ids.json."
            )

    def _transforms(self) -> Compose:
        """Base report2ct transforms, preceded by deriving + loading the teacher grid.

        The copy has to happen **before** the base pipeline's ``LoadImaged``, which replaces the
        ``image`` path string with a tensor.
        """
        base = _build_transforms(self.spacing_multiplier)
        return Compose(
            [
                CopyItemsd(keys="image", names="spectre"),  # path string → "spectre"
                Lambdad(
                    keys="spectre",
                    func=partial(
                        _teacher_from_image_path, spectre_dir=self.spectre_dir
                    ),
                ),  # → (T, 1080)
                *base.transforms,
            ]
        )


__all__ = ["Report2CTWanRepaDataModule"]
