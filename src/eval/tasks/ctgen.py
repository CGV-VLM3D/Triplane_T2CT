"""Task 4 (CT generation) evaluator — wraps vlm3d_dockers scripts via subprocess.

Script interfaces (verified by reading the actual source):
  evaluate_fvd.py      : argparse --generated_dir --gt_root --out_json
  evaluate_clip.py     : argparse --generated_dir --gt_root --prompt_xlsx --out_json
                         hardcodes /opt/app/models/{CT-CLIP_v2.pt,BiomedVLP-CXR-BERT-specialized}
  compute_fid_2-5d_ct.py : fire.Fire(main) with kwargs; logs results to logger.info;
                            expects real/synth file-lists of NIfTI/.mha filenames;
                            model downloaded from torch.hub "Warvito/radimagenet-models"
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.eval._vlm3d_paths import ctclip_pkg_parents, ctgen_eval_dir

log = logging.getLogger(__name__)

# Resolved once at import: ct_challenges/ layout (a945900+) with old-path fallback.
_EVAL_DIR = ctgen_eval_dir()
_FVD_SCRIPT = _EVAL_DIR / "evaluate_fvd.py"
_CLIP_SCRIPT = _EVAL_DIR / "evaluate_clip.py"
_FID_SCRIPT = _EVAL_DIR / "compute_fid_2-5d_ct.py"

# evaluate_clip.py hardcodes these paths (docker convention; we symlink into them)
_CLIP_MODELS_DIR = Path("/opt/app/models")
_CLIP_CKPT_EXPECTED = _CLIP_MODELS_DIR / "CT-CLIP_v2.pt"
_CLIP_BERT_EXPECTED = _CLIP_MODELS_DIR / "BiomedVLP-CXR-BERT-specialized"

_CTCLIP_DEFAULT = Path("/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt")

# --- FID-2.5D preprocessing params -------------------------------------------------------- #
# Kept as constants so the subprocess command and the shared-GT-feature cache key can never
# drift: any change here changes the cache key, forcing recomputation (correct-by-construction).
_FID_MODEL = "radimagenet_resnet50"
_FID_TARGET_SHAPE = "512x512x512"
_FID_RESAMPLE = "1.0x1.0x1.0"
_FID_ENABLE_PADDING = True
_FID_ENABLE_CENTER_CROP = True

# Shared GT FID-feature cache. GT features depend ONLY on the GT volume set + the preprocessing
# params above — NOT on the model under evaluation — so they are computed once and reused across
# every model. Without this, each model's _run_fid recomputes the (GPU-heavy) GT forward pass.
# (The volume load/resample still runs: that lives in the read-only upstream FID loop, which
# loads each volume before its `if os.path.isfile(out_fp)` cache check; only the forward is
# skipped when the feature file already exists.)
_SHARED_GT_FEAT_ROOT = Path("/workspace/data/vlm3d_eval/_shared_gt_fidfeat")


def _shared_gt_feat_dir(gt_dir: Path) -> Path:
    """Cache dir for a GT set's FID features, keyed on the GT-set name + all FID params."""
    key = (
        f"{gt_dir.name}__{_FID_MODEL}__{_FID_TARGET_SHAPE}"
        f"__rs{_FID_RESAMPLE}__pad{int(_FID_ENABLE_PADDING)}_crop{int(_FID_ENABLE_CENTER_CROP)}"
    )
    return _SHARED_GT_FEAT_ROOT / key


def _link_shared_gt_features(shared_gt: Path, gt_feat_dir: Path) -> int:
    """Hardlink cached GT features from the shared cache into this run's gt feature dir.

    The upstream loop skips its forward pass when the feature file already exists
    (compute_fid_2-5d_ct.py:611), so pre-linking shared features makes GT reuse a previous
    model's work. Hardlinks share inodes (no extra disk); the loop only reads existing feature
    files, so the shared inode is never mutated. Falls back to copy across filesystems.
    """
    if not shared_gt.is_dir():
        return 0
    linked = 0
    for src in shared_gt.iterdir():
        if not src.is_file():
            continue
        dst = gt_feat_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        linked += 1
    return linked


def _populate_shared_gt_features(gt_feat_dir: Path, shared_gt: Path) -> int:
    """Copy freshly-computed GT features into the shared cache (first model populates it)."""
    shared_gt.mkdir(parents=True, exist_ok=True)
    added = 0
    for src in gt_feat_dir.iterdir():
        if not src.is_file():
            continue
        dst = shared_gt / src.name
        if dst.exists():
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)  # atomic: concurrent readers never see a partial file
        added += 1
    if added:
        log.info("FID GT cache: populated %d new GT features into %s", added, shared_gt)
    return added


def _fid_from_cached_features(
    features_dir: Path, n_gt: int, n_pred: int
) -> dict[str, float] | None:
    """Compute 2.5D-FID on CPU from the per-volume feature files the FID loop already wrote.

    The upstream loop extracts features to ``features_dir/{gt,pred}`` but computes the final
    FID statistics on the GPU, which OOMs at full scale (~512k slice-features per plane across
    6 matrices). Since every per-volume feature file is on disk, we always compute the FID here
    on CPU instead (the host has far more RAM than the GPU). Returns ``None`` if the cached
    features are incomplete (i.e. the subprocess died before finishing extraction).
    """
    import numpy as _np  # noqa: PLC0415

    if not hasattr(
        _np, "float_"
    ):  # MONAI fid._sqrtm uses the NumPy-2-removed np.float_
        _np.float_ = _np.float64
    import torch  # noqa: PLC0415
    from monai.metrics import FIDMetric  # noqa: PLC0415

    gt_files = sorted(p for p in (features_dir / "gt").glob("*") if p.is_file())
    pred_files = sorted(p for p in (features_dir / "pred").glob("*") if p.is_file())
    if len(gt_files) < n_gt or len(pred_files) < n_pred:
        log.error(
            "CPU FID fallback: incomplete cached features (gt %d/%d, pred %d/%d).",
            len(gt_files),
            n_gt,
            len(pred_files),
            n_pred,
        )
        return None

    # Per-volume feature files are tuples (xy, yz, zx); upstream reports zx as XZ.
    plane_keys = {0: "FID_2p5D_XY", 1: "FID_2p5D_YZ", 2: "FID_2p5D_XZ"}
    fids: dict[str, float] = {}
    for idx, key in plane_keys.items():
        real = torch.vstack(
            [torch.load(f, weights_only=True)[idx].float() for f in gt_files]
        )
        synth = torch.vstack(
            [torch.load(f, weights_only=True)[idx].float() for f in pred_files]
        )
        fids[key] = float(FIDMetric()(synth, real))
        del real, synth
    fids["FID_2p5D_Avg"] = (
        fids["FID_2p5D_XY"] + fids["FID_2p5D_YZ"] + fids["FID_2p5D_XZ"]
    ) / 3.0
    log.info(
        "CPU FID fallback: Avg=%.4f (XY=%.4f YZ=%.4f XZ=%.4f)",
        fids["FID_2p5D_Avg"],
        fids["FID_2p5D_XY"],
        fids["FID_2p5D_YZ"],
        fids["FID_2p5D_XZ"],
    )
    return fids


class CTGenEvaluator:
    """Run FVD / CLIPScore / FID-2.5D on a directory of predicted .mha files.

    Args:
        gt_dir: directory with ground-truth ``*.mha`` files (matching pred filenames).
        metrics: dict with boolean flags ``fvd``, ``clip_score``, ``fid_2p5d``.
        ctclip_ckpt: local path to ``CT-CLIP_v2.pt``.  evaluate_clip.py hardcodes
            ``/opt/app/models/CT-CLIP_v2.pt``; we create a symlink there automatically.
        prompt_xlsx: XLSX file with (Names, Text_prompts) columns for CLIPScore I2T.
            If None, CLIPScore I2T is skipped; only I2I is meaningful.
    """

    def __init__(
        self,
        gt_dir: str | Path,
        metrics: dict | None = None,
        ctclip_ckpt: str | Path | None = None,
        prompt_xlsx: str | Path | None = None,
    ) -> None:
        """Store GT dir, metric flags, and weight paths; see class docstring for parameter details."""
        self.gt_dir = Path(gt_dir)
        self.metrics = metrics or {"fvd": True, "clip_score": True, "fid_2p5d": True}
        self.ctclip_ckpt = Path(ctclip_ckpt) if ctclip_ckpt else _CTCLIP_DEFAULT
        self.prompt_xlsx = Path(prompt_xlsx) if prompt_xlsx else None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run enabled metrics; merge results → out_dir/metrics.json."""
        pred_dir = Path(pred_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, float] = {}

        if self.metrics.get("fvd", True):
            results.update(self._run_fvd(pred_dir, out_dir))

        if self.metrics.get("clip_score", True):
            results.update(self._maybe_run_clip(pred_dir, out_dir))

        if self.metrics.get("fid_2p5d", True):
            results.update(self._run_fid(pred_dir, out_dir))

        merged_path = out_dir / "metrics.json"
        with open(merged_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Metrics saved → %s", merged_path)
        return results

    # ------------------------------------------------------------------ #
    #  FVD                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_fvd_paths(self) -> None:
        """FVD/fvd_pytorch.py hardcodes /opt/app/FVD/ctnet/... — symlink it.

        Re-points a stale symlink: after the a945900 reorg an existing
        /opt/app/FVD/ctnet may dangle at the old (pre-ct_challenges) path, where
        ``link.exists()`` is False but ``symlink_to`` still raises FileExistsError.
        """
        opt_fvd_dir = Path("/opt/app/FVD")
        opt_fvd_dir.mkdir(parents=True, exist_ok=True)
        target = (_EVAL_DIR / "FVD" / "ctnet").resolve()
        link = opt_fvd_dir / "ctnet"
        if not target.exists():
            return
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target:
                return  # already correct
            link.unlink()  # stale/wrong → replace
        link.symlink_to(target)
        log.info("Symlinked %s → %s", link, target)

    def _run_fvd(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run evaluate_fvd.py and return a dict with the ``FVD_CTNet`` key.

        Returns NaN for ``FVD_CTNet`` if the subprocess exits non-zero or the
        output JSON is not produced.
        """
        self._setup_fvd_paths()
        out_json = out_dir / "fvd.json"
        cmd = [
            sys.executable,
            str(_FVD_SCRIPT),
            "--generated_dir",
            str(pred_dir),
            "--gt_root",
            str(self.gt_dir),
            "--out_json",
            str(out_json),
        ]
        # `fvd_pytorch.py` imports top-level `ctnet`, a namespace package living in FVD/.
        # The upstream Dockerfile `pip install -e`s it; locally that editable mapping breaks
        # whenever the submodule dir moves (e.g. the a945900 ct_challenges reorg). Putting
        # FVD/ on PYTHONPATH resolves `ctnet` by namespace discovery, independent of any
        # stale editable install.
        env = os.environ.copy()
        fvd_pkg_dir = str((_EVAL_DIR / "FVD").resolve())
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (fvd_pkg_dir, env.get("PYTHONPATH", "")) if p
        )
        rc = self._run(cmd, cwd=_EVAL_DIR, env=env)
        if rc != 0 or not out_json.is_file():
            log.error("FVD evaluation failed (rc=%d).", rc)
            return {"FVD_CTNet": float("nan")}
        with open(out_json) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    #  CLIPScore                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_run_clip(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Guard-check prerequisites then delegate to ``_run_clip``; return NaN dict on skip.

        Returns NaN values for ``CLIPScore``, ``CLIPScore_I2I``, and ``CLIPScore_mean``
        when the CT-CLIP_v2.pt checkpoint is missing or ``/opt/app/models/`` cannot be
        set up; otherwise returns whatever ``_run_clip`` produces.
        """
        _nan = {
            "CLIPScore": float("nan"),
            "CLIPScore_I2I": float("nan"),
            "CLIPScore_mean": float("nan"),
        }
        if not self.ctclip_ckpt.is_file():
            log.warning(
                "CT-CLIP_v2.pt not found at %s — skipping CLIPScore.\n"
                '  Download: python -c "from huggingface_hub import hf_hub_download; '
                "hf_hub_download('ibrahimhamamci/CT-RATE', 'models/CT-CLIP-Related/CT-CLIP_v2.pt', "
                "repo_type='dataset')\"",
                self.ctclip_ckpt,
            )
            return _nan

        # evaluate_clip.py hardcodes /opt/app/models/ — set up symlinks
        if not self._setup_clip_paths():
            log.warning("Could not set up /opt/app/models/ paths — skipping CLIPScore.")
            return _nan

        return self._run_clip(pred_dir, out_dir)

    def _setup_clip_paths(self) -> bool:
        """Create /opt/app/models/ and symlink CT-CLIP weights + BiomedVLP-BERT."""
        try:
            _CLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Symlink CT-CLIP_v2.pt
            if not _CLIP_CKPT_EXPECTED.exists():
                _CLIP_CKPT_EXPECTED.symlink_to(self.ctclip_ckpt.resolve())
                log.info("Symlinked CT-CLIP_v2.pt → %s", self.ctclip_ckpt)

            # BiomedVLP-CXR-BERT: try HF cache snapshot first
            if not _CLIP_BERT_EXPECTED.exists():
                self._symlink_biomedvlp()

            return _CLIP_CKPT_EXPECTED.exists() and _CLIP_BERT_EXPECTED.exists()
        except PermissionError as e:
            log.warning("Cannot create /opt/app/models/: %s", e)
            return False

    def _symlink_biomedvlp(self) -> None:
        """Symlink BiomedVLP-CXR-BERT-specialized from HF cache or download it."""
        import os

        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        # Try common HF hub cache patterns
        candidates = list(
            hf_home.glob(
                "hub/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/*/"
            )
        )
        if candidates:
            snapshot = sorted(candidates)[-1]  # latest snapshot
            _CLIP_BERT_EXPECTED.symlink_to(snapshot.resolve())
            log.info("Symlinked BiomedVLP-CXR-BERT-specialized → %s", snapshot)
        else:
            log.warning(
                "BiomedVLP-CXR-BERT-specialized not in HF cache. Downloading to %s …",
                _CLIP_BERT_EXPECTED,
            )
            from huggingface_hub import snapshot_download

            snapshot_download(
                "microsoft/BiomedVLP-CXR-BERT-specialized",
                local_dir=str(_CLIP_BERT_EXPECTED),
            )

    def _run_clip(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run evaluate_clip.py and return a dict with ``CLIPScore``, ``CLIPScore_I2I``, and ``CLIPScore_mean``.

        Appends ``--prompt_xlsx`` to the command when ``self.prompt_xlsx`` is set and
        the file exists; otherwise the upstream script uses its own default path.
        Returns NaN for all three keys if the subprocess exits non-zero or the output
        JSON is not produced.
        """
        out_json = out_dir / "clip.json"
        cmd = [
            sys.executable,
            str(_CLIP_SCRIPT),
            "--generated_dir",
            str(pred_dir),
            "--gt_root",
            str(self.gt_dir),
            "--out_json",
            str(out_json),
        ]
        if self.prompt_xlsx is not None and self.prompt_xlsx.is_file():
            cmd += ["--prompt_xlsx", str(self.prompt_xlsx)]
        else:
            log.info(
                "No prompt_xlsx — CLIPScore I2T will use default xlsx path in script."
            )

        # ct_clip + transformer_maskgit must be importable from the submodule
        env = self._clip_env()
        log.info("Running CLIP eval: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=_EVAL_DIR, env=env, check=False)
        if result.returncode != 0 or not out_json.is_file():
            log.error("CLIPScore evaluation failed (rc=%d).", result.returncode)
            return {
                "CLIPScore": float("nan"),
                "CLIPScore_I2I": float("nan"),
                "CLIPScore_mean": float("nan"),
            }
        with open(out_json) as f:
            return json.load(f)

    @staticmethod
    def _clip_env() -> dict:
        """Env for CLIPScore subprocess.

        evaluate_clip.py imports top-level `transformer_maskgit` and `ct_clip`.
        Upstream `pip install -e`s them; locally that editable mapping breaks when
        the submodule moves (the a945900 reorg). We add each package's *parent*
        dir (not the package dir itself — that would put sub-packages at top level
        and break `import ct_clip.mlm`) to PYTHONPATH so the imports resolve
        regardless of any stale editable install.
        """
        import os

        env = os.environ.copy()
        parents = [str(p) for p in ctclip_pkg_parents()]
        if parents:
            env["PYTHONPATH"] = os.pathsep.join(
                [*parents, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        return env

    # ------------------------------------------------------------------ #
    #  FID-2.5D                                                            #
    # ------------------------------------------------------------------ #

    def _run_fid(self, pred_dir: Path, out_dir: Path) -> dict[str, float]:
        """Run compute_fid_2-5d_ct.py via fire.Fire kwargs interface.

        The script does NOT support --out_json; results are in logger.info output.
        We capture stderr+stdout and parse FID XY/YZ/ZX/Avg lines.
        """
        _nan = {
            "FID_2p5D_Avg": float("nan"),
            "FID_2p5D_XY": float("nan"),
            "FID_2p5D_XZ": float("nan"),
            "FID_2p5D_YZ": float("nan"),
        }

        # Build file lists (relative filenames, one per line)
        real_filelist = out_dir / "gt_filelist.txt"
        synth_filelist = out_dir / "pred_filelist.txt"

        gt_files = sorted(self.gt_dir.glob("*.mha"))
        pred_files = sorted(pred_dir.glob("*.mha"))
        if not gt_files or not pred_files:
            log.error(
                "FID-2.5D: empty gt_dir (%d) or pred_dir (%d).",
                len(gt_files),
                len(pred_files),
            )
            return _nan
        if len(gt_files) < 10 or len(pred_files) < 10:
            log.warning(
                "FID-2.5D may be unreliable with only %d GT / %d pred files (need ≥50).",
                len(gt_files),
                len(pred_files),
            )

        real_filelist.write_text("\n".join(f.name for f in gt_files))
        synth_filelist.write_text("\n".join(f.name for f in pred_files))

        features_dir = out_dir / "fid_features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Reuse GT features across models: hardlink any cached features for this GT set into
        # this run's gt dir so the upstream loop skips their (GPU) forward pass.
        gt_feat_dir = features_dir / "gt"
        gt_feat_dir.mkdir(parents=True, exist_ok=True)
        shared_gt = _shared_gt_feat_dir(self.gt_dir)
        linked = _link_shared_gt_features(shared_gt, gt_feat_dir)
        if linked:
            log.info(
                "FID GT cache: linked %d shared GT features from %s (forward skipped)",
                linked,
                shared_gt,
            )

        n = min(len(gt_files), len(pred_files))
        # Use our wrapper that applies the np.float_ → np.float64 shim before
        # importing MONAI's compute_fid_2-5d_ct.py (numpy 2.x compatibility)
        fid_runner = Path("/workspace/src/eval/tasks/_fid_runner.py")
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            str(fid_runner),
            f"--real_dataset_root={self.gt_dir}",
            f"--real_filelist={real_filelist}",
            "--real_features_dir=gt",
            f"--synth_dataset_root={pred_dir}",
            f"--synth_filelist={synth_filelist}",
            "--synth_features_dir=pred",
            f"--model_name={_FID_MODEL}",
            f"--num_images={n}",
            f"--output_root={features_dir}",
            f"--target_shape={_FID_TARGET_SHAPE}",
            f"--enable_padding={_FID_ENABLE_PADDING}",
            f"--enable_center_cropping={_FID_ENABLE_CENTER_CROP}",
            f"--enable_resampling_spacing={_FID_RESAMPLE}",
        ]
        log.info("Running FID-2.5D: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=_EVAL_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = result.stdout + "\n" + result.stderr

        # The subprocess is used purely as a GPU feature *extractor*: it writes per-volume
        # features to features_dir/{gt,pred} incrementally (torch.save), then attempts the final
        # FID statistic on the GPU. At full scale that aggregation — ~512k slice-features per
        # plane stacked across 6 matrices, plus an all_gather copy — OOMs the GPU. We therefore
        # always compute the FID ourselves on CPU from those cached features: deterministic,
        # memory-safe (plane-by-plane), ~1-2 min for 1000. This avoids the GPU OOM by design
        # rather than depending on (and parsing) the subprocess's doomed GPU FID. Feature
        # extraction is the only step that needs the GPU.
        fid_data = _fid_from_cached_features(
            features_dir, len(gt_files), len(pred_files)
        )

        if not fid_data:
            # Incomplete features -> the subprocess died *before* finishing extraction (a real
            # failure, distinct from the expected post-extraction GPU-FID OOM).
            log.error(
                "FID-2.5D: feature extraction did not complete; cannot compute FID.\n%s",
                combined[-2000:],
            )
            return _nan

        # Persist this run's GT features into the shared cache so the next model reuses them.
        _populate_shared_gt_features(gt_feat_dir, shared_gt)

        out_json = out_dir / "fid.json"
        with open(out_json, "w") as f:
            json.dump(fid_data, f, indent=2)
        return fid_data

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
        """Run a subprocess command and return its exit code.

        Args:
            cmd: command and arguments list passed to ``subprocess.run``.
            cwd: working directory for the subprocess.
            env: environment mapping; inherits the current process environment when None.

        Returns:
            The subprocess return code (0 on success).
        """
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        if result.returncode != 0:
            log.error("Command failed (rc=%d)", result.returncode)
        return result.returncode


__all__ = ["CTGenEvaluator"]
