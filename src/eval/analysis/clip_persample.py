"""Per-sample CLIP-T2I / CLIP-I2I — the one VLM3D metric that already computes a per-sample
value internally (upstream ``evaluate_clip.py``'s ``s_txt``/``s_i2i`` lists) before discarding
it via ``np.mean()``. This module runs the SAME model + SAME formula in-process and keeps the
per-sample scores, producing an aggregate JSON with an identical formula to
``CTGenEvaluator._run_clip`` (verified by an equivalence test) so it is a drop-in replacement.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NAN_AGG = {
    "CLIPScore": float("nan"),
    "CLIPScore_I2I": float("nan"),
    "CLIPScore_mean": float("nan"),
}


def _import_upstream_evaluate_clip():
    """Import the upstream ``evaluate_clip.py`` script as a module (safe: its ``main()`` only
    runs under ``if __name__ == "__main__"``, which is false when imported this way)."""
    from src.eval._vlm3d_paths import ctgen_eval_dir

    path = ctgen_eval_dir() / "evaluate_clip.py"
    spec = importlib.util.spec_from_file_location("_evaluate_clip_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    pred_dir: str | Path,
    gt_dir: str | Path,
    prompt_xlsx: str | Path,
    out_dir: str | Path,
    ctclip_ckpt: str | Path | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Run per-sample CLIP-T2I/I2I and write ``clip_persample.csv`` + ``clip.json``.

    Args:
        pred_dir: generated ``.mha`` directory.
        gt_dir: GT ``.mha`` directory.
        prompt_xlsx: (Names, Text_prompts) xlsx for CLIP-T2I text prompts.
        out_dir: where to write ``clip_persample.csv`` (into) and ``clip.json`` (top-level, so
            it is byte-for-byte the same artifact ``CTGenEvaluator._run_clip`` would write).
        ctclip_ckpt: path to ``CT-CLIP_v2.pt`` (default: ``CTGenEvaluator``'s default path).

    Returns:
        ``({sample_id: (clip_t2i, clip_i2i)}, {"CLIPScore":..., "CLIPScore_I2I":...,
        "CLIPScore_mean":...})`` — the aggregate dict is NaN-filled if prerequisites are missing
        (same skip policy as ``CTGenEvaluator._maybe_run_clip``).
    """
    from src.eval._vlm3d_paths import ctclip_pkg_parents
    from src.eval.tasks.ctgen import CTGenEvaluator

    out_dir = Path(out_dir)
    evaluator = CTGenEvaluator(gt_dir=gt_dir, ctclip_ckpt=ctclip_ckpt)

    if prompt_xlsx is None:
        # VolTextDS requires a real xlsx path (no None handling, unlike the upstream CLI's
        # skip-I2T-if-absent behavior) — skip cleanly rather than let Path(None) crash
        # (code review finding, 2026-07-27: hit under gt_mode="official" with no user-supplied
        # prompt_xlsx, since the auto-write in run_eval.py only fires for gt_mode="valid").
        log.warning("No prompt_xlsx given — skipping per-sample CLIP.")
        return {}, dict(_NAN_AGG)
    if not evaluator.ctclip_ckpt.is_file():
        log.warning(
            "CT-CLIP_v2.pt not found at %s — skipping per-sample CLIP.",
            evaluator.ctclip_ckpt,
        )
        return {}, dict(_NAN_AGG)
    if not evaluator._setup_clip_paths():
        log.warning("Could not set up /opt/app/models/ — skipping per-sample CLIP.")
        return {}, dict(_NAN_AGG)

    for parent in ctclip_pkg_parents():
        sp = str(parent)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    import torch
    from torch.utils.data import DataLoader
    from transformer_maskgit import CTViT
    from ct_clip import CTCLIP
    from transformers import BertTokenizer, BertModel

    upstream = _import_upstream_evaluate_clip()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DUPLICATION INTENTIONAL: mirrors evaluate_clip.py main():159-171 (model construction).
    # That upstream script only builds the model inside a __main__-guarded main(), so it can't
    # be called as a function; VolTextDS/_clip_score/_case_stem ARE reused via `upstream.*`.
    tok = BertTokenizer.from_pretrained(
        "/opt/app/models/BiomedVLP-CXR-BERT-specialized"
    )
    txt = BertModel.from_pretrained(
        "/opt/app/models/BiomedVLP-CXR-BERT-specialized"
    ).to(dev)
    img = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    ).to(dev)
    clip = CTCLIP(
        image_encoder=img,
        text_encoder=txt,
        dim_image=294_912,
        dim_text=768,
        dim_latent=512,
    ).to(dev)
    # Don't call upstream CTCLIP.load() — it does strict=True, which fails on benign
    # HF-transformers drift (CT-CLIP_v2.pt was saved with an older transformers that persisted
    # `text_transformer.embeddings.position_ids` as a buffer; current versions don't). Same
    # fix as src/baselines/ctclip_adapter.py (verified 2026-07-27: transformers==4.46.3 drops
    # this buffer from BertModel's own state_dict, so strict loading always raises here).
    state_dict = torch.load(
        str(evaluator.ctclip_ckpt), map_location=dev, weights_only=False
    )
    incompatible = clip.load_state_dict(state_dict, strict=False)
    # strict=False must not silently mask a genuinely wrong/renamed checkpoint (which would
    # load with random-init weights and produce plausible-but-wrong CLIP scores that then
    # overwrite the headline metrics.json via the dedup routing in run_eval.py) — assert the
    # drift is exactly the known-benign one (code review finding, 2026-07-27).
    if incompatible.missing_keys:
        raise RuntimeError(
            f"CT-CLIP checkpoint {evaluator.ctclip_ckpt} is missing keys the model expects "
            f"(would load with random-init weights): {incompatible.missing_keys}"
        )
    unexpected_non_benign = [
        k for k in incompatible.unexpected_keys if not k.endswith("position_ids")
    ]
    if unexpected_non_benign:
        raise RuntimeError(
            f"CT-CLIP checkpoint {evaluator.ctclip_ckpt} has unexpected keys beyond the known "
            f"benign position_ids drift: {unexpected_non_benign}"
        )
    clip.eval()

    ds = upstream.VolTextDS(Path(pred_dir), Path(gt_dir), Path(prompt_xlsx))
    ld = DataLoader(ds, batch_size=1, num_workers=0)

    s_txt: list[float] = []
    s_i2i: list[float] = []
    for gv, prompt, rv in ld:
        gv, rv = gv.to(dev), rv.to(dev)
        gv_r = gv / 1000.0
        rv_r = rv / 1000.0
        with torch.no_grad():
            toks = tok(
                prompt[0],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(dev)
            txt_lat, img_lat_g, _ = clip(
                toks, gv_r.unsqueeze(0), return_latents=True, device=dev
            )
            _, img_lat_r, _ = clip(
                toks, rv_r.unsqueeze(0), return_latents=True, device=dev
            )
        s_txt.append(upstream._clip_score(img_lat_g[0], txt_lat[0]))
        s_i2i.append(upstream._clip_score(img_lat_g[0], img_lat_r[0]))

    stems = [upstream._case_stem(p) for p in ds.gen_files]
    per_sample = {stem: (t, i) for stem, t, i in zip(stems, s_txt, s_i2i)}

    agg = {
        "CLIPScore": float(np.mean(s_txt)),
        "CLIPScore_I2I": float(np.mean(s_i2i)),
        "CLIPScore_mean": (float(np.mean(s_txt)) + float(np.mean(s_i2i))) / 2,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": stems, "clip_t2i": s_txt, "clip_i2i": s_i2i}).to_csv(
        out_dir / "clip_persample.csv", index=False
    )
    (out_dir / "clip.json").write_text(json.dumps(agg, indent=2))
    log.info(
        "Per-sample CLIP: %d samples -> %s", len(stems), out_dir / "clip_persample.csv"
    )

    return per_sample, agg


__all__ = ["run"]
