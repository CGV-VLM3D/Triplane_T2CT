"""clip_persample.py: per-sample runner's aggregate == CTGenEvaluator._run_clip's aggregate.

Real end-to-end equivalence check (not mocked) against 2 real prediction files — skips if the
CT-CLIP checkpoint isn't present (same skip convention as test_maisi_frozen_load.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.analysis import clip_persample
from src.eval.ct_rate_cases import load_eval_cases, write_prompt_xlsx
from src.eval.tasks.ctgen import CTGenEvaluator

_CTCLIP_CKPT = Path("/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt")
_PRED_DIR = Path(
    "/workspace/outputs/report2ct_wan_mask/eval_ep299_sp0.75_1.3_cfg5/predictions"
)
_GT_DIR = "/workspace/data/vlm3d_eval/_valid_full_3001"
_SCAN_IDS = ["valid_1004_a_2", "valid_1005_a_1"]


@pytest.mark.skipif(not _CTCLIP_CKPT.is_file(), reason="CT-CLIP_v2.pt not present")
@pytest.mark.skipif(
    not all((_PRED_DIR / f"{s}.mha").is_file() for s in _SCAN_IDS),
    reason="fixture prediction files not present",
)
def test_clip_persample_equivalence(tmp_path: Path):
    subset_dir = tmp_path / "pred_subset"
    subset_dir.mkdir()
    for sid in _SCAN_IDS:
        (subset_dir / f"{sid}.mha").symlink_to((_PRED_DIR / f"{sid}.mha").resolve())

    cases = [c for c in load_eval_cases(n_samples=None) if c.scan_id in _SCAN_IDS]
    prompt_xlsx = tmp_path / "prompts.xlsx"
    write_prompt_xlsx(cases, prompt_xlsx)

    # A) existing (subprocess) path
    existing_out = tmp_path / "existing_out"
    existing_out.mkdir()
    evaluator = CTGenEvaluator(
        gt_dir=_GT_DIR, ctclip_ckpt=str(_CTCLIP_CKPT), prompt_xlsx=prompt_xlsx
    )
    existing_agg = evaluator._run_clip(subset_dir, existing_out)

    # B) our per-sample runner's aggregate (same formula, kept per-sample too)
    _, our_agg = clip_persample.run(
        pred_dir=subset_dir,
        gt_dir=_GT_DIR,
        prompt_xlsx=prompt_xlsx,
        out_dir=tmp_path / "our_out",
        ctclip_ckpt=str(_CTCLIP_CKPT),
    )

    assert abs(existing_agg["CLIPScore"] - our_agg["CLIPScore"]) < 1e-3
    assert abs(existing_agg["CLIPScore_I2I"] - our_agg["CLIPScore_I2I"]) < 1e-3
    assert abs(existing_agg["CLIPScore_mean"] - our_agg["CLIPScore_mean"]) < 1e-3


def test_clip_persample_prompt_xlsx_none_skips_gracefully(tmp_path: Path):
    """gt_mode="official" with no user-supplied prompt_xlsx must skip cleanly (NaN), not crash
    with Path(None) (code review + independent-verifier finding, 2026-07-27)."""
    per_sample, agg = clip_persample.run(
        pred_dir=tmp_path, gt_dir=tmp_path, prompt_xlsx=None, out_dir=tmp_path / "out"
    )
    assert per_sample == {}
    assert agg["CLIPScore"] != agg["CLIPScore"]  # NaN != NaN


@pytest.mark.skipif(not _CTCLIP_CKPT.is_file(), reason="CT-CLIP_v2.pt not present")
def test_clip_persample_rejects_non_benign_checkpoint_drift(monkeypatch):
    """strict=False must not silently mask a genuinely wrong/renamed checkpoint — only the
    known-benign position_ids drift is tolerated (code review finding, 2026-07-27)."""
    import torch

    real_load_state_dict = torch.nn.Module.load_state_dict

    def _inject_bogus_unexpected_key(self, state_dict, *args, **kwargs):
        state_dict = dict(state_dict)
        state_dict["totally_bogus_renamed_layer.weight"] = torch.zeros(1)
        return real_load_state_dict(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.Module, "load_state_dict", _inject_bogus_unexpected_key
    )

    with pytest.raises(RuntimeError, match="unexpected keys beyond"):
        clip_persample.run(
            pred_dir=_PRED_DIR,
            gt_dir=_GT_DIR,
            prompt_xlsx=Path(
                "/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt"
            ),  # unused before the raise
            out_dir=Path("/tmp/unused_clip_persample_test_out"),
            ctclip_ckpt=str(_CTCLIP_CKPT),
        )
