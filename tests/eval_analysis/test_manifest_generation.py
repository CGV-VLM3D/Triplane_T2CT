"""Generation-side manifest consumption: case loading, output naming, mask routing, noise
seeding, the CLIP GT view, and the FID/FVD refusal.

CPU-only — no checkpoint is loaded (the Wan samplers build their models lazily in ``_init``), so
these exercise the ROUTING that decides which file each row reads and writes, not the diffusion
itself (that is the GPU smoke, unit 2b).
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.eval.ct_rate_cases import (
    load_eval_cases,
    load_manifest_cases,
    write_prompt_xlsx,
)
from src.eval.manifest import (
    build_gt_view,
    check_generation_provenance,
    read_manifest_rows,
)
from src.eval.samplers.base import EvalCase
from src.eval.samplers.report2ct_wan import (
    Report2CTWanEvalSampler,
    Report2CTWanMaskLatentSampler,
    _noise_seed,
)
from tests.eval_analysis._helpers import write_mha

VALID_V2_IDS = Path("/workspace/data/ctrate_toy_v2/valid_v2/ids.json")


def _row(target: str, condition: str, source: str | None, seed: int = 0) -> dict:
    return {
        "sample_id": f"{target}__{condition}__src-{source or 'none'}__sm-1.0__seed-{seed}",
        "target_id": target,
        "condition": condition,
        "cond_mask_source_id": source,
        "seed": seed,
        "cfg_scale_text": 5.0,
        "cfg_scale_mask": 1.0,
        "run_id": "test",
        "ckpt": "/tmp/fake.ckpt",
    }


@pytest.fixture(scope="module")
def targets() -> list[str]:
    return json.loads(VALID_V2_IDS.read_text())["ids"][:2]


@pytest.fixture
def manifest(tmp_path: Path, targets) -> tuple[Path, list[dict]]:
    """4 rows: gt + swap + null for target[0], gt for target[1]."""
    rows = [
        _row(targets[0], "gt", targets[0]),
        _row(targets[0], "label_mismatched_swap", targets[1]),
        _row(targets[0], "null", None),
        _row(targets[1], "gt", targets[1]),
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def test_cases_are_one_per_row_with_target_text(manifest, targets):
    """One case per ROW (not per scan); text conditioning always comes from the TARGET."""
    path, rows = manifest
    cases = load_manifest_cases(path)

    assert len(cases) == len(rows)
    assert [c.sample_id for c in cases] == [r["sample_id"] for r in rows]
    assert [c.scan_id for c in cases] == [r["target_id"] for r in rows]
    assert [c.cond_mask_source_id for c in cases] == [
        r["cond_mask_source_id"] for r in rows
    ]
    assert [c.out_stem for c in cases] == [r["sample_id"] for r in rows]

    plain = {c.scan_id: c for c in load_eval_cases(n_samples=None)}
    for case in cases:
        assert case.findings == plain[case.scan_id].findings
        assert case.impression == plain[case.scan_id].impression
    # the swap row's text is the TARGET's, not the mask donor's
    swap = next(c for c in cases if c.condition == "label_mismatched_swap")
    assert swap.findings == plain[targets[0]].findings
    assert swap.cond_mask_source_id == targets[1]


def test_plain_case_out_stem_is_scan_id():
    """A plain run keeps "1 scan = 1 prediction named after the scan"."""
    case = EvalCase(scan_id="valid_1_a_1", findings="f", impression="i")
    assert case.out_stem == "valid_1_a_1"
    assert case.sample_id is None and case.seed is None


def test_prompt_xlsx_is_keyed_by_sample_id(manifest, tmp_path):
    """Upstream CLIP looks the prompt up by the PREDICTION's stem, so Names must be sample_ids
    while the text stays the target's report (identical for every condition of one target)."""
    path, rows = manifest
    cases = load_manifest_cases(path)
    xlsx = write_prompt_xlsx(cases, tmp_path / "prompts.xlsx")

    df = pd.read_excel(xlsx, engine="openpyxl")
    assert list(df["Names"]) == [f"{r['sample_id']}.mha" for r in rows]

    text_by_name = dict(zip(df["Names"], df["Text_prompts"]))
    target0_rows = [r for r in rows if r["target_id"] == rows[0]["target_id"]]
    texts = {text_by_name[f"{r['sample_id']}.mha"] for r in target0_rows}
    assert len(texts) == 1, (
        "all conditions of one target must share the target's prompt"
    )


def _mask_sampler(tmp_path: Path) -> Report2CTWanMaskLatentSampler:
    """Sampler instance without any weights (Wan samplers load lazily in ``_init``)."""
    sampler = Report2CTWanMaskLatentSampler(
        ckpt_path="/nonexistent.ckpt",
        mask_dir=str(tmp_path),
        n_steps=1,
        cfg_scale=1.0,
        spacing_mm=[0.75, 0.75, 1.3],
    )
    sampler._device = torch.device("cpu")
    return sampler


def _write_mask_latent(mask_dir: Path, scan_id: str) -> Path:
    """Tiny stand-in for a Wan mask latent: (H, W, D, C) NIfTI, as the precompute script writes."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    path = mask_dir / f"{scan_id}_mask_emb.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4, 16), dtype=np.float32), affine=np.eye(4)),
        str(path),
    )
    return path


def test_mask_comes_from_cond_mask_source_id(tmp_path, targets):
    """The conditioning mask is the ROW's source scan — the donor for a swap, the target's own
    for gt/plain — and its absence is a loud error, never a silent skip."""
    sampler = _mask_sampler(tmp_path)
    _write_mask_latent(tmp_path, targets[1])

    swap_case = EvalCase(
        scan_id=targets[0],
        findings="f",
        impression="i",
        sample_id="s1",
        condition="label_mismatched_swap",
        cond_mask_source_id=targets[1],
        seed=0,
    )
    assert sampler._mask_source_id(swap_case) == targets[1]
    latent = sampler._case_mask_latent(swap_case)
    assert latent.shape == (1, 16, 4, 4, 4)  # (B, C, H, W, D)

    plain_case = EvalCase(scan_id=targets[0], findings="f", impression="i")
    assert sampler._mask_source_id(plain_case) == targets[0]  # own mask, no manifest
    with pytest.raises(FileNotFoundError, match="mask latent missing"):
        sampler._case_mask_latent(plain_case)


def test_condition_null_needs_a_learned_null_mask(tmp_path, targets):
    """condition=null designates no mask; a model without ``no_mask_embed`` must refuse."""
    sampler = _mask_sampler(tmp_path)
    null_case = EvalCase(
        scan_id=targets[0],
        findings="f",
        impression="i",
        sample_id="s_null",
        condition="null",
        cond_mask_source_id=None,
        seed=0,
    )
    assert sampler._mask_source_id(null_case) is None
    with pytest.raises(RuntimeError, match="no_mask_embed"):
        sampler._case_mask_latent(null_case)

    sampler._null_mask = torch.zeros(16)  # what a mask_cfg=true ckpt provides
    assert sampler._case_mask_latent(null_case) is None


def test_noise_seed_is_shared_across_conditions_of_one_target(targets):
    """Every condition of a target starts from the SAME noise, so the intervention is the only
    difference between its volumes; a different target draws different noise."""

    def case(target: str, condition: str, seed: int = 0) -> EvalCase:
        return EvalCase(
            scan_id=target,
            findings="",
            impression="",
            sample_id=f"{target}__{condition}",
            condition=condition,
            seed=seed,
        )

    gt, swap = case(targets[0], "gt"), case(targets[0], "label_mismatched_swap")
    assert _noise_seed(gt) == _noise_seed(swap)
    assert _noise_seed(gt) != _noise_seed(case(targets[1], "gt"))
    assert _noise_seed(gt) != _noise_seed(case(targets[0], "gt", seed=1))

    torch.manual_seed(_noise_seed(gt))
    z_gt = torch.randn(1, 16, 4, 4, 4)  # (B, C, H, W, D)
    torch.manual_seed(_noise_seed(swap))
    z_swap = torch.randn(1, 16, 4, 4, 4)
    assert torch.equal(z_gt, z_swap)


def test_eval_sampler_finds_predictions_by_sample_id(manifest, tmp_path):
    """The pass-through sampler that run_eval instantiates looks for <sample_id>.mha."""
    path, rows = manifest
    cases = load_manifest_cases(path)
    pred_dir = tmp_path / "predictions"
    for row in rows:
        write_mha(pred_dir / f"{row['sample_id']}.mha", np.zeros((2, 2, 2), np.int16))

    found = Report2CTWanEvalSampler().generate(cases, pred_dir, torch.device("cpu"))
    assert [p.stem for p in found] == [r["sample_id"] for r in rows]

    (pred_dir / f"{rows[0]['sample_id']}.mha").unlink()
    with pytest.raises(FileNotFoundError):
        Report2CTWanEvalSampler().generate(cases, pred_dir, torch.device("cpu"))


def test_gt_view_links_sample_id_to_its_target(manifest, tmp_path):
    """CLIP matches GT by the prediction's stem, so the view exposes each target's GT under the
    row's sample_id — and a missing target GT is an error, not a silently unscored sample."""
    path, rows = manifest
    cases = load_manifest_cases(path)
    gt_dir = tmp_path / "gt"
    for target in {r["target_id"] for r in rows}:
        write_mha(gt_dir / f"{target}.mha", np.zeros((2, 2, 2), np.int16))

    view = build_gt_view(cases, gt_dir, tmp_path / "gt_view")
    for case in cases:
        link = view / f"{case.out_stem}.mha"
        assert link.resolve() == (gt_dir / f"{case.scan_id}.mha").resolve()
    assert build_gt_view(cases, gt_dir, view) == view  # idempotent

    (gt_dir / f"{rows[0]['target_id']}.mha").unlink()
    with pytest.raises(FileNotFoundError):
        build_gt_view(cases, gt_dir, tmp_path / "gt_view2")


def test_read_manifest_rows_raises_where_the_scoring_reader_skips(tmp_path, targets):
    """Generation must fail loudly: a row it cannot read is a volume that never gets generated."""
    good = json.dumps(_row(targets[0], "gt", targets[0]))

    (tmp_path / "bad_json.jsonl").write_text(good + "\n{not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_manifest_rows(tmp_path / "bad_json.jsonl")

    (tmp_path / "dup.jsonl").write_text(good + "\n" + good + "\n")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        read_manifest_rows(tmp_path / "dup.jsonl")

    unsafe = _row(targets[0], "gt", targets[0])
    unsafe["cond_mask_source_id"] = "../../etc/passwd"
    (tmp_path / "unsafe.jsonl").write_text(json.dumps(unsafe) + "\n")
    with pytest.raises(ValueError, match="unsafe id"):
        read_manifest_rows(tmp_path / "unsafe.jsonl")

    (tmp_path / "empty.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="no rows"):
        read_manifest_rows(tmp_path / "empty.jsonl")


def test_generation_provenance_must_match_the_manifest(manifest):
    """sample_id encodes s_m and the rows record the ckpt, so generating them with other
    settings would produce files whose names claim settings they were not generated with."""
    _, rows = manifest
    check_generation_provenance(rows, "/tmp/fake.ckpt", 5.0, 1.0)
    with pytest.raises(ValueError, match="cfg_scale_mask"):
        check_generation_provenance(rows, "/tmp/fake.ckpt", 5.0, 0.0)
    with pytest.raises(ValueError, match="ckpt"):
        check_generation_provenance(rows, "/tmp/other.ckpt", 5.0, 1.0)


def test_v2_dual_cfg_input_is_unchanged_for_a_designated_mask(tmp_path):
    """The dual-CFG batch a mask-carrying row builds must be exactly what it was before manifest
    support existed (the old expression is written out below), and a null row must instead put
    the learned null on BOTH branches so the s_m term is exactly zero."""
    from src.eval.samplers.report2ct_wan import Report2CTWanMaskV2LatentSampler

    sampler = Report2CTWanMaskV2LatentSampler(
        ckpt_path="/nonexistent.ckpt",
        mask_dir=str(tmp_path),
        cfg_scale_text=5.0,
        cfg_scale_mask=1.0,
        n_steps=1,
        spacing_mm=[0.75, 0.75, 1.3],
    )
    sampler._device = torch.device("cpu")
    sampler._null_mask = torch.arange(16, dtype=torch.float32)  # (16,)
    z = torch.randn(1, 16, 4, 4, 4)  # (B, C, H, W, D)
    mask = torch.randn(1, 16, 4, 4, 4)  # (B, C, H, W, D)

    seen: list[torch.Tensor] = []

    def _fake_unet(*, x, timesteps, context, class_labels, spacing_tensor):
        seen.append(x)  # (3, 32, 4, 4, 4)
        return torch.zeros(3, 16, 4, 4, 4)

    sampler._unet = _fake_unet
    args = (
        torch.tensor([1.0]),
        torch.zeros(1, 2, 2560),
        torch.tensor([1]),
        torch.ones(1, 3),
    )

    sampler._mask_latent = mask
    sampler._predict(z, *args)
    old_null_m = sampler._null_mask.to(mask.dtype).view(
        1, -1, 1, 1, 1
    )  # pre-manifest code
    old_batch = torch.cat(
        [
            torch.cat([z, old_null_m.expand_as(mask)], dim=1),
            torch.cat([z, old_null_m.expand_as(mask)], dim=1),
            torch.cat([z, mask], dim=1),
        ],
        dim=0,
    )  # (3, 32, 4, 4, 4)
    assert torch.equal(seen[-1], old_batch)

    sampler._mask_latent = None  # condition=null
    sampler._predict(z, *args)
    null_x, _, real_x = seen[-1].chunk(3, dim=0)
    assert torch.equal(null_x, real_x), "null row must give e(m,t) == e(∅m,t)"


def test_setlevel_metrics_are_refused_on_a_manifest_run():
    """FID/FVD assume 1 volume = 1 independent sample, which a manifest run breaks."""
    from scripts.run_eval import _refuse_setlevel_metrics

    def cfg(manifest, **metrics):
        base = {
            "fid_2p5d": False,
            "fvd": False,
            "fvd_ctclip": False,
            "per_sample": True,
        }
        return OmegaConf.create(
            {"task": {"manifest": manifest, "metrics": {**base, **metrics}}}
        )

    _refuse_setlevel_metrics(
        cfg(None, fid_2p5d=True, fvd=True)
    )  # plain run: unaffected
    _refuse_setlevel_metrics(cfg("/tmp/m.jsonl"))  # manifest + diagnostics only: fine
    for metric in ("fid_2p5d", "fvd", "fvd_ctclip"):
        with pytest.raises(RuntimeError, match=metric):
            _refuse_setlevel_metrics(cfg("/tmp/m.jsonl", **{metric: True}))
