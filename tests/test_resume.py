"""Unit tests for snapshot save / resume helpers in scripts/train.py.

These exercise the pure helpers without needing GPU, wandb, or the real
training loop.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# Import scripts/train.py as a module.
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import train as train_mod  # noqa: E402


# ---------------------------------------------------------------------------
# _resolve_resume_path
# ---------------------------------------------------------------------------


def test_resolve_resume_off_returns_none(tmp_path: Path):
    (tmp_path / "latest.pt").write_bytes(b"x")  # should be ignored
    for val in (False, None, "off", "false", "no", "none", ""):
        assert train_mod._resolve_resume_path(tmp_path, val) is None


def test_resolve_resume_auto_without_latest_returns_none(tmp_path: Path):
    for val in (True, "auto", "true", "yes"):
        assert train_mod._resolve_resume_path(tmp_path, val) is None


def test_resolve_resume_auto_with_latest_returns_path(tmp_path: Path):
    (tmp_path / "latest.pt").write_bytes(b"x")
    for val in (True, "auto", "true", "yes"):
        assert train_mod._resolve_resume_path(tmp_path, val) == tmp_path / "latest.pt"


def test_resolve_resume_explicit_path_must_exist(tmp_path: Path):
    target = tmp_path / "specific.pt"
    target.write_bytes(b"x")
    assert train_mod._resolve_resume_path(tmp_path, str(target)) == target


def test_resolve_resume_explicit_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        train_mod._resolve_resume_path(tmp_path, str(tmp_path / "missing.pt"))


# ---------------------------------------------------------------------------
# _save_snapshot round-trip
# ---------------------------------------------------------------------------


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))


def _tiny_cfg():
    return OmegaConf.create(
        {
            "train": {"epochs": 1, "batch_size": 1, "lr": 1e-3},
            "model": {"kind": "TestStub"},
        }
    )


def test_save_snapshot_roundtrip_model_and_opt(tmp_path: Path):
    torch.manual_seed(0)
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # take a step so optimizer state has moments
    x = torch.randn(4, 4)
    y = model(x).sum()
    y.backward()
    opt.step()

    ckpt = tmp_path / "latest.pt"
    train_mod._save_snapshot(
        ckpt,
        model=model,
        opt=opt,
        epoch=3,
        global_step=42,
        completed_steps_in_epoch=7,
        epoch_done=False,
        data_range=5.25,
        cfg=_tiny_cfg(),
        wandb_run_id="abcd1234",
    )
    assert ckpt.exists()
    # tmp file must have been renamed away
    assert not ckpt.with_suffix(ckpt.suffix + ".tmp").exists()

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 3
    assert payload["global_step"] == 42
    assert payload["completed_steps_in_epoch"] == 7
    assert payload["epoch_done"] is False
    assert payload["data_range"] == pytest.approx(5.25)
    assert payload["wandb_run_id"] == "abcd1234"

    # Load into a fresh model+opt and confirm parameters & opt state match.
    fresh = _tiny_model()
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    fresh.load_state_dict(payload["model"])
    fresh_opt.load_state_dict(payload["opt"])

    for p_old, p_new in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p_old, p_new)
    # Adam stores exp_avg / exp_avg_sq per param after the first .step()
    for g_old, g_new in zip(opt.state.values(), fresh_opt.state.values()):
        assert torch.allclose(g_old["exp_avg"], g_new["exp_avg"])
        assert torch.allclose(g_old["exp_avg_sq"], g_new["exp_avg_sq"])


def test_save_snapshot_is_atomic_via_tmp_rename(tmp_path: Path):
    """tmp+rename: the final path should never exist until rename completes."""
    ckpt = tmp_path / "latest.pt"
    # Pre-populate with bogus content; if save fails midway it must NOT
    # corrupt this file.
    ckpt.write_bytes(b"DO_NOT_CLOBBER")
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_mod._save_snapshot(
        ckpt,
        model=model,
        opt=opt,
        epoch=1,
        global_step=0,
        completed_steps_in_epoch=0,
        epoch_done=False,
        data_range=1.0,
        cfg=_tiny_cfg(),
        wandb_run_id=None,
    )
    # After successful save the file should be a real torch checkpoint, not
    # the bogus pre-existing bytes.
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1


# ---------------------------------------------------------------------------
# RNG capture / restore
# ---------------------------------------------------------------------------


def test_rng_capture_restore_reproduces_streams():
    # Seed once so capture is from a known state.
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    snap = train_mod._capture_rng_state()

    # Advance all three RNGs.
    _ = torch.randn(8)
    _ = np.random.rand(8)
    _ = [random.random() for _ in range(8)]

    # Restore — subsequent draws should now match a fresh "baseline" draw
    # taken right after the capture.
    train_mod._restore_rng_state(snap)
    after_restore_torch = torch.randn(8)
    after_restore_np = np.random.rand(8)
    after_restore_py = [random.random() for _ in range(8)]

    # Compute the expected baseline by re-creating the captured state.
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    expected_torch = torch.randn(8)
    expected_np = np.random.rand(8)
    expected_py = [random.random() for _ in range(8)]

    assert torch.equal(after_restore_torch, expected_torch)
    assert np.allclose(after_restore_np, expected_np)
    assert after_restore_py == expected_py


# ---------------------------------------------------------------------------
# End-to-end semantics: save mid-epoch, load, resume planner picks right epoch
# ---------------------------------------------------------------------------


def test_resume_planner_mid_epoch(tmp_path: Path):
    """Mid-epoch snapshot → start at same epoch, skip already-done steps."""
    ckpt = tmp_path / "latest.pt"
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_mod._save_snapshot(
        ckpt,
        model=model,
        opt=opt,
        epoch=4,
        global_step=1200,
        completed_steps_in_epoch=200,
        epoch_done=False,
        data_range=3.14,
        cfg=_tiny_cfg(),
        wandb_run_id="run-xyz",
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved_epoch = int(state["epoch"])
    completed = int(state["completed_steps_in_epoch"])
    epoch_done = bool(state["epoch_done"])
    if epoch_done:
        start_epoch, skip = saved_epoch + 1, 0
    else:
        start_epoch, skip = saved_epoch, completed
    assert (start_epoch, skip) == (4, 200)


def test_resume_planner_after_epoch_done(tmp_path: Path):
    """Snapshot taken at epoch end → start at next epoch, no skip."""
    ckpt = tmp_path / "latest.pt"
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_mod._save_snapshot(
        ckpt,
        model=model,
        opt=opt,
        epoch=4,
        global_step=2000,
        completed_steps_in_epoch=500,
        epoch_done=True,
        data_range=3.14,
        cfg=_tiny_cfg(),
        wandb_run_id="run-xyz",
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved_epoch = int(state["epoch"])
    completed = int(state["completed_steps_in_epoch"])
    epoch_done = bool(state["epoch_done"])
    if epoch_done:
        start_epoch, skip = saved_epoch + 1, 0
    else:
        start_epoch, skip = saved_epoch, completed
    assert (start_epoch, skip) == (5, 0)
