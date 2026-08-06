"""FID-profile plumbing: cache-key separation, name-filtered linking, exact-set scoring.

These cover the silent-failure modes the two 2.5D-FID profiles introduce — a wrong number
that still looks plausible — rather than the FID math itself (that lives in
``test_fid_gpu_frechet.py`` / ``test_subgroup_refstats.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.eval.tasks.ctgen import (
    _DEFAULT_FID_PROFILE,
    _DEFAULT_FID_PROFILES,
    _DEFAULT_SUBGROUP_FID_PROFILE,
    _FID_PROFILES,
    _RESEARCH_FID_MODEL,
    _fid_from_cached_features,
    _link_shared_features,
    _missing_names,
    _shared_gt_feat_dir,
    _shared_pred_feat_dir,
    CTGenEvaluator,
    resolve_fid_profiles,
)

_DIM = 4
_SLICES = 32


def _write_feature(path: Path, seed: int) -> None:
    """Write one per-volume feature file: a 3-tuple of ``(32, 4)`` plane features."""
    path.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    torch.save(tuple(torch.randn(_SLICES, _DIM, generator=g) for _ in range(3)), path)


def test_profiles_never_share_a_feature_directory():
    """Upstream names feature files after the volume only, so the dirs must differ per profile —
    this is the actual safety-critical invariant (a shared feat_subdir risks the
    os.path.isfile-cache-check collision described in the module comment). Sharing a `model` is
    fine and intentional (e.g. "docker_n300" deliberately reuses "docker"'s squeezenet1_1 network
    so it reproduces the container's FID scale at a larger volume count) — the cross-run shared
    GT-feature cache and subgroup ref-stats cache are correctly keyed on model alone so profiles
    sharing a network reuse each other's already-extracted GT features."""
    subdirs = {name: p["feat_subdir"] for name, p in _FID_PROFILES.items()}
    assert len(set(subdirs.values())) == len(subdirs), subdirs


def test_research_cache_key_is_unchanged(tmp_path: Path):
    """The research key must keep addressing the existing on-disk cache (38 GB + refstats npz)."""
    gt = tmp_path / "_valid_full_3001"
    key = _shared_gt_feat_dir(gt, _RESEARCH_FID_MODEL).name
    assert key == (
        "_valid_full_3001__radimagenet_resnet50__512x512x512__rs1.0x1.0x1.0__pad1_crop1"
    )
    docker_key = _shared_gt_feat_dir(gt, _FID_PROFILES["docker"]["model"]).name
    assert docker_key != key and "squeezenet1_1" in docker_key


def test_shared_gt_dir_requires_an_explicit_model(tmp_path: Path):
    """No default: a caller must never silently inherit a changed global feature network."""
    with pytest.raises(TypeError):
        _shared_gt_feat_dir(tmp_path)  # type: ignore[call-arg]


def test_shared_pred_dir_is_scoped_to_the_run_not_global(tmp_path: Path):
    """Unlike GT (one fixed cross-run root), predictions differ per run — the cache must live
    beside that run's own predictions, keyed on the feature model within it."""
    run_a, run_b = (
        tmp_path / "run_a" / "predictions",
        tmp_path / "run_b" / "predictions",
    )
    a = _shared_pred_feat_dir(run_a, "squeezenet1_1")
    b = _shared_pred_feat_dir(run_b, "squeezenet1_1")
    assert a != b
    assert run_a.parent in a.parents


def test_shared_pred_dir_does_not_collide_across_pred_dirs_sharing_one_parent(
    tmp_path: Path,
):
    """Regression for the harness pattern used by tests/orientation_quant/run_quant.py and
    tests/spacing_fov/decompose_2x2.py: several DIFFERENT prediction sets (different
    models/cells) can live under one shared parent directory and reuse the same case-id
    filenames for genuinely different volume content. Keying only on the parent (the original
    Part B implementation) would make one model's/cell's cached features silently answer for
    another's — this is that exact scenario, minus the parent-only key."""
    shared_parent = tmp_path / "preds"
    report2ct = shared_parent / "report2ct"
    wan = shared_parent / "wan"
    a = _shared_pred_feat_dir(report2ct, "squeezenet1_1")
    b = _shared_pred_feat_dir(wan, "squeezenet1_1")
    assert a != b

    # Same pred_dir, asked twice (e.g. docker then docker_n300 in one run) -> same cache dir,
    # or the whole sharing mechanism has nothing to link against.
    assert _shared_pred_feat_dir(report2ct, "squeezenet1_1") == a


def test_missing_names_filters_only_what_is_locally_absent(tmp_path: Path):
    """The partial-cache filter must key on local presence alone — it doesn't care how a file
    got there (hardlinked from a shared cache or left over from a prior partial run)."""
    feat_dir = tmp_path / "pred"
    feat_dir.mkdir()
    (feat_dir / "a.mha").touch()
    (feat_dir / "c.mha").touch()
    assert _missing_names(feat_dir, ["a.mha", "b.mha", "c.mha", "d.mha"]) == [
        "b.mha",
        "d.mha",
    ]
    assert _missing_names(feat_dir, ["a.mha", "c.mha"]) == []  # everything cached
    assert _missing_names(feat_dir, ["x.mha"]) == ["x.mha"]  # nothing cached


def test_evaluator_default_profile_and_validation(tmp_path: Path):
    assert CTGenEvaluator(gt_dir=tmp_path).fid_profile == _DEFAULT_FID_PROFILE
    assert (
        CTGenEvaluator(gt_dir=tmp_path, fid_profile="research").fid_profile
        == "research"
    )
    with pytest.raises(ValueError, match="unknown fid_profile"):
        CTGenEvaluator(gt_dir=tmp_path, fid_profile="nope")


def test_config_defaults_match_the_code_defaults():
    """The Hydra config and the code default must not drift apart — a mismatch would make the
    documented default a lie for anyone reading either one alone.

    run_eval.py deliberately passes no fallback of its own (``cfg.task.get("fid_profile")`` →
    None → ``_DEFAULT_FID_PROFILES``), so there is exactly ONE place a default is written down
    in code. CTGenEvaluator keeps its own single-profile default; it is a different contract
    (one instance, one feature space) and is covered by the test above."""
    import yaml

    cfg = yaml.safe_load(open("/workspace/configs/eval/task/ctgen.yaml"))
    assert cfg["fid_profile"] == list(_DEFAULT_FID_PROFILES)
    assert all(p in _FID_PROFILES for p in cfg["fid_profile"])
    # The single-profile default must remain one of the profiles the run default scores, so a
    # rescore of an existing run lands beside — not next to — what run_eval.py just wrote.
    assert _DEFAULT_FID_PROFILE in _DEFAULT_FID_PROFILES

    # run_eval.py must not re-declare a default of its own.
    src = Path("/workspace/scripts/run_eval.py").read_text()
    assert 'resolve_fid_profiles(cfg.task.get("fid_profile"))' in src

    # The separate subgroup_fid_profile default (independent knob, own constant — see
    # _DEFAULT_SUBGROUP_FID_PROFILE's docstring) must not drift from the yaml either. Every
    # Python call site imports the constant directly, so this is the only place drift could
    # still sneak in: the yaml value, which can't import Python.
    assert cfg["subgroup_fid_profile"] == _DEFAULT_SUBGROUP_FID_PROFILE
    assert _DEFAULT_SUBGROUP_FID_PROFILE in _FID_PROFILES


def test_resolve_fid_profiles_normalises_and_validates():
    """`task.fid_profile` accepts one name or a list; unknown/empty must fail where the mistake
    was made, not by silently skipping FID."""
    assert resolve_fid_profiles(None) == list(_DEFAULT_FID_PROFILES)
    assert resolve_fid_profiles("research") == [
        "research"
    ]  # historical single-name form
    assert resolve_fid_profiles(["research", "docker"]) == [
        "research",
        "docker",
    ]  # order kept
    assert resolve_fid_profiles(["docker", "docker"]) == [
        "docker"
    ]  # would double-score a dir
    with pytest.raises(ValueError, match="unknown fid_profile"):
        resolve_fid_profiles(["docker", "nope"])
    with pytest.raises(ValueError, match="is empty"):
        resolve_fid_profiles([])


def test_subprocess_paths_are_absolute(tmp_path: Path, monkeypatch):
    """Every metric's upstream script runs with cwd=<vlm3d_dockers eval dir>, so a relative path
    handed to the child resolves against THAT directory and it dies with FileNotFoundError."""
    from src.eval.tasks import ctgen as ctgen_mod

    (tmp_path / "gt").mkdir()
    (tmp_path / "pred").mkdir()
    for name in ("a.mha", "b.mha"):  # non-empty dirs so _run_fid gets past its guards
        (tmp_path / "gt" / name).touch()
        (tmp_path / "pred" / name).touch()

    captured: dict = {}

    class _Result:
        returncode, stdout, stderr = 1, "", ""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ctgen_mod.subprocess,
        "run",
        lambda cmd, **kw: (captured.update(cmd=cmd), _Result())[1],
    )

    ev = CTGenEvaluator(gt_dir="gt")  # relative on purpose
    assert ev.gt_dir.is_absolute()
    ev._run_fid(Path("pred"), Path("out"))

    path_args = [
        a.split("=", 1)[1]
        for a in captured["cmd"]
        if a.startswith("--") and "=" in a and "/" in a.split("=", 1)[1]
    ]
    assert path_args, captured["cmd"]
    assert all(Path(p).is_absolute() for p in path_args), path_args


def test_link_shared_features_honours_the_name_filter(tmp_path: Path):
    """docker links only its scored stems; research (names=None) links the whole cache."""
    shared = tmp_path / "shared"
    for i, name in enumerate(["a.mha", "b.mha", "c.mha"]):
        _write_feature(shared / name, seed=i)

    subset = tmp_path / "run_subset" / "gt"
    subset.mkdir(parents=True)
    assert _link_shared_features(shared, subset, names={"a.mha", "c.mha"}) == 2
    assert {p.name for p in subset.iterdir()} == {"a.mha", "c.mha"}

    everything = tmp_path / "run_all" / "gt"
    everything.mkdir(parents=True)
    assert _link_shared_features(shared, everything, names=None) == 3
    assert {p.name for p in everything.iterdir()} == {"a.mha", "b.mha", "c.mha"}


def test_fid_scores_only_the_named_files(tmp_path: Path):
    """A stale dir holding extra features must not enlarge either distribution."""
    clean, dirty = tmp_path / "clean", tmp_path / "dirty"
    names = ["a.mha", "b.mha", "c.mha"]
    for base in (clean, dirty):
        for i, n in enumerate(names):
            _write_feature(base / "gt" / n, seed=100 + i)
            _write_feature(base / "pred" / n, seed=200 + i)
    # `dirty` additionally holds leftovers from some earlier, different set.
    for i, n in enumerate(["stale1.mha", "stale2.mha"]):
        _write_feature(dirty / "gt" / n, seed=900 + i)
        _write_feature(dirty / "pred" / n, seed=950 + i)

    expected = _fid_from_cached_features(clean, names, names)
    got = _fid_from_cached_features(dirty, names, names)
    assert expected is not None and got is not None
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, abs=1e-12), key


def test_ref_stats_fast_path_also_scores_only_the_named_files(tmp_path: Path):
    """The research default goes through the precomputed-(mu, Sigma) branch, which used to glob
    the prediction dir — so stale extras moved the FID while the full-stack branch was clean."""
    from src.eval.tasks._fid_refstats import compute_ref_stats, save_ref_stats

    names = [f"s{i}.mha" for i in range(6)]
    clean, dirty = tmp_path / "clean", tmp_path / "dirty"
    for base in (clean, dirty):
        for i, n in enumerate(names):
            _write_feature(base / "gt" / n, seed=100 + i)
            _write_feature(base / "pred" / n, seed=200 + i)
    for i, n in enumerate(
        ["x1.mha", "x2.mha"]
    ):  # leftovers from an earlier, different set
        _write_feature(dirty / "gt" / n, seed=900 + i)
        _write_feature(dirty / "pred" / n, seed=950 + i)

    npz = tmp_path / "ref.npz"
    save_ref_stats(compute_ref_stats(clean / "gt", names=names), npz)

    expected = _fid_from_cached_features(clean, names, names, ref_stats_path=npz)
    got = _fid_from_cached_features(dirty, names, names, ref_stats_path=npz)
    assert expected is not None and got is not None
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, abs=1e-12), key


def test_run_fid_reuses_shared_pred_and_gt_features_across_docker_profiles(
    tmp_path: Path, monkeypatch
):
    """The composed, load-bearing invariant behind the whole caching optimization: whatever
    `_run_fid` skips loading (via the shared-cache hardlinks + the partial-cache filter) must
    never change the reported FID. Runs "docker" then "docker_n300" against the same
    predictions dir (docker's stems are always the first N of docker_n300's), with a fake
    subprocess that only extracts features for whatever it's actually handed — so any name
    silently missing from the final feature set would surface as a wrong/NaN FID, not merely a
    speed difference the earlier, isolated-helper tests couldn't catch."""
    import src.eval.tasks.ctgen as ctgen_mod

    shared_gt_root = tmp_path / "_shared_gt_root"
    shared_gt_root.mkdir()
    monkeypatch.setattr(ctgen_mod, "_SHARED_GT_FEAT_ROOT", shared_gt_root)
    monkeypatch.setitem(
        ctgen_mod._FID_PROFILES,
        "docker",
        {**ctgen_mod._FID_PROFILES["docker"], "num_images": 3},
    )
    monkeypatch.setitem(
        ctgen_mod._FID_PROFILES,
        "docker_n300",
        {**ctgen_mod._FID_PROFILES["docker_n300"], "num_images": 6},
    )

    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "predictions"
    gt_dir.mkdir()
    pred_dir.mkdir()
    names = [f"s{i}.mha" for i in range(6)]
    for n in names:
        (gt_dir / n).touch()
        (pred_dir / n).touch()

    def _fake_run(cmd, **kw):
        args = dict(a[2:].split("=", 1) for a in cmd if a.startswith("--") and "=" in a)
        out_root = Path(args["output_root"])
        for filelist_arg, subdir in (
            ("real_filelist", "gt"),
            ("synth_filelist", "pred"),
        ):
            for name in Path(args[filelist_arg]).read_text().splitlines():
                if name:
                    seed = (100 if subdir == "gt" else 200) + names.index(name)
                    _write_feature(out_root / subdir / name, seed=seed)

        class _Result:
            returncode, stdout, stderr = 0, "", ""

        return _Result()

    monkeypatch.setattr(ctgen_mod.subprocess, "run", _fake_run)

    out_dir = tmp_path / "run"
    docker_result = CTGenEvaluator(gt_dir=gt_dir, fid_profile="docker")._run_fid(
        pred_dir, out_dir / "fid_docker"
    )
    n300_result = CTGenEvaluator(gt_dir=gt_dir, fid_profile="docker_n300")._run_fid(
        pred_dir, out_dir / "fid_docker_n300"
    )
    assert not any(v != v for v in docker_result.values()), docker_result  # no NaN
    assert not any(v != v for v in n300_result.values()), n300_result

    # Independent reference: the same seeds written directly into a fully clean, uncached
    # feature set (no shared cache, no filtering) — must match docker_n300 bit-for-bit.
    clean = tmp_path / "clean"
    for i, n in enumerate(names):
        _write_feature(clean / "gt" / n, seed=100 + i)
        _write_feature(clean / "pred" / n, seed=200 + i)
    expected = _fid_from_cached_features(clean, names, names)
    assert expected is not None
    for key, value in expected.items():
        assert n300_result[key] == pytest.approx(value, abs=1e-9), key

    # And confirm the sharing actually happened rather than being coincidentally bypassed: the
    # per-pred-dir shared cache must hold all 6 names after both profiles ran.
    shared_pred = ctgen_mod._shared_pred_feat_dir(pred_dir, "squeezenet1_1")
    assert {p.name for p in shared_pred.iterdir()} == set(names)


def test_fid_returns_none_when_a_named_feature_is_missing(tmp_path: Path):
    names = ["a.mha", "b.mha"]
    for i, n in enumerate(names):
        _write_feature(tmp_path / "gt" / n, seed=i)
    _write_feature(tmp_path / "pred" / "a.mha", seed=7)  # "b.mha" never extracted
    assert _fid_from_cached_features(tmp_path, names, names) is None


def test_docker_n300_shares_docker_model_but_not_its_dirs():
    """docker_n300 must reproduce the container's feature network (so its FID scale matches
    "docker", unlike "research") while still satisfying the feat_subdir-uniqueness invariant."""
    docker, n300 = _FID_PROFILES["docker"], _FID_PROFILES["docker_n300"]
    assert n300["model"] == docker["model"] == "squeezenet1_1"
    assert n300["feat_subdir"] != docker["feat_subdir"]
    assert n300["num_images"] == 300
    assert n300["num_images"] != docker["num_images"]


def test_docker_n300_is_not_the_single_instance_default():
    """`docker_n300` IS in the default run_eval.py profile LIST (_DEFAULT_FID_PROFILES, since
    2026-08-01) — this test is about a different, single-profile constant: CTGenEvaluator's own
    constructor default / rescore_predictions.py's bare default, which stays the
    container-faithful "docker" so a single-profile rescore lands beside what run_eval.py's
    first pass already wrote."""
    assert _DEFAULT_FID_PROFILE == "docker"


# --------------------------------------------------------------------------- #
#  Clobber / cache-provenance guards (added 2026-07-30 after both fired for real)
# --------------------------------------------------------------------------- #


def test_feature_dir_is_bound_to_one_gt_set(tmp_path: Path):
    """Same basenames + different content (gt_lps vs gt_ras) must not reuse each other's
    features: upstream skips extraction on filename alone, so the second GT set would silently
    be scored with the first set's features."""
    feats = tmp_path / "fid_features"
    feats.mkdir()
    ev_a = CTGenEvaluator(gt_dir=tmp_path / "gt_lps")
    ev_b = CTGenEvaluator(gt_dir=tmp_path / "gt_ras")

    assert ev_a._gt_set_matches(feats) is True  # first use records the GT set
    assert ev_a._gt_set_matches(feats) is True  # same GT set -> still fine
    assert ev_b._gt_set_matches(feats) is False  # different GT set -> refused


def test_gt_set_marker_stays_out_of_the_shared_cache(tmp_path: Path):
    """The marker must NOT live in features_dir/gt/: _populate_shared_features copies every
    file there into the cross-model cache, which _link_shared_features then hardlinks into
    other runs — the marker would travel with it."""
    feats = tmp_path / "fid_features"
    (feats / "gt").mkdir(parents=True)
    CTGenEvaluator(gt_dir=tmp_path / "gt_lps")._gt_set_matches(feats)
    assert (feats / ".gt_set").is_file()
    assert not any(p.name == ".gt_set" for p in (feats / "gt").iterdir())


def test_run_fid_refuses_a_second_gt_set(tmp_path: Path, monkeypatch):
    """End to end through _run_fid: the refusal returns NaN rather than a wrong number."""
    from src.eval.tasks import ctgen as ctgen_mod

    for split in ("gt_lps", "gt_ras", "pred"):
        (tmp_path / split).mkdir()
        for name in ("a.mha", "b.mha"):
            (tmp_path / split / name).touch()
    monkeypatch.setattr(
        ctgen_mod.subprocess, "run", lambda *a, **k: pytest.fail("must not run")
    )

    feats = tmp_path / "out" / _FID_PROFILES[_DEFAULT_FID_PROFILE]["feat_subdir"]
    feats.mkdir(parents=True)
    (feats / ".gt_set").write_text(str((tmp_path / "gt_lps").resolve()) + "\n")

    got = CTGenEvaluator(gt_dir=tmp_path / "gt_ras")._run_fid(
        tmp_path / "pred", tmp_path / "out"
    )
    assert all(v != v for v in got.values()), got  # all NaN


def test_profiles_cannot_clobber_each_other_because_they_land_in_different_folders(
    tmp_path: Path,
):
    """The 2026-07-29 clobber (two wan_mask_v2 research FIDs lost to a docker re-run) is now
    prevented by the LAYOUT, not by a guard: every scoring pass — run_eval.py's and
    rescore_predictions.py's alike — writes to ``<eval_dir>/fid_<profile>/``, so a docker pass
    physically cannot reach a research pass's metrics.json. The runtime guard
    ``_refuse_cross_profile_overwrite`` was deleted on 2026-07-31 as redundant."""
    import json

    from src.eval.tasks.ctgen import _merge_metrics

    research = tmp_path / "fid_research"
    docker = tmp_path / "fid_docker"
    research.mkdir()
    docker.mkdir()

    _merge_metrics(
        research, {"FID_2p5D_Avg": 1.4556, "fid_profile": "research"}, ["fid_2p5d"]
    )
    _merge_metrics(
        docker, {"FID_2p5D_Avg": 61.20, "fid_profile": "docker"}, ["fid_2p5d"]
    )

    assert json.loads((research / "metrics.json").read_text())["FID_2p5D_Avg"] == 1.4556
    assert json.loads((docker / "metrics.json").read_text())["FID_2p5D_Avg"] == 61.20
    # Nothing at the eval-dir top level to be ambiguous about.
    assert not (tmp_path / "metrics.json").exists()


def test_pre_existing_feature_dir_is_adopted_with_a_warning(tmp_path: Path, caplog):
    """Dirs populated before the marker existed must keep working — but say the provenance is
    unverified. Feature files are named `<volume>.mha` (upstream's .pt rename never matches our
    input), so a `*.pt` glob here would silently never fire."""
    feats = tmp_path / "fid_features"
    (feats / "gt").mkdir(parents=True)
    (feats / "gt" / "valid_1005_a_1.mha").touch()  # a real feature filename

    with caplog.at_level("WARNING"):
        assert CTGenEvaluator(gt_dir=tmp_path / "gt_lps")._gt_set_matches(feats) is True
    assert "no .gt_set marker" in caplog.text
    assert (feats / ".gt_set").read_text().strip() == str(
        (tmp_path / "gt_lps").resolve()
    )


def test_run_eval_refuses_a_split_run_dir(tmp_path: Path):
    """Overriding hydra.run.dir= (instead of out_dir=) sends results and .hydra/+log to different
    folders, leaving the results with no record of the command that made them."""
    import sys

    sys.path.insert(0, "/workspace/scripts")
    from omegaconf import OmegaConf

    from run_eval import _refuse_split_run_dir

    results, elsewhere = tmp_path / "results", tmp_path / "elsewhere"
    results.mkdir()
    elsewhere.mkdir()

    _refuse_split_run_dir(OmegaConf.create({"out_dir": str(results)}), str(results))
    # same folder reached by a non-normalised path is still the same folder
    _refuse_split_run_dir(
        OmegaConf.create({"out_dir": f"{results}/../results"}), str(results)
    )
    with pytest.raises(RuntimeError, match="different folders"):
        _refuse_split_run_dir(
            OmegaConf.create({"out_dir": str(results)}), str(elsewhere)
        )
