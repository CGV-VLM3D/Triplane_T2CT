#!/usr/bin/env python3
"""E5 — 2x2 decomposition: does the FID gap come from the conditioning or the declaration?

    +--------------------+--------------------+--------------------+
    |                    | header says 1.0    | header says 0.8    |
    +--------------------+--------------------+--------------------+
    | generated at 1.0   | A  (the old setup) | B                  |
    | generated at 0.8   | D                  | C  (current setup) |
    +--------------------+--------------------+--------------------+

    A -> B  and  D -> C : only the declared spacing changed, voxels identical  => effect (A)
    A -> D  and  B -> C : only the conditioning changed, declaration identical => effect (B)
    A -> C              : the total gap that was actually observed in production

Both generated arms come from the same per-case noise seed (gen_seeded.py), so the A->D and
B->C comparisons are paired: nothing differs except the spacing fed to the UNet.

Every cell is scored with the **production** `CTGenEvaluator` against the same GT and prompt
file the reported numbers use, so the values are directly comparable to existing results.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/decompose_2x2.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import SimpleITK as sitk

sys.path.insert(0, "/workspace")

from src.eval.tasks.ctgen import CTGenEvaluator  # noqa: E402

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
CELLS_DIR = HERE / "_cells"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_lps_reeval/gt_lps_seed100")
PROMPT_XLSX = pathlib.Path(
    "/workspace/data/vlm3d_eval/_lps_reeval/prompts_seed100.xlsx"
)
METRICS = {"fvd": False, "fvd_ctclip": True, "clip_score": True, "fid_2p5d": True}

Z_MM = (
    1.5  # both arms were generated with z=1.5; only the in-plane label is swapped here
)

# cell key -> (content dir under preds/, declared in-plane spacing)
CELLS: dict[str, tuple[str, float]] = {
    "A_gen1.0_label1.0": ("gen_sp1.0", 1.0),
    "B_gen1.0_label0.8": ("gen_sp1.0", 0.8),
    "C_gen0.8_label0.8": ("gen_sp0.8", 0.8),
    "D_gen0.8_label1.0": ("gen_sp0.8", 1.0),
}


def _shared_cases() -> list[str]:
    """Cases present in EVERY content set, so all four cells are scored on identical patients.

    Without this the cells silently use whatever each arm happens to contain, and the
    conditioning rows of the 2x2 confound spacing with patient composition.
    """
    common: set[str] | None = None
    for content in {c for c, _ in CELLS.values()}:
        ids = {p.stem for p in (HERE / "preds" / content).glob("*.mha")}
        common = ids if common is None else (common & ids)
    return sorted(common or [])


def _materialise(
    content: str, label: float, cases: list[str], out_dir: pathlib.Path
) -> int:
    """Populate ``out_dir`` with the content set declared at ``label`` mm in-plane.

    When the label already matches how the volumes were generated the files are symlinked
    (no copy); otherwise each volume is rewritten with the substituted header spacing. Only
    the header changes — the voxel array is passed through untouched.

    Returns the number of volumes in the dir.
    """
    src_dir = HERE / "preds" / content
    files = [src_dir / f"{c}.mha" for c in cases]
    missing = [f for f in files if not f.is_file()]
    if missing:
        sys.exit(f"{len(missing)} shared cases missing from {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    native = sitk.ReadImage(str(files[0])).GetSpacing()[0]
    identity = abs(native - label) < 1e-6

    for f in files:
        dst = out_dir / f.name
        # Reuse only if the existing file really carries the header this cell needs — a cell
        # dir left over from a different Z_MM / label would otherwise be silently rescored.
        if (dst.exists() or dst.is_symlink()) and _declared(dst) == (
            label,
            label,
            Z_MM,
        ):
            continue
        dst.unlink(missing_ok=True)
        if identity:
            dst.symlink_to(f)
            continue
        img = sitk.ReadImage(str(f))
        img.SetSpacing([label, label, Z_MM])
        sitk.WriteImage(img, str(dst), useCompression=True)

    stale = [p for p in out_dir.glob("*.mha") if p.stem not in set(cases)]
    for p in stale:  # e.g. a wider case set from an earlier, partially-generated run
        p.unlink()
    return len(files)


def _declared(path: pathlib.Path) -> tuple[float, float, float] | None:
    """Header spacing of an existing cell file, or None if it cannot be read."""
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(str(path))
        r.ReadImageInformation()
        return tuple(round(float(s), 6) for s in r.GetSpacing())
    except RuntimeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", nargs="*", default=None, help="subset of cell keys to run"
    )
    ap.add_argument(
        "--require",
        type=int,
        default=100,
        help="minimum number of cases shared by every content set before scoring",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = {k: v for k, v in CELLS.items() if args.only is None or k in args.only}

    cases = _shared_cases()
    if len(cases) < args.require:
        sys.exit(
            f"only {len(cases)} cases are present in every content set (need {args.require}). "
            "Scoring the cells on different patients would confound the spacing effect with "
            "patient composition. Wait for generation to finish, or lower --require knowing "
            "the result is a preview."
        )
    print(f"[E5] {len(cases)} cases shared by every content set")

    out: dict[str, dict] = {}
    out_path = RESULTS / "decompose.json"
    if out_path.is_file():
        out = json.loads(out_path.read_text())

    for key, (content, label) in cells.items():
        cell_dir = CELLS_DIR / key
        n = _materialise(content, label, cases, cell_dir)
        print(f"\n===== {key}: {n} volumes, header in-plane {label} mm =====")
        ev = CTGenEvaluator(gt_dir=GT_DIR, metrics=METRICS, prompt_xlsx=PROMPT_XLSX)
        out[key] = ev.evaluate(cell_dir, HERE / "runs" / key)
        out_path.write_text(json.dumps(out, indent=2))

    _report(out)


def _report(out: dict) -> None:
    """Print the 2x2 and the two effect sizes, and write results/decompose.md."""
    keys = [
        "FID_2p5D_Avg",
        "FID_2p5D_XY",
        "FID_2p5D_YZ",
        "FID_2p5D_XZ",
        "CLIPScore",
        "CLIPScore_I2I",
        "FVD_CTCLIP",
    ]
    lines = ["# E5 — 2x2 decomposition (n=100, same seed per case)\n"]
    lines.append(f"{'metric':16s}" + "".join(f"{k.split('_')[0]:>22s}" for k in CELLS))
    for m in keys:
        row = f"{m:16s}"
        for cell in CELLS:
            v = out.get(cell, {}).get(m)
            row += f"{v:>22.3f}" if isinstance(v, (int, float)) else f"{'—':>22s}"
        lines.append(row)

    def _d(a: str, b: str, m: str = "FID_2p5D_Avg") -> str:
        va, vb = out.get(a, {}).get(m), out.get(b, {}).get(m)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            return "—"
        return f"{vb - va:+.3f}"

    lines += [
        "",
        "## Effect sizes on FID_2p5D_Avg",
        f"declaration only (A→B, gen@1.0 relabelled to 0.8): {_d('A_gen1.0_label1.0', 'B_gen1.0_label0.8')}",
        f"declaration only (D→C, gen@0.8 relabelled to 0.8): {_d('D_gen0.8_label1.0', 'C_gen0.8_label0.8')}",
        f"conditioning only (A→D, same 1.0 label):           {_d('A_gen1.0_label1.0', 'D_gen0.8_label1.0')}",
        f"conditioning only (B→C, same 0.8 label):           {_d('B_gen1.0_label0.8', 'C_gen0.8_label0.8')}",
        f"total observed  (A→C):                             {_d('A_gen1.0_label1.0', 'C_gen0.8_label0.8')}",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    (RESULTS / "decompose.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
