#!/usr/bin/env python3
"""Write a gen_params.txt (spacing / cfg / voxel dim) next to each LPS re-eval result.

voxel_dim + spacing are read authoritatively from a prediction ``.mha`` header (the in-plane
LPS flip does not change size or spacing). cfg is parsed from, in order: the run's
``.hydra/config.yaml`` (``cfg_scale`` / ``guidance_scale``), ``.hydra/overrides.yaml``, the eval
dir name (``cfg<N>``), then ``run_eval.log`` — falling back to ``??`` when not found.

Idempotent: safe to run repeatedly (e.g. as dirs complete). Writes ``<eval_dir>/eval_LPS_n100/gen_params.txt``.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import re

import SimpleITK as sitk


def _first_pred(eval_dir: pathlib.Path) -> pathlib.Path | None:
    for sub in ("predictions", "eval_LPS_n100/predictions"):
        fs = sorted((eval_dir / sub).glob("*.mha")) if (eval_dir / sub).is_dir() else []
        if fs:
            return fs[0]
    return None


def _parse_cfg(eval_dir: pathlib.Path) -> tuple[str, str]:
    """Return (cfg_value, source). '??' if not found anywhere."""
    hydra = eval_dir / ".hydra"
    # 1. .hydra/config.yaml — cfg_scale or guidance_scale
    for name in ("config.yaml", "overrides.yaml"):
        f = hydra / name
        if f.is_file():
            txt = f.read_text()
            m = re.search(r"(?:cfg_scale|guidance_scale)[:=]\s*([0-9.]+)", txt)
            if m:
                return m.group(1), f".hydra/{name}"
    # 2. dir name cfg<N>
    m = re.search(r"cfg(\d+(?:\.\d+)?)", eval_dir.name)
    if m:
        return m.group(1), "dir name"
    # 3. run_eval.log
    log = eval_dir / "run_eval.log"
    if log.is_file():
        m = re.search(r"(?:cfg_scale|guidance_scale):\s*([0-9.]+)", log.read_text())
        if m:
            return m.group(1), "run_eval.log"
    # 4. nested hydra run dir (e.g. eval_dir/<date>/<time>/.hydra/config.yaml)
    for f in list(eval_dir.rglob(".hydra/config.yaml")) + list(
        eval_dir.rglob("run_eval.log")
    ):
        if "eval_LPS_n100" in f.parts:
            continue
        m = re.search(r"(?:cfg_scale|guidance_scale):\s*([0-9.]+)", f.read_text())
        if m:
            return m.group(1), f"nested {f.relative_to(eval_dir)}"
    return "??", "not found"


def _spacing_tag(eval_dir: pathlib.Path) -> str:
    """Configured spacing_mm from .hydra/log if present, else '' (header is authoritative anyway)."""
    for f in (eval_dir / ".hydra" / "config.yaml", eval_dir / "run_eval.log"):
        if f.is_file():
            m = re.search(
                r"spacing_mm:\s*\n\s*-\s*([0-9.]+)\s*\n\s*-\s*([0-9.]+)\s*\n\s*-\s*([0-9.]+)",
                f.read_text(),
            )
            if m:
                return f"[{m.group(1)},{m.group(2)},{m.group(3)}]"
    return ""


def write_one(eval_dir: pathlib.Path, out_subdir: str = "eval_LPS_n100") -> None:
    pred = _first_pred(eval_dir)
    if pred is None:
        print(f"[skip] no predictions: {eval_dir}")
        return
    r = sitk.ImageFileReader()
    r.SetFileName(str(pred))
    r.ReadImageInformation()
    size = r.GetSize()  # (x,y,z)
    sp = tuple(round(s, 4) for s in r.GetSpacing())  # (x,y,z)
    cfg, cfg_src = _parse_cfg(eval_dir)
    cfg_cfg = _spacing_tag(eval_dir)

    out_dir = eval_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"eval_dir:    {eval_dir}",
        f"model:       {eval_dir.parent.name}",
        f"voxel_dim:   {size}          # (x, y, z) from prediction .mha header",
        f"spacing_mm:  {sp}   # (x, y, z) from prediction .mha header",
        f"cfg:         {cfg}          # source: {cfg_src}",
    ]
    if cfg_cfg:
        lines.append(f"spacing_cfg: {cfg_cfg}   # configured spacing_mm (.hydra/log)")
    (out_dir / "gen_params.txt").write_text("\n".join(lines) + "\n")
    print(f"[ok] {eval_dir.name}: dim={size} sp={sp} cfg={cfg}({cfg_src})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+", help="eval dirs (or globs)")
    args = ap.parse_args()
    dirs: list[str] = []
    for d in args.eval_dirs:
        dirs.extend(glob.glob(d)) if any(c in d for c in "*?[") else dirs.append(d)
    for d in dirs:
        write_one(pathlib.Path(d))


if __name__ == "__main__":
    main()
