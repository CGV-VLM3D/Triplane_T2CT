#!/usr/bin/env python3
"""Gather only the small result files from outputs/ into outputs/gather_summary/.

Mirrors the folder structure of /workspace/outputs but copies ONLY small files
(summary.json / metrics.json / clip.json / fid.json / fvd_ctclip.json / gen_params.txt /
.hydra configs / run_eval.log …) — excluding predictions, *.mha, latents, fid_features, and
anything large. Also writes an ``info.txt`` (model / spacing / voxel_dim / cfg / n_preds) into
each gathered eval dir so the run config is captured without needing the volumes.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

import SimpleITK as sitk

sys.path.insert(0, "/workspace")
from scripts.write_gen_params import _first_pred, _parse_cfg  # noqa: E402

SRC = pathlib.Path("/workspace/outputs")
DST = SRC / "gather_summary"
RESULT_JSONS = {
    "summary.json",
    "metrics.json",
    "clip.json",
    "fid.json",
    "fvd_ctclip.json",
    "fvd.json",
}
KEEP_EXT = {".json", ".txt", ".yaml", ".log"}
SKIP_DIRS = {"predictions", "fid_features", "latents", "gather_summary"}
MAX_BYTES = 2_000_000  # 2 MB guard: never copy anything large


def _write_info(src_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Write info.txt (model / spacing / voxel_dim / cfg / n_preds) from the source dir."""
    pred = _first_pred(src_dir)
    dim = sp = "??"
    if pred is not None:
        r = sitk.ImageFileReader()
        r.SetFileName(str(pred))
        r.ReadImageInformation()
        dim = str(r.GetSize())
        sp = str(tuple(round(s, 4) for s in r.GetSpacing()))
    cfg, cfg_src = _parse_cfg(src_dir)
    n_preds = (
        len(list((src_dir / "predictions").glob("*.mha")))
        if (src_dir / "predictions").is_dir()
        else 0
    )
    lps = [d.name for d in src_dir.glob("eval_LPS_n*") if (d / "metrics.json").exists()]
    lines = [
        f"model:       {src_dir.parent.name}",
        f"eval_dir:    {src_dir.relative_to(SRC)}",
        f"voxel_dim:   {dim}   # (x, y, z)",
        f"spacing_mm:  {sp}    # (x, y, z)",
        f"cfg:         {cfg}   # source: {cfg_src}",
        f"n_preds:     {n_preds}",
        f"lps_reevals: {', '.join(sorted(lps)) if lps else '(none)'}",
    ]
    (out_dir / "info.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    n_dirs = n_files = 0
    for dirpath, dirnames, filenames in os.walk(SRC):
        dp = pathlib.Path(dirpath)
        if dp == DST or DST in dp.parents:
            dirnames[:] = []
            continue
        # startswith: the FID feature dir is named per profile (fid_features,
        # fid_features_squeezenet1_1, ...) — never descend into any of them.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith("fid_features")
        ]

        # only mirror dirs that actually hold a result json
        if not (RESULT_JSONS & set(filenames)):
            continue
        rel = dp.relative_to(SRC)
        out_dir = DST / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in filenames:
            p = dp / f
            if (
                p.suffix in KEEP_EXT
                and not f.endswith(".mha")
                and p.stat().st_size <= MAX_BYTES
            ):
                shutil.copy2(p, out_dir / f)
                n_files += 1
        # info.txt only at the top eval dir (has predictions/); LPS subdirs keep their gen_params.txt
        if (dp / "predictions").is_dir():
            try:
                _write_info(dp, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] info.txt failed for {rel}: {e}")
        n_dirs += 1
    print(f"gathered {n_files} files across {n_dirs} result dirs → {DST}")


if __name__ == "__main__":
    main()
