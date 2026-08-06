#!/usr/bin/env python3
"""Does declaring z=1.31 vs z=1.5 help CLIPScore, the way it helps FID?

It cannot be assumed, because the two eval chains treat z asymmetrically:

  FID  : z resampled to 1 mm, then PADDED into a 512^3 frame — a 256-slice volume at either
         z stays well under 512, so both z declarations are pad-only. z is symmetric.
  CLIP : z resampled to 1.5 mm, then center-crop/pad to 240 slices. A 256-slice volume:
           z=1.50 -> 256*1.50/1.5 = 256 slices -> CROP to 240 (16 real slices truncated)
           z=1.31 -> 256*1.31/1.5 = 224 slices -> PAD to 240 (16 air slices added)
         So z=1.5 removes anatomy and z=1.31 dilutes with air — opposite operations.

This measures CLIP (and, for cross-check, FID) with the production `CTGenEvaluator`, on the
byte-identical content sets re-declared at a grid of (in-plane, z) spacings. Only the header
changes; the voxels never do.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/z_sweep_clip.py --content gen_sp0.8
    CUDA_VISIBLE_DEVICES=3 python tests/spacing_fov/z_sweep_clip.py --content gen_sp1.0
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
CELLS = HERE / "_zcells"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_lps_reeval/gt_lps_seed100")
PROMPT_XLSX = pathlib.Path(
    "/workspace/data/vlm3d_eval/_lps_reeval/prompts_seed100.xlsx"
)
# CLIP-only: FID at z=1.31 is already known from the gated replica sweep, and the FID feature
# extraction is the GPU-heavy step that made this slow. Turn it off — CLIP eval processes one
# volume pair at a time (~a few GB), so it stays well clear of OOM on a shared GPU.
METRICS = {"fvd": False, "fvd_ctclip": False, "clip_score": True, "fid_2p5d": False}

# (in-plane mm, z mm) — hold in-plane at the two FID optima, vary z between the two candidates.
CONFIGS: list[tuple[float, float]] = [
    (0.8, 1.5),
    (0.8, 1.31),
    (0.757, 1.5),
    (0.757, 1.31),
    (0.77, 1.5),  # training MEAN in-plane (0.774)
    (0.77, 1.31),
    # high-z tail: does CLIP keep rising past 1.5, or peak? (more truncation as z grows)
    (0.8, 1.6),
    (0.8, 1.7),
    (0.8, 1.8),
    (0.8, 2.0),
]


def _materialise(
    content: str,
    sx: float,
    sz: float,
    out_dir: pathlib.Path,
    src_dir: pathlib.Path | None = None,
) -> None:
    """Re-declare the content set at ``(sx, sx, sz)``; header-only, voxels untouched."""
    src_dir = src_dir or HERE / "preds" / content
    files = sorted(src_dir.glob("*.mha"))
    if not files:
        sys.exit(f"no volumes in {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    want = (round(sx, 6), round(sx, 6), round(sz, 6))
    for f in files:
        dst = out_dir / f.name
        if dst.exists():
            try:
                # A full ReadImage, not just ReadImageInformation: a file killed mid-write can
                # have an intact header (spacing/size metadata) but a truncated COMPRESSED data
                # block — ReadImageInformation only parses the header and would call that file
                # "fine", masking the corruption until something tries to decode the pixels.
                existing = sitk.ReadImage(str(dst))
                if tuple(round(float(s), 6) for s in existing.GetSpacing()) == want:
                    continue
            except RuntimeError:
                # Unreadable/truncated (e.g. a killed prior run) — fall through to rewrite.
                pass
            dst.unlink()
        img = sitk.ReadImage(str(f))
        img.SetSpacing([sx, sx, sz])
        sitk.WriteImage(img, str(dst), useCompression=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--prompts", default=str(PROMPT_XLSX))
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for the output json (keeps n=100 and n=300 separate)",
    )
    ap.add_argument(
        "--config-pairs",
        nargs="*",
        default=None,
        metavar="SX,SZ",
        help="explicit 'inplane,z' spacings (overrides CONFIGS) — for a non-480 grid (Wan)",
    )
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    # .resolve() is required, not cosmetic: CTGenEvaluator's CLIP step shells out to
    # evaluate_clip.py with cwd=<vlm3d_dockers eval dir>, so a relative path here would
    # resolve against THAT directory inside the subprocess and silently fail to be found.
    gt_dir = pathlib.Path(args.gt_dir).resolve()
    prompts = pathlib.Path(args.prompts).resolve()
    configs = (
        [tuple(float(x) for x in p.split(",")) for p in args.config_pairs]
        if args.config_pairs is not None
        else CONFIGS
    )

    out_path = RESULTS / f"zsweep_clip__{args.content}{args.tag}.json"
    out = json.loads(out_path.read_text()) if out_path.is_file() else {}

    for sx, sz in configs:
        key = f"{sx}_{sz}"
        if key in out and out[key].get("CLIPScore") is not None:
            print(
                f"[z-sweep] {key} already computed (CLIP {out[key]['CLIPScore']:.2f}) — skip"
            )
            continue
        cell_dir = CELLS / f"{args.content}__{key}"
        _materialise(args.content, sx, sz, cell_dir)
        print(f"\n===== {args.content} declared ({sx}, {sx}, {sz}) =====")
        ev = CTGenEvaluator(gt_dir=gt_dir, metrics=METRICS, prompt_xlsx=prompts)
        out[key] = ev.evaluate(cell_dir, HERE / "runs" / f"z_{args.content}_{key}")
        out_path.write_text(json.dumps(out, indent=2))

    print(f"\n{'declared':16s}{'FID':>8s}{'CLIP-T2I':>10s}{'CLIP-I2I':>10s}")
    for sx, sz in configs:
        v = out[f"{sx}_{sz}"]
        print(
            f"({sx},{sx},{sz})".ljust(16)
            + f"{v.get('FID_2p5D_Avg', float('nan')):8.2f}"
            + f"{v.get('CLIPScore', float('nan')):10.2f}"
            + f"{v.get('CLIPScore_I2I', float('nan')):10.2f}"
        )
    print(f"\n[z-sweep] → {out_path}")


if __name__ == "__main__":
    main()
