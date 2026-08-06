"""Precompute top-K soft organ masks aligned to the text2ct latent grid (128,128,32).

For each scan, the ts_seg TotalSegmentator label map is put through the SAME spatial pipeline
that produced the text2ct image latents (``third_party/text2ct/scripts/prep_toy_v2.py``:
reorient→RAS, resize to the fixed ``(128,512,512)`` (Z,Y,X) grid) — but with **nearest**
interpolation (labels must never be blended) — then 4x4x4 area-pooled to the latent grid and
reduced to the **top-K** ``(class, area_fraction)`` per latent voxel (K=4 is empirically lossless:
top-3 already covers 100% of cell area; ≥5 distinct classes per cell never occur).

Output per scan: ``<out>/<vol>.pt`` = ``{"classes": (K,128,128,32) uint8, "fracs": (K,128,128,32) fp16}``
consumed by MaskConditioner as ``e_mask = Σ_k frac_k · embedding(class_k)`` (= LDM channel_mapper
on the soft one-hot, top-K truncated). ~6 MB/scan.

Alignment (baseline-clone #4 — verify with scripts/verify_mask_alignment.py before trusting):
latent ``(H,W,D)=(128,128,32)`` ↔ CT ``(X,Y,Z)=(512,512,128)``; a latent voxel ``(h,w,d)`` maps to
the CT block ``[4h:4h+4, 4w:4w+4, 4d:4d+4]``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.organ_groups import NUM_GROUPS, apply_grouping  # noqa: E402

NUM_CLASSES = 118
LATENT_HWD = (128, 128, 32)
TARGET_ZYX = (128, 512, 512)  # prep_toy_v2.clip_and_resample final size (Z,Y,X)
TS_TOTAL = "/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/ts_total"


def load_ras_zyx(path: str) -> np.ndarray:
    """Load a label ``.nii.gz``, reorient to RAS, return ``(Z,Y,X)`` int16 (mirrors prep_toy_v2)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)  # native label dtype — avoid float64 blow-up
    t = ornt_transform(io_orientation(img.affine), axcodes2ornt(("R", "A", "S")))
    data_ras = nib.orientations.apply_orientation(data, t)
    return np.transpose(data_ras, (2, 1, 0)).astype(np.int16)  # (X,Y,Z) -> (Z,Y,X)


def mask_to_topk(
    mask_path: str, k: int = 4, device: str = "cuda:0", group_organs: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Aligned top-K soft mask for one scan.

    Args:
        group_organs: if True, remap the 117 TS labels to the 4 흉부 organ group
            (:func:`src.data.organ_groups.apply_grouping`; classes 0..4) before pooling —
            MaskConditioner then uses ``num_classes = NUM_GROUPS + 1 = 5``.

    Returns:
        classes ``(K, 128, 128, 32)`` uint8 and fracs ``(K, 128, 128, 32)`` fp16 (fracs sum to 1
        over K per voxel). Axis order matches the latent ``(C, H, W, D)``.
    """
    m = load_ras_zyx(mask_path)  # (Z,Y,X)
    if group_organs:
        m = apply_grouping(m)  # 117 TS labels -> 4 흉부 organ group (values 0..4)
    n_classes = NUM_GROUPS + 1 if group_organs else NUM_CLASSES  # 5 vs 118
    t = torch.from_numpy(m).float()[None, None].to(device)  # (1,1,Z,Y,X)
    t = F.interpolate(
        t, size=TARGET_ZYX, mode="nearest"
    )  # (1,1,128,512,512) — labels: nearest
    lab = (
        t[0, 0].long().permute(2, 1, 0)
    )  # (Z,Y,X) -> (X,Y,Z)=(512,512,128), matches CT save order
    # 4x4x4 block-pool: (Xc,Xi,Yc,Yi,Zc,Zi) -> (Xc,Yc,Zc, 64)
    blocks = (
        lab.reshape(128, 4, 128, 4, 32, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1, 64)
    )  # (Ncell, 64)
    counts = torch.zeros(blocks.shape[0], n_classes, device=device)
    counts.scatter_add_(
        1, blocks, torch.ones_like(blocks, dtype=torch.float)
    )  # (Ncell, 118)
    # Exactness audit: distinct classes per cell (incl. background). Truncation (approximation)
    # happens ONLY where a cell has > k distinct classes — then top-k drops the smallest-area tail.
    ndist = (counts > 0).sum(1)  # (Ncell,)
    max_distinct = int(ndist.max())
    n_truncated = int((ndist > k).sum())
    topv, topc = counts.topk(k, dim=1)  # (Ncell, K) area counts + class ids, desc
    fr = topv / topv.sum(1, keepdim=True).clamp_min(1.0)  # renormalize top-K -> sum 1
    classes = topc.reshape(*LATENT_HWD, k).permute(3, 0, 1, 2).contiguous()  # (K,H,W,D)
    fracs = fr.reshape(*LATENT_HWD, k).permute(3, 0, 1, 2).contiguous()  # (K,H,W,D)
    return (
        classes.to(torch.uint8).cpu(),
        fracs.to(torch.float16).cpu(),
        max_distinct,
        n_truncated,
    )


def mask_path_for(vol: str, split: str) -> str:
    """``train_10000_a_1`` -> ts_total/<split>/train_10000/train_10000_a/train_10000_a_1.nii.gz."""
    study = vol.rsplit("_", 1)[0]
    patient = study.rsplit("_", 1)[0]
    return os.path.join(TS_TOTAL, split, patient, study, f"{vol}.nii.gz")


def _vols_from_datalist(path: str) -> list[str]:
    d = json.load(open(path))
    out = []
    for split_key in ("training", "validation"):
        for e in d.get(split_key, []):
            base = os.path.basename(e["image"])
            out.append(base.replace("_emb.nii.gz", "").replace(".nii.gz", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute top-K latent-grid organ masks")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--datalist", help="JSON with training/validation image entries -> vol ids"
    )
    g.add_argument(
        "--emb-glob", help="glob of *_emb.nii.gz / *.nii.gz latents -> vol ids"
    )
    ap.add_argument(
        "--split", default="train_fixed", choices=["train_fixed", "valid_fixed"]
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--group-organs",
        action="store_true",
        help="merge 117 TS labels -> 4 흉부 organ group (num_classes 118->5); "
        "use a SEPARATE --out-dir (e.g. data/text2ct_mask_g4) so it never mixes with 118 masks",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.datalist:
        vols = _vols_from_datalist(args.datalist)
    else:
        vols = [
            os.path.basename(p).replace("_emb.nii.gz", "").replace(".nii.gz", "")
            for p in sorted(glob.glob(args.emb_glob))
        ]
    if args.limit:
        vols = vols[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scheme = (
        f"4-organ group (num_classes=5)"
        if args.group_organs
        else "full TS (num_classes=118)"
    )
    print(f"vols: {len(vols)} -> {out_dir}  | scheme: {scheme}")

    done = miss = 0
    global_max_distinct = 0
    total_truncated = 0
    for vol in tqdm(vols):
        out = out_dir / f"{vol}.pt"
        if out.is_file():
            done += 1
            continue
        mp = mask_path_for(vol, args.split)
        if not os.path.isfile(mp):
            miss += 1
            continue
        classes, fracs, max_distinct, n_truncated = mask_to_topk(
            mp, k=args.k, device=args.device, group_organs=args.group_organs
        )
        global_max_distinct = max(global_max_distinct, max_distinct)
        total_truncated += n_truncated
        tmp = out.with_suffix(".pt.tmp")
        torch.save({"classes": classes, "fracs": fracs}, tmp)
        os.replace(tmp, out)
        done += 1
    print(f"done/skip={done}  missing_mask={miss}")
    # NOTE: stats cover only newly-computed volumes (skipped/pre-existing ones are not re-scanned),
    # so run on a clean out-dir for a complete audit.
    verdict = "EXACT — no omission" if total_truncated == 0 else "TRUNCATION OCCURRED"
    print(
        f"[exactness audit] global max distinct classes/cell = {global_max_distinct} "
        f"(K={args.k}) | cells truncated (>K) = {total_truncated} | {verdict}"
    )


if __name__ == "__main__":
    main()
