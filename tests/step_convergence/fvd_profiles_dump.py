"""Dump the per-volume 18-d CT-CLIP profiles that FVD_CTCLIP is built from.

Question (user): FVD_CTCLIP has only ever been reported at a single n. How noisy is it,
and how does that noise fall with n? Answering needs the same bootstrap treatment FID got
(`fid_bootstrap_ncurve.py`), but FVD recomputes its features inline and throws them away
(`_fvd_ctclip.compute_fvd_ctclip`), so there is nothing to resample. This script runs the
encode pass ONCE and persists the profiles; the curve itself (`fvd_bootstrap_ncurve.py`)
is then pure numpy.

Faithfulness: file ordering (`_list_volumes`) and GT pairing (`map_to_gt`) are the upstream
docker functions, imported through `_fvd_ctclip._load_upstream()` — so row i here is the
same volume that would land in chunk i//4 of a production `compute_fvd_ctclip` call. That
lets the curve script reproduce the eval's published FVD exactly as a gate.

GT profiles are model-independent (they depend only on the case stem), so they are cached
in a shared npz and re-encoded only for stems not seen before — the second model onwards
pays roughly half the GPU time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/step_convergence/fvd_profiles_dump.py \
        --pred-dir outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg5/predictions \
        --tag wan_ep299_cfg5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")

from src.eval.tasks._fvd_ctclip import (  # noqa: E402
    TARGET_DHW,
    CTCLIPZeroShotProfiler,
    _load_upstream,
    _load_vol,
)

OUT = Path(__file__).parent / "_profiles"
GT_CACHE = OUT / "_gt_profiles.npz"
GT_3001 = "/workspace/data/vlm3d_eval/_valid_full_3001"


def _load_gt_cache() -> dict[str, np.ndarray]:
    """Stem → ``(18,)`` GT profile, from the shared cache (empty dict if absent)."""
    if not GT_CACHE.exists():
        return {}
    with np.load(GT_CACHE) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", default=GT_3001)
    ap.add_argument("--tag", required=True, help="names the output npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    _, efvd = _load_upstream()
    profiler = CTCLIPZeroShotProfiler(device_str=args.device)

    gen_files = efvd._list_volumes(Path(args.pred_dir))  # upstream order == chunk order
    gt_root = Path(args.gt_dir)
    gt_cache = _load_gt_cache()
    print(f"{len(gen_files)} volumes; {len(gt_cache)} GT profiles cached", flush=True)

    stems, gen_prof, gt_prof = [], [], []
    t0 = time.time()
    for i, g in enumerate(gen_files):
        stem = efvd._case_stem(g)
        r = efvd.map_to_gt(g, gt_root)
        vg = (_load_vol(g) / 1000.0).reshape(1, 1, *TARGET_DHW)  # HU/1000 ∈ [-1,1]
        gen_prof.append(profiler.profile(vg).numpy())  # (18,)
        if stem not in gt_cache:
            vr = (_load_vol(r) / 1000.0).reshape(1, 1, *TARGET_DHW)
            gt_cache[stem] = profiler.profile(vr).numpy()  # (18,)
        gt_prof.append(gt_cache[stem])
        stems.append(stem)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(
                f"  {i + 1}/{len(gen_files)}  {el:.0f}s  "
                f"({el / (i + 1):.2f}s/vol, eta {el / (i + 1) * (len(gen_files) - i - 1):.0f}s)",
                flush=True,
            )

    np.savez(
        OUT / f"{args.tag}.npz",
        gen=np.stack(gen_prof),  # (V, 18)
        gt=np.stack(gt_prof),  # (V, 18)
        stems=np.array(stems),  # (V,)
    )
    np.savez(GT_CACHE, **gt_cache)
    print(
        f"wrote {OUT / f'{args.tag}.npz'} ({len(stems)} vols, {time.time() - t0:.0f}s); "
        f"GT cache now {len(gt_cache)} stems",
        flush=True,
    )


if __name__ == "__main__":
    main()
