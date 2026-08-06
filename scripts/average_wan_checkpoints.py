"""Average N report2ct_wan Lightning checkpoints' state_dicts into one checkpoint.

Classic checkpoint/weight averaging over a training plateau (uniform mean) — a cheap
standalone proxy for the online EMA this run never tracked (see NVlabs/edm2 post-hoc EMA
discussion; our raw per-epoch snapshots support this simpler averaging, not the paper's
full dual-profile reconstruction).

Only writes ``{"state_dict": <averaged>}`` — src/eval/samplers/report2ct_wan.py's
``_load_wan_checkpoint`` only ever reads ``ckpt["state_dict"]`` (scale_factor + unet.*
keys), never hyper_parameters/optimizer_states/etc., so nothing else is needed for eval.

Usage:
    python scripts/average_wan_checkpoints.py \\
        --ckpt-dir outputs/report2ct_wan/2026-07-16_3/checkpoints \\
        --epochs 189 199 209 219 229 239 249 259 269 279 289 299 \\
        --out outputs/report2ct_wan/derived_checkpoints/avg_ep189-299.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt-dir", required=True, help="dir containing epoch_NNN.ckpt files"
    )
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [Path(args.ckpt_dir) / f"epoch_{ep:03d}.ckpt" for ep in args.epochs]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoint(s): {missing}")

    ref_keys: set[str] | None = None
    summed: dict[str, torch.Tensor] = {}
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        if ref_keys is None:
            ref_keys = set(sd.keys())
            summed = {k: v.double() for k, v in sd.items()}
        else:
            if set(sd.keys()) != ref_keys:
                raise SystemExit(f"state_dict key mismatch in {p}")
            for k, v in sd.items():
                summed[k] += v.double()
        print(f"accumulated {p.name}")

    n = len(paths)
    averaged = {k: (v / n).to(torch.float32) for k, v in summed.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": averaged}, out_path)
    print(f"wrote {out_path} (averaged {n} checkpoints: {args.epochs})")


if __name__ == "__main__":
    main()
