"""Extract the EMA weights tracked by ``src/callbacks/ema.py`` out of a Lightning checkpoint.

``EMACallback`` keeps its shadow in Lightning's callback-state slot
(``ckpt["callbacks"]["EMACallback"]["shadow"]``) so a resumed run continues the same average.
That means every snapshot a run saves already carries its own EMA — this script just turns
one into a standalone checkpoint the eval samplers load unchanged.

Only writes ``{"state_dict": <ema>}``, same as ``scripts/average_wan_checkpoints.py``:
``src/eval/samplers/report2ct.py``'s ``_load_checkpoint`` (and the wan variant) reads nothing
but ``ckpt["state_dict"]``. Parameters come from the shadow; **buffers come from the raw
state_dict** — ``scale_factor`` is a data statistic, not something to average (see the
``EMACallback`` docstring).

Usage:
    python scripts/extract_ema.py \\
        --ckpt outputs/report2ct_wan/<run>/checkpoints/epoch_299.ckpt \\
        --out outputs/report2ct_wan/derived_checkpoints/ema_ep299.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt", required=True, help="Lightning .ckpt saved with EMACallback"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    shadow = ckpt.get("callbacks", {}).get("EMACallback", {}).get("shadow")
    if shadow is None:
        raise SystemExit(
            f"no EMA state in {args.ckpt} — was the run trained with callbacks/ema.yaml?"
        )

    state = {k: v.detach().cpu() for k, v in ckpt["state_dict"].items()}
    unknown = set(shadow) - set(state)
    if unknown:
        raise SystemExit(f"EMA keys absent from state_dict: {sorted(unknown)}")
    state.update({k: v.detach().cpu() for k, v in shadow.items()})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state}, out_path)
    print(
        f"wrote {out_path} — {len(shadow)} EMA tensors "
        f"({len(state) - len(shadow)} buffers copied from raw)"
    )


if __name__ == "__main__":
    main()
