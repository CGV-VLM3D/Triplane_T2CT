#!/usr/bin/env python3
"""Verify empirically that GT and every prediction arm carry the SAME (LPS) orientation.

Why this cannot be assumed: report2ct decodes RAS content and `_save_mha` flips it to LPS
in-sampler, while other arms in this repo were flipped after the fact and some were never
flipped at all. The eval readers compare raw voxel arrays and ignore ITK direction metadata,
so a silent RAS/LPS mismatch changes CLIPScore_I2I by ~25 points and FID by ~0.5 without any
error being raised. The saved direction matrix is identity everywhere, so it proves nothing —
orientation has to be read off the anatomy.

Two independent anatomical markers, both computed on raw voxel arrays indexed (Z, Y, X):

  left-right (X): the liver is a large dense mass on the patient's RIGHT, while the
      stomach/spleen side carries gas. In LPS, +X points to the patient's LEFT, so the
      patient's right sits at LOW x. Statistic: mean HU(low x) - mean HU(high x) over body
      voxels in the lower third of the volume. **LPS ⇒ positive.**

  anterior-posterior (Y): the spine is posterior and bony. In LPS, +Y points POSTERIOR, so
      the spine sits at HIGH y. Statistic: bone fraction(high y) - bone fraction(low y).
      **LPS ⇒ positive.**

A sign that matches GT means the arm agrees with GT; a flipped sign means that arm is RAS and
its metrics are being computed against a mirrored reference.

Usage:
    python tests/spacing_fov/orientation_check.py --n 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/workspace")

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"

ARMS = {
    "real (GT)": pathlib.Path("/workspace/data/vlm3d_eval/_lps_reeval/gt_lps_seed100"),
    "gen@1.0": HERE / "preds" / "gen_sp1.0",
    "gen@0.8": HERE / "preds" / "gen_sp0.8",
}

BODY_HU = -500.0
BONE_HU = 200.0


def markers(path: pathlib.Path) -> dict[str, float]:
    """Left-right and anterior-posterior orientation statistics for one volume.

    Args:
        path: volume file; read as ``(Z, Y, X)``.

    Returns:
        ``{"lr_liver": float, "ap_spine": float}`` — both positive for LPS content.
    """
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
        np.float32
    )  # (Z,Y,X)
    nz = arr.shape[0]

    # --- left-right: liver vs stomach side, on the lower third (upper abdomen) ---
    sub = arr[: max(1, nz // 3)]  # (Z/3, Y, X) — inferior slices
    body = sub > BODY_HU
    mid_x = sub.shape[2] // 2
    right_half = body[:, :, :mid_x]  # low x  = patient RIGHT if LPS
    left_half = body[:, :, mid_x:]  # high x = patient LEFT  if LPS
    lr = (
        float(
            (sub[:, :, :mid_x][right_half]).mean()
            - (sub[:, :, mid_x:][left_half]).mean()
        )
        if right_half.any() and left_half.any()
        else float("nan")
    )

    # --- anterior-posterior: where the bone is, on the mid third (thorax) ---
    mid = arr[nz // 3 : 2 * nz // 3]  # (Z/3, Y, X)
    bone = mid > BONE_HU
    mid_y = mid.shape[1] // 2
    ap = float(bone[:, mid_y:, :].mean() - bone[:, :mid_y, :].mean())  # high y − low y

    return {"lr_liver": lr, "ap_spine": ap}


def _control(gt_dir: pathlib.Path, cases: list[str]) -> dict:
    """Prove the two statistics can actually detect a flip before trusting their verdict.

    Flips GT in-plane (``[:, ::-1, ::-1]`` — exactly the LPS↔RAS transform in
    ``src/eval/samplers/_orient.py``) and checks that both markers invert. If they do not,
    the markers are not measuring orientation and every verdict below is meaningless.
    """
    import tempfile  # noqa: PLC0415

    tmp = pathlib.Path(tempfile.mkdtemp())
    inverted_lr, inverted_ap = [], []
    for c in cases:
        img = sitk.ReadImage(str(gt_dir / f"{c}.mha"))
        arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        flipped = sitk.GetImageFromArray(arr[:, ::-1, ::-1].copy())
        flipped.SetSpacing(img.GetSpacing())
        path = tmp / f"{c}.mha"
        sitk.WriteImage(flipped, str(path))
        before, after = markers(gt_dir / f"{c}.mha"), markers(path)
        inverted_lr.append(np.sign(before["lr_liver"]) != np.sign(after["lr_liver"]))
        inverted_ap.append(np.sign(before["ap_spine"]) != np.sign(after["ap_spine"]))
        path.unlink()
    tmp.rmdir()
    passed = bool(all(inverted_lr) and all(inverted_ap))
    print(
        f"[orient] control (flip GT in-plane): lr inverts {sum(inverted_lr)}/{len(cases)}, "
        f"ap inverts {sum(inverted_ap)}/{len(cases)} → {'PASS' if passed else 'FAIL'}\n"
    )
    return {"passed": passed, "n": len(cases)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument(
        "--control-n", type=int, default=4, help="cases used for the flip control"
    )
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    common = None
    for d in ARMS.values():
        ids = {p.stem for p in d.glob("*.mha")}
        common = ids if common is None else (common & ids)
    cases = sorted(common or [])[: args.n]
    if not cases:
        sys.exit("no case present in every arm yet")
    print(f"[orient] {len(cases)} cases\n")
    control = _control(ARMS["real (GT)"], cases[: args.control_n])

    out: dict[str, dict] = {}
    for arm, d in ARMS.items():
        lr = [markers(d / f"{c}.mha") for c in cases]
        lr_v = np.array([m["lr_liver"] for m in lr])
        ap_v = np.array([m["ap_spine"] for m in lr])
        out[arm] = {
            "lr_liver_mean": float(np.nanmean(lr_v)),
            "lr_liver_positive_frac": float(np.nanmean(lr_v > 0)),
            "ap_spine_mean": float(np.nanmean(ap_v)),
            "ap_spine_positive_frac": float(np.nanmean(ap_v > 0)),
        }
        print(
            f"{arm:12s} liver L-R {out[arm]['lr_liver_mean']:+8.2f} HU "
            f"({out[arm]['lr_liver_positive_frac'] * 100:3.0f}% positive)   "
            f"spine A-P {out[arm]['ap_spine_mean']:+.4f} "
            f"({out[arm]['ap_spine_positive_frac'] * 100:3.0f}% positive)"
        )

    # The spine marker is the strong one (well separated, unanimous in practice); the liver
    # marker is noisier — a low-contrast abdomen can flip a single case. Report the per-case
    # agreement rate rather than hiding it behind a binary label.
    ref = out["real (GT)"]
    verdict = {}
    for arm, v in out.items():
        same_lr = np.sign(v["lr_liver_mean"]) == np.sign(ref["lr_liver_mean"])
        same_ap = np.sign(v["ap_spine_mean"]) == np.sign(ref["ap_spine_mean"])
        if not (same_lr and same_ap):
            verdict[arm] = "MISMATCH vs GT"
        elif v["lr_liver_positive_frac"] == 1.0 and v["ap_spine_positive_frac"] == 1.0:
            verdict[arm] = "matches GT (unanimous per case)"
        else:
            verdict[arm] = (
                f"matches GT (mean sign; per-case liver "
                f"{v['lr_liver_positive_frac'] * 100:.0f}%, spine "
                f"{v['ap_spine_positive_frac'] * 100:.0f}%)"
            )
    print()
    for arm, v in verdict.items():
        print(f"  {arm:12s} {v}")

    if not control["passed"]:
        print(
            "\n!! control FAILED — the markers do not detect a flip; verdicts above are void"
        )
    (RESULTS / "orientation_check.json").write_text(
        json.dumps(
            {
                "n_cases": len(cases),
                "control": control,
                "stats": out,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n[orient] → {RESULTS / 'orientation_check.json'}")


if __name__ == "__main__":
    main()
