#!/usr/bin/env python3
"""VLM3D ctgen challenge entrypoint: /input/prompts.json → /output/<case>.nii.gz.

Runs `report2ct_wan` end-to-end in ONE process. Locally the Wan flow is three steps (main env
RFlow → `wan` env VAE decode → scoring, see docs/wan_latent_runbook.md) only because the two
envs disagree on the diffusers version; the submission image ships a single env that can do
both, so there is no `.npy` hand-off here.

Contract (README §3.3 of third_party/vlm3d_dockers):
  /input/prompts.json  — JSON array of {"input_image_name", "report"}
  /output/<stem>.nii.gz — one loose volume per prompt, stem = input_image_name without extension

Deliberate differences from our research path (`scripts/decode_wan_latents.py`), both required by
the platform and confined to this file — the research pipeline keeps writing int16 .mha:
  * saves **float32 .nii.gz**, not int16 .mha. `forithmus validate` checks dtype and rejects
    int16 ("Expected float32 but got int16"); shape 512x512x253 and spacing 0.75/0.75/1.3 pass.
    Voxels are still rounded to integer HU first, so the submitted volume is numerically the
    same one we validated locally — only the container dtype differs.

Resumability (spot preemption / time-budget stop). The platform **clears `/output` between runs**
and restores only `/checkpoint`, so "skip what is already in /output" resumes nothing on its own —
see the `Checkpoint` class, which backs each finished volume up and copies it back on the next
start. Writes are NOT atomic: gcsfuse has no usable same-dir staging (the evaluator's
`_list_volumes` rglobs, so a staging subdir would be scored) and NiftiImageIO refuses a non-NIfTI
temp extension. A stem is recorded only after its backup completes, so a restore never replays a
half-written file; without a checkpoint, startup instead drops the single newest output, the only
one a mid-write kill could have truncated.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
import SimpleITK as sitk
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # report_parse, co-located in both layouts
if not (_HERE / "src").is_dir():
    # Local run from the repo: `src` is the parent's, not ours. In the image both sit in /opt/app.
    sys.path.insert(0, str(_HERE.parent))

from report_parse import report_markers, split_report  # noqa: E402
from src.baselines.wan_vae import WanVAE  # noqa: E402
from src.eval.samplers._orient import ras_to_lps  # noqa: E402
from src.eval.samplers.base import EvalCase  # noqa: E402
from src.eval.samplers.report2ct_wan import Report2CTWanLatentSampler  # noqa: E402

# --- fixed generation settings -------------------------------------------------------------
# The image takes no CLI arguments, so these constants ARE the run configuration. They are all
# logged at startup ("loud knob" convention, CLAUDE.md) — the eval-dir naming rule that normally
# guards them does not apply to a submission.
SPACING_MM = [
    0.75,
    0.75,
    1.3,
]  # generated at == stamped in the affine (spacing-conditional model)
CFG_SCALE = 5.0
N_STEPS = 50
MODALITY_CLASS_LABEL = 1  # CT, matches training (baseline-clone #8)

INPUT_DIR = Path(os.environ.get("FORITHMUS_INPUT", "/input"))
OUTPUT_DIR = Path(os.environ.get("FORITHMUS_OUTPUT", "/output"))
WEIGHTS_DIR = Path(os.environ.get("FORITHMUS_WEIGHTS", "/weights"))
CHECKPOINT_DIR = Path(os.environ.get("FORITHMUS_CHECKPOINT", "/checkpoint"))
CKPT_PATH = WEIGHTS_DIR / "report2ct_wan_ep299.ckpt"

BACKUP_DIR = CHECKPOINT_DIR / "outputs"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ctgen_submission")


def _on_sigterm(signum, frame):  # noqa: ARG001 — signal handler signature
    """Log how far we got, then exit. Preemption and time-budget both arrive as SIGTERM.

    Nothing has to be flushed here: ``Checkpoint.record`` rewrites ``progress.json`` after every
    volume, so the checkpoint is always current. Exiting raises SystemExit, and
    ``concurrent.futures``' atexit hook joins the writer thread — letting an in-flight
    write+backup finish inside the platform's 30 s grace window.

    Exit code 0 follows the platform's own checkpointing example. The status ("timed_out", with a
    Continue button) is set by the platform, which knows it sent the signal; a non-zero code risks
    being read as a genuine failure instead.
    """
    n = len(list(OUTPUT_DIR.glob("*.nii.gz"))) if OUTPUT_DIR.is_dir() else 0
    log.warning(
        "SIGTERM received — %d volume(s) in %s; flushing and exiting", n, OUTPUT_DIR
    )
    sys.exit(0)


def load_prompts(input_dir: Path) -> list[EvalCase]:
    """Read the prompts JSON and turn each entry into an EvalCase.

    ``input_image_name`` may or may not carry an extension (the real schema sample is
    ``"<uuid>.mha"``, `forithmus generate` emits a bare ``"case_001"``); ``Path(...).stem``
    handles both, matching the shipped baseline (ctgen_example_docker/process.py:189).
    """
    matches = sorted(input_dir.glob("*.json"))
    if not matches:
        raise SystemExit(f"No JSON file found in {input_dir}")
    log.info("Reading prompts from %s", matches[0])
    entries = json.loads(matches[0].read_text())
    if not isinstance(entries, list):
        raise SystemExit("prompts JSON must be a list of objects")

    cases, markers, f_len, i_len = [], Counter(), [], []
    for e in entries:
        findings, impression = split_report(e["report"])
        markers[report_markers(e["report"])] += 1
        f_len.append(len(findings))
        i_len.append(len(impression))
        cases.append(
            EvalCase(
                scan_id=Path(e["input_image_name"]).stem,
                findings=findings,
                impression=impression,
                spacing_mm=list(SPACING_MM),
            )
        )

    # Parse diagnostic — counts and lengths only, never prompt text. The format assumption comes
    # from a single sample in .forithmus/config.json and has never been checked against the real
    # set; if it is wrong, every volume is conditioned on a half-empty context.
    log.info(
        "prompts: %d | markers %s | findings median %d chars, impression median %d",
        len(cases),
        dict(markers),
        int(median(f_len)) if f_len else 0,
        int(median(i_len)) if i_len else 0,
    )
    if markers["both"] != len(cases):
        # Not fatal: a real set may legitimately carry a few odd reports, and aborting here would
        # also kill the mock dry-run (`forithmus generate` writes report="sample_report"). But
        # `findings_only` silently empties half the conditioning, so it must not be a mere debug
        # line — a partial run is worth killing by hand early rather than paying for in full.
        log.error(
            "%d/%d prompt(s) lack both Findings:/Impression: markers — conditioning is degraded "
            "for those; check the parser against the real prompt format",
            len(cases) - markers["both"],
            len(cases),
        )
    return cases


def drop_possibly_truncated(output_dir: Path) -> None:
    """Delete the newest output volume — the only one a mid-write kill could have truncated."""
    existing = sorted(output_dir.glob("*.nii.gz"), key=lambda p: p.stat().st_mtime)
    if existing:
        log.warning(
            "Dropping newest output %s (may be truncated); it will be regenerated",
            existing[-1].name,
        )
        existing[-1].unlink()


class Checkpoint:
    """Progress + output backup under ``/checkpoint``, the only dir that survives a restart.

    The platform clears ``/output`` between runs — on a spot preemption's auto-retry as much as on
    a time-budget "Continue" — and restores only ``/checkpoint``. So skipping on "the output file
    already exists" resumes nothing by itself; a completed volume has to be copied here and copied
    back on the next start.

    **Best-effort by design.** 3024 volumes × 87 MiB ≈ 257 GiB, and the platform documents
    ``/checkpoint`` as local disk that is uploaded to GCS at teardown — it may well be smaller than
    that. So a failed backup is logged loudly and the run continues: the volume is still written to
    ``/output`` and still scored, only its resumability is lost. Crashing at volume 2000 because the
    checkpoint disk filled would waste far more than it protects. A stem is recorded as processed
    **only after** its backup lands, so the restore never replays a partially-copied file.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.backup_dir = root / "outputs"
        self.progress_file = root / "progress.json"
        self.processed: list[str] = []
        self.enabled = False
        self._backup_failed = False

        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.enabled = True
        except OSError as e:
            log.warning(
                "Checkpoint dir %s unusable (%s) — resume will not be possible", root, e
            )
            return

        total, used, free = shutil.disk_usage(root)
        log.info(
            "Checkpoint %s: %.1f GiB free of %.1f GiB (need ~257 GiB to back up every volume)",
            root,
            free / 2**30,
            total / 2**30,
        )
        if self.progress_file.is_file():
            self.processed = json.loads(self.progress_file.read_text())["processed"]

    def restore(self, output_dir: Path) -> int:
        """Copy backed-up volumes back into ``/output``; returns how many were restored.

        Logs every 200 files: late in the run this copies hundreds of 87 MiB volumes back over the
        gcsfuse-mounted /output and can take tens of minutes before generation even starts. Still
        far cheaper than regenerating them, but it must not look hung.
        """
        if not self.processed:
            return 0
        log.info("Restoring %d volume(s) from checkpoint …", len(self.processed))
        n = 0
        for i, stem in enumerate(self.processed, 1):
            src, dst = self.backup_dir / f"{stem}.nii.gz", output_dir / f"{stem}.nii.gz"
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
                n += 1
            if i % 200 == 0:
                log.info("  restored %d/%d", i, len(self.processed))
        log.info(
            "Resuming from checkpoint: %d recorded, %d restored into %s",
            len(self.processed),
            n,
            output_dir,
        )
        return n

    def record(self, out_path: Path) -> None:
        """Back up one finished volume and mark it processed. Never raises."""
        if not self.enabled or self._backup_failed:
            return
        try:
            shutil.copy2(out_path, self.backup_dir / out_path.name)
            self.processed.append(out_path.name[: -len(".nii.gz")])
            tmp = self.progress_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"processed": self.processed}))
            tmp.replace(self.progress_file)  # atomic within /checkpoint
        except OSError as e:
            self._backup_failed = True
            _, _, free = shutil.disk_usage(self.root)
            log.error(
                "Checkpoint backup FAILED at %d volume(s) (%s; %.1f GiB free). Generation "
                "continues and every volume is still written to /output, but anything after "
                "this point will NOT survive a restart.",
                len(self.processed),
                e,
                free / 2**30,
            )


def save_volume(hu_zxy: np.ndarray, out_path: Path) -> None:
    """Save a decoded HU volume as float32 .nii.gz.

    Args:
        hu_zxy: decoded volume in HU, ``(Z, X, Y)`` as WanVAE.decode returns it.
        out_path: destination ``<stem>.nii.gz``.

    Mirrors ``scripts/decode_wan_latents.py::_save_mha`` — SimpleITK wants (Z, Y, X), and the Wan
    VAE decodes RAS content so the in-plane axes are flipped to LPS to match the real CT-RATE the
    hidden GT is drawn from. Only the final dtype/extension differ (see module docstring).
    """
    arr_zyx = np.ascontiguousarray(hu_zxy.transpose(0, 2, 1))  # (Z, Y, X)
    arr_zyx = ras_to_lps(arr_zyx)  # (Z, Y, X), X/Y reversed
    # Round to integer HU exactly as the research path does, then widen to the float32 the
    # platform schema requires — same voxel values, different container.
    img = sitk.GetImageFromArray(arr_zyx.astype(np.int16).astype(np.float32))
    img.SetSpacing([float(s) for s in SPACING_MM])
    sitk.WriteImage(img, str(out_path))


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("report2ct_wan ctgen submission")
    log.info(
        "  ckpt=%s  spacing=%s  cfg=%.1f  n_steps=%d  class=%d",
        CKPT_PATH,
        SPACING_MM,
        CFG_SCALE,
        N_STEPS,
        MODALITY_CLASS_LABEL,
    )
    log.info(
        "  input=%s  output=%s  cuda=%s",
        INPUT_DIR,
        OUTPUT_DIR,
        torch.cuda.is_available(),
    )

    cases = load_prompts(INPUT_DIR)
    ckpt = Checkpoint(CHECKPOINT_DIR)
    ckpt.restore(OUTPUT_DIR)
    if not ckpt.processed:
        # No checkpoint to trust: any file already in /output came from a kill we cannot reason
        # about, so drop the one that could be half-written. With a checkpoint the restored files
        # are known-complete and must not be thrown away.
        drop_possibly_truncated(OUTPUT_DIR)
    log.info(
        "%d prompt(s); %d already done",
        len(cases),
        len(list(OUTPUT_DIR.glob("*.nii.gz"))),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = Report2CTWanLatentSampler(
        ckpt_path=str(CKPT_PATH),
        n_steps=N_STEPS,
        modality_class_label=MODALITY_CLASS_LABEL,
        spacing_mm=list(SPACING_MM),
        cfg_scale=CFG_SCALE,
    )
    sampler._init(device)
    vae = WanVAE(device=str(device))

    # Saving is CPU-bound gzip (measured 7.8 s/volume, independent of GPU tier) and would
    # otherwise stall the GPU between volumes. One worker overlaps it with the next volume's
    # generation, so at most one write is ever in flight — which is what `drop_possibly_truncated`
    # already assumes. The checkpoint copy rides along in the same worker, keeping "write, then
    # back up, then record" strictly ordered. Write exceptions are re-raised, never swallowed
    # (fail loud); backup failures are not fatal — see Checkpoint.
    def _save_and_record(hu: np.ndarray, out_path: Path) -> None:
        save_volume(hu, out_path)
        ckpt.record(out_path)

    writer = ThreadPoolExecutor(max_workers=1)
    pending: list[Future] = []

    spacing_tensor = sampler._make_spacing_tensor(SPACING_MM)  # (1, 3)
    for n, case in enumerate(cases, 1):
        out_path = OUTPUT_DIR / f"{case.scan_id}.nii.gz"
        if out_path.exists():
            continue
        for f in [f for f in pending if f.done()]:
            f.result()  # surface a failed write immediately
            pending.remove(f)
        t0 = time.time()

        # Mirrors Report2CTWanLatentSampler.generate (report2ct_wan.py:269-284) minus the .npy
        # write: denoise, then re-lay the latent from the UNet's (C,H,W,D) to Wan's (C,T,H,W).
        context = sampler._case_to_context(case)  # (1, 2, 2560)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = sampler._denoise(context, spacing_tensor)  # (1, 16, 64, 64, 64)
        z_wan = z.squeeze(0).permute(0, 3, 1, 2).contiguous()  # (16, 64, 64, 64)

        hu = vae.decode(z_wan).numpy()  # (Z=253, X=512, Y=512) HU float32
        pending.append(writer.submit(_save_and_record, hu, out_path))
        log.info(
            "[%d/%d] %s  HU[%.0f, %.0f]  %.1fs (gpu)",
            n,
            len(cases),
            out_path.name,
            hu.min(),
            hu.max(),
            time.time() - t0,
        )

    writer.shutdown(wait=True)
    for f in pending:
        f.result()
    log.info(
        "Done — %d volume(s) in %s", len(list(OUTPUT_DIR.glob("*.nii.gz"))), OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
