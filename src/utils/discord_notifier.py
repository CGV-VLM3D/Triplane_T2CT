"""Lightweight Discord webhook notifier for training events.

Reads `DISCORD_WEBHOOK_URL` from the environment (or `.env.local`).
All sends are best-effort: a network failure must never crash training.
Posts run on a background thread so they don't block the training loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def _load_env_local() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    # Search order: walk up from this file (covers the current repo / worktree),
    # then fall back to /workspace/.env.local (the main repo) since worktrees
    # don't inherit gitignored files.
    here = Path(__file__).resolve()
    candidates = [parent / ".env.local" for parent in [here.parent, *here.parents]]
    candidates.append(Path("/workspace/.env.local"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Don't clobber values already set in the real environment.
            os.environ.setdefault(k, v)
        break


def _webhook_url() -> Optional[str]:
    _load_env_local()
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    return url or None


# Discord embed color codes
COLOR_BLUE = 0x3498DB  # info / start
COLOR_GREEN = 0x2ECC71  # success / new best
COLOR_YELLOW = 0xF1C40F  # warning
COLOR_RED = 0xE74C3C  # failure / NaN
COLOR_GREY = 0x95A5A6  # end-of-epoch summary


def _is_main_process() -> bool:
    """True on rank 0 (or when not running under torchrun)."""
    rank = os.environ.get("RANK", "0")
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _post(payload: dict, timeout: float = 5.0) -> None:
    """Blocking POST. Swallows all errors — never raises."""
    if not _is_main_process():
        return
    url = _webhook_url()
    if not url:
        return
    data = json.dumps(payload).encode("utf-8")
    # Discord is fronted by Cloudflare, which 403s the default Python-urllib
    # User-Agent (CF error 1010). Discord's docs also require a descriptive UA.
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "triplane-ae-trainer/1.0 (+https://github.com/CGV-VLM3D)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        # Best-effort: never crash training because Discord is unreachable.
        pass


def _post_async(payload: dict) -> None:
    threading.Thread(target=_post, args=(payload,), daemon=True).start()


def notify(
    title: str,
    description: str = "",
    *,
    color: int = COLOR_BLUE,
    fields: Optional[list[dict]] = None,
    url: Optional[str] = None,
    blocking: bool = False,
) -> None:
    """Send an embed message. `fields` is a list of {name, value, inline?}.

    If `blocking=True`, waits for the HTTP request to complete (use for
    final messages so the process doesn't exit before the send finishes).
    """
    embed: dict = {
        "title": title[:256],
        "description": description[:4096],
        "color": color,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if fields:
        embed["fields"] = [
            {
                "name": str(f["name"])[:256],
                "value": str(f["value"])[:1024],
                "inline": bool(f.get("inline", True)),
            }
            for f in fields[:25]
        ]
    if url:
        embed["url"] = url
    payload = {"embeds": [embed]}
    if blocking:
        _post(payload)
    else:
        _post_async(payload)


# ---------------------------------------------------------------------------
# Convenience helpers for the training script
# ---------------------------------------------------------------------------


def notify_training_start(
    *,
    exp_name: str,
    model_kind: str,
    batch_size: int,
    epochs: int,
    n_batches_per_epoch: int,
    wandb_url: Optional[str] = None,
    extras: Optional[dict] = None,
) -> None:
    fields = [
        {"name": "model", "value": model_kind, "inline": True},
        {"name": "batch_size", "value": str(batch_size), "inline": True},
        {"name": "epochs", "value": str(epochs), "inline": True},
        {"name": "batches/epoch", "value": str(n_batches_per_epoch), "inline": True},
    ]
    if extras:
        for k, v in extras.items():
            fields.append({"name": str(k), "value": str(v), "inline": True})
    notify(
        title=f"🚀 Training started — {exp_name}",
        description=wandb_url or "",
        color=COLOR_BLUE,
        fields=fields,
        url=wandb_url,
    )


def notify_new_best(
    *,
    exp_name: str,
    metric_name: str,
    metric_value: float,
    prev_value: Optional[float],
    epoch: int,
    wandb_url: Optional[str] = None,
) -> None:
    prev_str = (
        f"{prev_value:.4f}"
        if prev_value is not None and prev_value != float("-inf")
        else "n/a"
    )
    notify(
        title=f"🏆 New best — {exp_name}",
        description=f"epoch {epoch}: **{metric_name} = {metric_value:.4f}** (prev {prev_str})",
        color=COLOR_GREEN,
        url=wandb_url,
    )


def notify_epoch_summary(
    *,
    exp_name: str,
    epoch: int,
    total_epochs: int,
    metrics: dict,
    wandb_url: Optional[str] = None,
) -> None:
    """Lightweight per-epoch progress ping. Caller decides how often to send."""
    fields = [
        {
            "name": k,
            "value": f"{v:.4f}" if isinstance(v, float) else str(v),
            "inline": True,
        }
        for k, v in metrics.items()
    ]
    notify(
        title=f"📊 epoch {epoch}/{total_epochs} — {exp_name}",
        color=COLOR_GREY,
        fields=fields,
        url=wandb_url,
    )


def notify_nan(
    *,
    exp_name: str,
    epoch: int,
    step: int,
    loss_value: float,
    wandb_url: Optional[str] = None,
) -> None:
    notify(
        title=f"⚠️ NaN/Inf loss — {exp_name}",
        description=(
            f"epoch {epoch} step {step}: loss = {loss_value!r}. Training stopping."
        ),
        color=COLOR_YELLOW,
        url=wandb_url,
    )


def notify_training_done(
    *,
    exp_name: str,
    final_metrics: dict,
    best_metric_name: Optional[str] = None,
    best_metric_value: Optional[float] = None,
    wandb_url: Optional[str] = None,
) -> None:
    fields = [
        {
            "name": k,
            "value": f"{v:.4f}" if isinstance(v, float) else str(v),
            "inline": True,
        }
        for k, v in final_metrics.items()
    ]
    if best_metric_name is not None and best_metric_value is not None:
        fields.append(
            {
                "name": f"best {best_metric_name}",
                "value": f"{best_metric_value:.4f}",
                "inline": True,
            }
        )
    notify(
        title=f"✅ Training finished — {exp_name}",
        description=wandb_url or "",
        color=COLOR_GREEN,
        fields=fields,
        url=wandb_url,
        blocking=True,
    )


def notify_exception(
    *,
    exp_name: str,
    exc: BaseException,
    wandb_url: Optional[str] = None,
) -> None:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_tail = "".join(tb_lines)[-1500:]  # last ~1.5k chars
    notify(
        title=f"❌ Training crashed — {exp_name}",
        description=f"`{type(exc).__name__}: {exc}`\n```\n{tb_tail}\n```",
        color=COLOR_RED,
        url=wandb_url,
        blocking=True,
    )
