"""Dump a forithmus submission's record + container logs.

The CLI (0.1.10) has no command for either: `forithmus status` reports *phase* data-upload
readiness, not submissions, and nothing fetches run logs. These endpoints do exist on the API, so
this is a read-only GET against our own submissions.

    python vlm3d_challenge_eval/inspect_submission.py [submission_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, "/opt/conda/lib/python3.11/site-packages")
from forithmus_cli.api import _headers  # noqa: E402
from forithmus_cli.auth import get_api_url  # noqa: E402

SLUG = "ct-volume-generation"
OUT = Path("/workspace/data/vlm3d_submission/run_logs.json")


def get(path: str):
    r = httpx.get(f"{get_api_url()}{path}", headers=_headers(), timeout=30)
    if r.status_code == 401:
        raise SystemExit("Not logged in — run: forithmus login")
    return r.json() if r.status_code == 200 else None


def main() -> None:
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    if sid is None:
        subs = get("/users/me/submissions") or []
        if not subs:
            raise SystemExit("No submissions found")
        print(f"=== 제출 {len(subs)}건 ===")
        for s in subs:
            print(
                f"  {s['id']}  status={s['status']} tier={s['tier']} "
                f"dur={s.get('duration_seconds')} cost_actual={s.get('cost_actual')}"
            )
        sid = subs[0]["id"]

    d = get(f"/challenges/{SLUG}/submissions/{sid}") or {}
    print(f"\n=== 상세 {sid} ===")
    for k, v in d.items():
        if k == "metrics" and isinstance(v, dict):
            for mk, mv in v.items():
                print(f"  metrics.{mk} = {json.dumps(mv)}")
        else:
            print(f"  {k:24s} = {str(v)[:140]}")

    budget, dur = d.get("time_budget_minutes"), d.get("duration_seconds")
    if budget and dur:
        print(
            f"\n  예산 {budget}분 vs 실행 {dur / 60:.1f}분 "
            f"→ {(dur / 60) / budget:.1f}x 초과 | cost_actual={d.get('cost_actual')}"
        )

    logs = get(f"/challenges/{SLUG}/submissions/{sid}/logs") or {}
    for stream in ("stdout", "stderr"):
        text = logs.get(stream, "")
        lines = text.splitlines()
        print(f"\n--- {stream}: {len(text)} chars, {len(lines)} lines ---")
        if lines:
            for ln in lines[:60]:
                print(f"  {ln}")
            if len(lines) > 60:
                print(f"  … (뒤 {len(lines) - 60}줄 생략, 전체는 {OUT})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"detail": d, "logs": logs}, indent=2))
    print(f"\n전체 저장: {OUT}")


if __name__ == "__main__":
    main()
