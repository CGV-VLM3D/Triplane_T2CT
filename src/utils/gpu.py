"""노트북/스크립트용 GPU 메모리 정리 헬퍼."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def free_gpu_after():
    """종료 시 `del` + `empty_cache`를 실행하는 컨텍스트 매니저 (노트북 메모리 정리용)."""
    try:
        yield
    finally:
        import gc  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def free_gpu_now(ns: dict | None = None, *, verbose: bool = True) -> None:
    """``ns`` (기본값: 호출자 globals)의 모든 torch.Tensor / nn.Module을 삭제하고,
    IPython 출력 캐시를 비운 뒤 모든 GPU에서 ``empty_cache`` + ``ipc_collect`` 실행.

    두 가지를 모두 하는 이유: ``torch.cuda.empty_cache()``는 *캐시*만 해제하며,
    여전히 참조되는 텐서(노트북 globals, ``Out[..]`` 히스토리, ``_``/``__``/``___``)는
    그대로 남아 있음. 먼저 참조를 끊어야 메모리 해제가 실질적으로 이루어짐.
    """
    import gc  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import torch.nn as nn  # noqa: PLC0415

    if ns is None:
        ns = sys._getframe(1).f_globals

    dropped: list[str] = []
    for name in list(ns.keys()):
        if name.startswith("_"):
            continue
        try:
            obj = ns[name]
        except Exception:
            continue
        if isinstance(obj, (torch.Tensor, nn.Module)):
            del ns[name]
            dropped.append(name)

    try:
        ip = sys.modules["IPython"].get_ipython()  # type: ignore[attr-defined]
        if ip is not None:
            ip.user_ns.get("Out", {}).clear()
            for k in ("_", "__", "___"):
                ip.user_ns[k] = None
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except RuntimeError:
                # 다른 프로세스가 디바이스를 완전히 점유 중 — 건너뜀
                continue

    if verbose:
        if torch.cuda.is_available():
            usage = ", ".join(
                f"cuda:{i}={torch.cuda.memory_allocated(i) / 1024**3:.2f} GiB"
                for i in range(torch.cuda.device_count())
            )
        else:
            usage = "no cuda"
        print(
            f"free_gpu_now: dropped {len(dropped)} var(s) {dropped} | allocated: {usage}"
        )
