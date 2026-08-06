"""embed 스테이지 — 한 인코더로 한 split의 텍스트 임베딩 추출 + 저장.

CT-CLIP과 GenerateCT가 둘 다 transformer_maskgit(다른 시그니처)을 쓰므로 한 프로세스에
한 인코더만 빌드해야 안전 → main을 인코더당 1회 호출(stage=embed encoder=X split=Y).
출력: embeddings/<split>/<encoder>.npz (emb (N,d), scan_ids; fVLM은 emb_organs (N,4,256) 추가)
"""

from __future__ import annotations

import numpy as np
import torch

from tests.alignment_probe.utils import io
from tests.alignment_probe.utils.cases import load_cases
from tests.alignment_probe.utils.text_encoders import _EXTRACTORS, _extract_fvlm


def run(cfg) -> None:
    """cfg.encoder로 cfg.split 전체를 인코딩해 npz 저장."""
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        idx = int(cfg.device.split(":")[-1]) if ":" in cfg.device else 0
        torch.cuda.set_device(idx)

    cases = load_cases(cfg.split)
    print(f"[{cfg.encoder}/{cfg.split}] {len(cases)} cases — extracting...", flush=True)

    if cfg.encoder == "fvlm":
        emb, organs = _extract_fvlm(cases, cfg.device, cfg.split)
    else:
        emb, organs = _EXTRACTORS[cfg.encoder](cases, cfg.device)

    payload = {"emb": emb, "scan_ids": np.array([c.scan_id for c in cases])}
    if organs is not None:
        payload["emb_organs"] = organs
    out_path = io.EMB_ROOT / cfg.split / f"{cfg.encoder}.npz"
    io.save_npz(out_path, **payload)
    print(f"  saved {emb.shape} → {out_path}", flush=True)
