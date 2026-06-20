"""Unit 1b — 5개 인코더 → 단일 텍스트 임베딩 추출 + 디스크 캐시.

인코더당 **1프로세스**(CLI)로 실행한다: CT-CLIP과 GenerateCT가 둘 다 `transformer_maskgit`
패키지를 (서로 다른 `CTViT.forward` 시그니처로) 쓰기 때문에 한 프로세스에 섞으면
sys.path/sys.modules가 오염된다. 한 번에 한 인코더만 빌드하면 원천 차단된다.

사용:
    CUDA_VISIBLE_DEVICES=0 python -m tests.alignment_probe.text_embed \
        --encoder ctclip --split train

출력: tests/alignment_probe/embeddings/<split>/<encoder>.npz
      keys: emb (N, d), scan_ids (N,)  [fVLM은 emb_organs (N, 4, 256) 추가]
행 순서 = load_cases(split) 순서(= toy_v2 ids.json 순서) → 모든 인코더가 동일 정렬.

native 입력 텍스트는 templates.py가 단일 소스. 각 추출기는 upstream API를 그대로 호출:
  T5/GenerateCT  : t5_encode_text → (B,L,768) pad-zeroed → masked-mean → (B,768)
  Report2CT      : encode(findings)/encode(impression) 각 2560 → mean → 2560
  CT-CLIP        : tokenize → encode_text → (B,512) L2-norm
  Text2CT CLIP3D : clip.encode_text(list) → (B,1,768) L2-norm → squeeze
  fVLM           : encode_organ_texts → (4,256) → concat → 1024 (+per-organ 보존)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from tests.alignment_probe import templates
from tests.alignment_probe.cases import AlignmentCase, load_cases

_OUT_ROOT = Path("tests/alignment_probe/embeddings")
_GENERATECT_TM = Path("/workspace/third_party/generatect/transformer_maskgit")

# fVLM 분해(desc/conc JSON)가 사는 디렉토리 — split별로 다름.
_FVLM_DECOMP: dict[str, Path] = {
    "train": Path(
        "/workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report"
    ),  # 저자 릴리스
    "valid_v2": Path("/workspace/data/fVLM/decomposed_report_valid"),  # Qwen
}


def _batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --- per-encoder 추출기: cases → (emb, extra)  (extra: fVLM per-organ만, 그 외 None) ---


def _extract_t5(cases: list[AlignmentCase], device: str):
    if str(_GENERATECT_TM) not in sys.path:
        sys.path.insert(0, str(_GENERATECT_TM))
    from transformer_maskgit.t5 import t5_encode_text  # noqa: PLC0415

    out = []
    for chunk in _batched(cases, 32):
        texts = [templates.t5_text(c) for c in chunk]
        emb = t5_encode_text(texts)  # (B, L, 768), pad 위치는 0으로 채워짐
        mask = (emb.abs().sum(-1) > 0).float()  # (B, L) — 비-pad 토큰
        pooled = emb.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)  # (B, 768)
        out.append(pooled.cpu().float())
    return torch.cat(out).numpy(), None


def _extract_report2ct(cases: list[AlignmentCase], device: str):
    from src.baselines.report2ct_text_encoder import Report2CTTextEncoder  # noqa: PLC0415

    enc = Report2CTTextEncoder(device=device)
    out = []
    for c in cases:
        findings, impression = templates.report2ct_texts(c)
        vf = enc.encode(findings)  # (2560,)
        vi = enc.encode(impression)  # (2560,)
        out.append((vf + vi) / 2.0)  # findings+impression mean → (2560,)
    return torch.stack(out).numpy(), None


def _extract_ctclip(cases: list[AlignmentCase], device: str):
    from src.baselines.ctclip_adapter import CTCLIPBackbone  # noqa: PLC0415

    bb = CTCLIPBackbone(device_str=device, load_weights=True)
    out = []
    for chunk in _batched(cases, 16):
        texts = [templates.ctclip_text(c) for c in chunk]
        toks = bb.tokenize(texts)
        emb = bb.encode_text(toks["input_ids"], toks["attention_mask"])  # (B, 512)
        out.append(emb.cpu().float())
    return torch.cat(out).numpy(), None


def _extract_text2ct(cases: list[AlignmentCase], device: str):
    from src.baselines.text2ct_adapter import Text2CTAdapter  # noqa: PLC0415

    adapter = Text2CTAdapter(device_str=device, load_weights=True)
    adapter._ensure_built()  # AE+UNet+CLIP3D 빌드(1회). text는 _clip만 사용.
    clip = adapter._clip
    out = []
    for chunk in _batched(cases, 16):
        texts = [templates.text2ct_text(c) for c in chunk]
        emb = clip.encode_text(texts)  # (B, 1, 768) L2-norm
        out.append(emb.squeeze(1).detach().cpu().float())
    return torch.cat(out).numpy(), None


def _extract_fvlm(cases: list[AlignmentCase], device: str, split: str):
    from src.baselines.fvlm_adapter import FVLMBackbone  # noqa: PLC0415
    from src.data.fvlm_organ_report import (  # noqa: PLC0415
        load_decomposed_report,
        scan_id_from_volume_name,
    )

    decomp = _FVLM_DECOMP[split]
    desc, conc = load_decomposed_report(
        decomp / "desc_info.json", decomp / "conc_info.json"
    )

    # fail-fast: 분해에 없는 scan은 build_organ_text가 조용히 "정상"으로 채우므로
    # (잘못된 임베딩) — 누락이 있으면 분해를 먼저 끝내라고 에러.
    have = set(desc) & set(conc)
    missing = [
        c.scan_id
        for c in cases
        if scan_id_from_volume_name(f"{c.scan_id}.nii.gz") not in have
    ]
    if missing:
        raise SystemExit(
            f"fVLM decomposition incomplete for split={split!r}: "
            f"{len(missing)}/{len(cases)} scans missing (e.g. {missing[:3]}). "
            f"Finish {decomp}/desc_info.json first."
        )

    bb = FVLMBackbone(device_str=device, load_weights=True)
    organ_feats = []
    for c in cases:
        organ_text = templates.fvlm_organ_texts(c, desc, conc)  # {organ: text}
        feat = bb.encode_organ_texts(organ_text)  # (4, 256)
        organ_feats.append(feat.cpu().float())
    organs = torch.stack(organ_feats)  # (N, 4, 256)
    glob = organs.reshape(organs.shape[0], -1)  # (N, 1024) — 4-organ concat
    return glob.numpy(), organs.numpy()


_EXTRACTORS: dict[str, Callable] = {
    "t5": _extract_t5,
    "report2ct": _extract_report2ct,
    "ctclip": _extract_ctclip,
    "text2ct": _extract_text2ct,
    "fvlm": _extract_fvlm,  # split을 추가로 받음 (main에서 분기)
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, choices=sorted(_EXTRACTORS))
    p.add_argument("--split", required=True, choices=("train", "valid_v2"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=str(_OUT_ROOT))
    args = p.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(
            int(args.device.split(":")[-1]) if ":" in args.device else 0
        )

    cases = load_cases(args.split)
    print(
        f"[{args.encoder}/{args.split}] {len(cases)} cases — extracting...", flush=True
    )

    if args.encoder == "fvlm":
        emb, organs = _extract_fvlm(cases, args.device, args.split)
    else:
        emb, organs = _EXTRACTORS[args.encoder](cases, args.device)

    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_ids = np.array([c.scan_id for c in cases])
    payload = {"emb": emb, "scan_ids": scan_ids}
    if organs is not None:
        payload["emb_organs"] = organs
    out_path = out_dir / f"{args.encoder}.npz"
    np.savez(out_path, **payload)
    print(f"  saved {emb.shape} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
