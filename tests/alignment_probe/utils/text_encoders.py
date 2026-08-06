"""5개 인코더 → 단일 텍스트 임베딩 추출기 (cases → numpy array).

native 입력 텍스트는 templates가 단일 소스. 각 추출기는 upstream API를 그대로 호출:
  T5/GenerateCT  : t5_encode_text → (B,L,768) pad-zeroed → masked-mean → (B,768)
  Report2CT      : encode(findings)/encode(impression) 각 2560 → mean → 2560
  CT-CLIP        : tokenize → encode_text → (B,512) L2-norm
  Text2CT CLIP3D : clip.encode_text(list) → (B,1,768) L2-norm → squeeze
  fVLM           : encode_organ_texts → (4,256) → concat → 1024 (+per-organ 보존)

각 추출기 반환: (emb (N,d), extra)  — extra는 fVLM per-organ (N,4,256), 그 외 None.
CT-CLIP과 GenerateCT가 둘 다 `transformer_maskgit`를 (다른 시그니처로) 쓰므로 한 프로세스에
한 인코더만 빌드해야 안전(embed 스테이지가 인코더당 1프로세스로 호출).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterator

import torch

from tests.alignment_probe.utils import templates
from tests.alignment_probe.utils.cases import AlignmentCase

_GENERATECT_TM = Path("/workspace/third_party/generatect/transformer_maskgit")

# fVLM 분해(desc/conc JSON)가 사는 디렉토리 — split별로 다름.
_FVLM_DECOMP: dict[str, Path] = {
    "train": Path(
        "/workspace/data/checkpoints/fvlm/anatomy_descriptions/decomposed_report"
    ),  # 저자 릴리스
    "valid_v2": Path("/workspace/data/fVLM/decomposed_report_valid"),  # Qwen
}


def _batched(items: list, n: int) -> Iterator[list]:
    """리스트를 크기 n의 청크로 순서대로 yield하는 제너레이터."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _extract_t5(cases: list[AlignmentCase], device: str):
    """GenerateCT T5 인코더로 텍스트 임베딩을 추출한다.

    t5_encode_text → ``(B, L, 768)`` pad-zeroed → 비-pad 마스크 masked-mean → ``(B, 768)``.

    Returns:
        ``(emb, None)`` — emb shape ``(N, 768)`` float32 numpy array.
    """
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
    """Report2CT 텍스트 인코더로 findings·impression 각 ``(2560,)``을 mean → ``(2560,)``으로 합산한다.

    Returns:
        ``(emb, None)`` — emb shape ``(N, 2560)`` float32 numpy array.
    """
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
    """CT-CLIP 텍스트 인코더로 전체 report를 L2-norm 임베딩으로 추출한다.

    tokenize → encode_text → ``(B, 512)`` L2-norm.

    Returns:
        ``(emb, None)`` — emb shape ``(N, 512)`` float32 numpy array.
    """
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
    """Text2CT CLIP3D 인코더로 Finding+Impression 임베딩을 추출한다.

    clip.encode_text → ``(B, 1, 768)`` L2-norm → squeeze → ``(B, 768)``.

    Returns:
        ``(emb, None)`` — emb shape ``(N, 768)`` float32 numpy array.
    """
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
    """fVLM 텍스트 인코더로 per-organ 임베딩을 추출해 전역 concat 임베딩을 반환한다.

    encode_organ_texts → ``(4, 256)`` per-organ → reshape → ``(1024,)`` 전역 concat.
    분해 JSON(desc_info/conc_info)이 없는 scan이 있으면 SystemExit으로 즉시 실패.

    Returns:
        ``(glob, organs)`` — glob shape ``(N, 1024)``, organs shape ``(N, 4, 256)``, 모두 float32 numpy array.
    """
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
    # (잘못된 임베딩) — 누락이 있으면 decompose 먼저 하라고 에러
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
    "fvlm": _extract_fvlm,  # split을 추가로 받음 (embed 스테이지에서 분기)
}
