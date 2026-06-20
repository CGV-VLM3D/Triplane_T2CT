#!/usr/bin/env python3
"""Reproduce fVLM's anatomy-wise report decomposition with Qwen (paper §3.1).

The fVLM authors released the per-anatomy decomposition only for CT-RATE
**train** (`/workspace/data/fVLM/decomposed_report/{desc_info,conc_info}.json`).
This script regenerates the SAME two JSON artifacts for any other split — by
default the official `valid_fixed` set, which the authors did NOT release — so
the downstream loader `src.data.fvlm_organ_report.build_organ_text` is a drop-in for
either path.

Method (fVLM paper §3.1 + Appendix Fig 6/7), applied to the *Findings* and
*Impression* sections INDEPENDENTLY, per organ:
  Step 1 (Fig 6): ask the LLM whether the anatomy is mentioned in the section
    ("Yes"/"No"), OR a complementary substring match on sub-structure keywords
    (e.g. "aortic arch" -> aorta) — `mentioned = LLM-Yes or substring`.
  Step 2 (Fig 7): for each mentioned anatomy, ask the LLM to extract that
    anatomy's descriptive text from the section.
  Step 3 (post-process): write the Findings extractions into `desc_info` and the
    Impression extractions into `conc_info`, keyed by scan id. INVARIANT: do NOT
    bake the "{organ} shows no significant abnormalities." default or the
    findings+impression concatenation into the JSON — unmentioned organs simply
    have NO key (like the author files); the default/concat is synthesized at
    load time by `build_organ_text`, so valid_fixed uses the identical rule as
    train.

Model size is unspecified by the paper. We default to `Qwen/Qwen2.5-7B-Instruct`
in bf16 (≈15 GB, one 96 GB GPU, no quantization needed).
`scripts/calibrate_decompose_fvlm.py` lets you check 7B fidelity against the
author's train GT before committing compute.

Usage (GPU; GPU2 is full on this box — pin 0 or 1):
  CUDA_VISIBLE_DEVICES=0 python scripts/decompose_reports_fvlm.py --split valid \
      --out-dir /workspace/data/fVLM/decomposed_report_valid \
      --model-id Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --limit 16   # smoke
  # drop --limit for the full ~1564 scans.

The pure decomposition logic (`decompose_scan`, `mentioned`, prompts) imports
WITHOUT torch/transformers so it can be unit-tested with a mock backend; heavy
deps load lazily inside the backend classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

from src.data.fvlm_organ_report import ORGANS, scan_id_from_volume_name

# ---------------------------------------------------------------------------
# Prompts — VERBATIM from fVLM Appendix Fig 6 (mention) and Fig 7 (extraction).
# {section} = the Findings or Impression text; {anatomy} = one organ name.
# ---------------------------------------------------------------------------
MENTION_PROMPT = (
    "{section}\n\n"
    "You are a professional radiologist. Please determine if the anatomy "
    "({anatomy}) is mentioned in this CT image report. Please answer directly "
    'with "Yes" or "No".'
)

EXTRACT_PROMPT = (
    "{section}\n\n"
    "You are a professional radiologist. Please extract the descriptive "
    "information about the specific anatomy ({anatomy}) from this CT image "
    "diagnostic report. Please follow these guidelines:\n\n"
    "**Precise extraction**: Extract the descriptive information directly related "
    "to {anatomy} from the report.\n"
    "**Specify anatomical details**: If the report mentions specific areas, parts, "
    "or anatomical details of {anatomy}, make sure to include this information "
    "in the description. This could include affected areas, normal structures, "
    "or any notable features.\n"
    "**Concise and clear**: Directly extract the report content, avoiding "
    "unnecessary explanations or background.\n"
    "**Format requirement**: Please provide the information in the format "
    '"{anatomy}: descriptive information", ensuring to use {anatomy} as the '
    "unified prefix for the item. Even if the anatomy has multiple independent parts "
    "or multiple lateral characteristics, it should be treated and described as "
    "a whole, returning only one comprehensive piece of information about that "
    "anatomy."
)

# Complementary substring matching (paper §3.1): catch anatomies referenced only
# by sub-structure / lateral terms the LLM might miss. Recall-only OR with the
# LLM judgment. Lowercased substring test.
SUBSTRUCTURE_KEYWORDS: dict[str, list[str]] = {
    "lung": ["lung", "lobe", "bronch", "pulmonary", "parenchyma", "pleur"],
    "heart": ["heart", "myocard", "atri", "ventric", "pericard", "cardiac", "coronary"],
    "esophagus": ["esophag", "hiatal", "hiatus"],
    "aorta": ["aort"],
}


class QwenChat(Protocol):
    """Minimal chat-LLM contract shared by both backends (and test mocks)."""

    def generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]: ...


# ---------------------------------------------------------------------------
# Pure decomposition logic (no torch) — unit-testable with a mock backend.
# ---------------------------------------------------------------------------


def mention_prompt(section_text: str, anatomy: str) -> str:
    return MENTION_PROMPT.format(section=section_text, anatomy=anatomy)


def extract_prompt(section_text: str, anatomy: str) -> str:
    return EXTRACT_PROMPT.format(section=section_text, anatomy=anatomy)


def substring_mentioned(section_text: str, anatomy: str) -> bool:
    low = section_text.lower()
    return any(kw in low for kw in SUBSTRUCTURE_KEYWORDS.get(anatomy, [anatomy]))


def parse_yes_no(response: str) -> bool:
    """Greedy decoding yields 'Yes'/'Yes.'/'No' — treat a leading 'yes' as True."""
    return response.strip().lower().startswith("yes")


def normalize_extraction(text: str, anatomy: str) -> str:
    """Strip the leading '{anatomy}:' echo Fig 7 asks the model to emit.

    The author's stored values carry no prefix (lung text begins e.g. "Linear
    atelectasis ..."). Only strips on an explicit ':' delimiter so a genuine
    sentence starting with the organ word ("Lung parenchyma ...") is preserved.
    """
    t = text.strip()
    prefix = anatomy.lower() + ":"
    if t.lower().startswith(prefix):
        t = t[len(prefix) :].lstrip()
    return t


def mentioned(
    section_text: str, anatomy: str, backend: QwenChat, max_new_tokens: int = 8
) -> bool:
    """Step 1 for one (section, anatomy): substring OR LLM 'Yes'."""
    if substring_mentioned(section_text, anatomy):
        return True
    resp = backend.generate(
        [mention_prompt(section_text, anatomy)], max_new_tokens=max_new_tokens
    )[0]
    return parse_yes_no(resp)


def section_mentioned_organs(
    section_text: str, organs: tuple[str, ...], backend: QwenChat
) -> list[str]:
    """Step 1 over all organs (LLM mention calls batched for the ambiguous ones)."""
    section_text = section_text.strip()
    if not section_text:
        return []
    decided = {o: substring_mentioned(section_text, o) for o in organs}
    need_llm = [o for o in organs if not decided[o]]
    if need_llm:
        responses = backend.generate(
            [mention_prompt(section_text, o) for o in need_llm], max_new_tokens=8
        )
        for organ, resp in zip(need_llm, responses):
            decided[organ] = parse_yes_no(resp)
    return [o for o in organs if decided[o]]


def section_extractions(
    section_text: str,
    mentioned_organs: list[str],
    backend: QwenChat,
    max_new_tokens: int,
) -> dict[str, str]:
    """Step 2: extract per-organ text for the mentioned organs (batched)."""
    if not mentioned_organs:
        return {}
    responses = backend.generate(
        [extract_prompt(section_text, o) for o in mentioned_organs],
        max_new_tokens=max_new_tokens,
    )
    out = {
        organ: normalize_extraction(resp, organ)
        for organ, resp in zip(mentioned_organs, responses)
    }
    # Drop empties so build_organ_text never sees a lone "." (treat empty
    # extraction as not-mentioned).
    return {o: t for o, t in out.items() if t.strip()}


def decompose_section(
    section_text: str,
    organs: tuple[str, ...],
    backend: QwenChat,
    max_new_tokens: int = 256,
) -> dict[str, str]:
    """Steps 1+2 for one section -> {organ: extracted_text} for mentioned organs."""
    section_text = (section_text or "").strip()
    if not section_text:
        return {}
    mentioned_organs = section_mentioned_organs(section_text, organs, backend)
    return section_extractions(section_text, mentioned_organs, backend, max_new_tokens)


def decompose_scan(
    findings: str,
    impression: str,
    backend: QwenChat,
    organs: tuple[str, ...] = ORGANS,
    max_new_tokens: int = 256,
) -> tuple[dict[str, str], dict[str, str]]:
    """Decompose one scan -> (findings_organs, impression_organs).

    The two dicts go straight into desc_info[scan] (minus the "Findings" key)
    and conc_info[scan] (minus the "Conclusion" key) respectively.
    """
    findings_organs = decompose_section(findings, organs, backend, max_new_tokens)
    impression_organs = decompose_section(impression, organs, backend, max_new_tokens)
    return findings_organs, impression_organs


# ---------------------------------------------------------------------------
# Backends (heavy imports are lazy — kept out of module import for tests).
# ---------------------------------------------------------------------------


class TransformersBackend:
    """HF transformers greedy chat backend (the default; no extra install)."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # decoder-only batched generation

        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        torch = self._torch
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy / deterministic
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]  # strip the prompt tokens
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class VLLMBackend:
    """Optional vLLM backend (faster, multi-GPU). Requires `pip install vllm`."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        tp: int = 1,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "vLLM backend requested but `vllm` is not installed. "
                "Run `pip install vllm`, or use --backend transformers."
            ) from exc
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_id,
            dtype="auto" if dtype == "auto" else dtype,
            tensor_parallel_size=tp,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        sp = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        outs = self.llm.generate(texts, sp)
        return [o.outputs[0].text for o in outs]


def build_backend(args: argparse.Namespace) -> QwenChat:
    if args.backend == "transformers":
        return TransformersBackend(args.model_id, args.dtype, args.device)
    if args.backend == "vllm":
        return VLLMBackend(args.model_id, args.dtype, args.tp)
    raise ValueError(f"unknown backend {args.backend!r}")


# ---------------------------------------------------------------------------
# IO + driver
# ---------------------------------------------------------------------------

_DEFAULT_CSV = {
    "valid": "/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    "train": "/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv",
}


def load_scan_reports(reports_csv: str | Path) -> dict[str, tuple[str, str]]:
    """CT-RATE reports CSV -> {scan_id: (findings, impression)}, scan-level dedup.

    validation_reports.csv has per-reconstruction rows (valid_1_a_1, valid_1_a_2)
    that share identical text; collapse to one row per scan-study id.
    """
    import pandas as pd  # noqa: PLC0415 (lazy — keep module import light)

    df = pd.read_csv(reports_csv)
    # NOTE: a leading-underscore column name gets renamed to a positional field
    # by df.itertuples(); use a plain identifier ("scan_id") so row.scan_id works.
    df["scan_id"] = df["VolumeName"].map(scan_id_from_volume_name)
    df = df.drop_duplicates("scan_id", keep="first")
    out: dict[str, tuple[str, str]] = {}
    for row in df.itertuples():
        findings = str(getattr(row, "Findings_EN", "") or "").strip()
        impression = str(getattr(row, "Impressions_EN", "") or "").strip()
        out[row.scan_id] = (findings, impression)
    return out


def _prompts_sha() -> str:
    return hashlib.sha256(
        (MENTION_PROMPT + "\x00" + EXTRACT_PROMPT).encode()
    ).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen anatomy-wise report decomposition (fVLM §3.1)"
    )
    p.add_argument("--split", choices=["valid", "train", "custom"], default="valid")
    p.add_argument("--reports-csv", default=None, help="defaults from --split")
    p.add_argument("--out-dir", default="/workspace/data/fVLM/decomposed_report_valid")
    p.add_argument(
        "--backend", choices=["transformers", "vllm"], default="transformers"
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--dtype", choices=["bfloat16", "float16", "auto"], default="bfloat16"
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--device", default="cuda:0", help="GPU2 is full on this box — use 0 or 1."
    )
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    p.add_argument(
        "--limit", type=int, default=None, help="process at most N scans (smoke)"
    )
    p.add_argument(
        "--scan-ids-file", default=None, help="restrict to scan ids in this file"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="if set with --limit, random sample"
    )
    p.add_argument("--checkpoint-every", type=int, default=100)
    return p


def _write_outputs(out_dir: Path, desc_info: dict, conc_info: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "desc_info.json").write_text(json.dumps(desc_info, ensure_ascii=False))
    (out_dir / "conc_info.json").write_text(json.dumps(conc_info, ensure_ascii=False))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    reports_csv = args.reports_csv or _DEFAULT_CSV[args.split]
    out_dir = Path(args.out_dir)

    reports = load_scan_reports(reports_csv)
    all_scans = list(reports)

    # Resume: reuse anything already decomposed in out_dir.
    desc_info: dict = {}
    conc_info: dict = {}
    if (out_dir / "desc_info.json").is_file() and (
        out_dir / "conc_info.json"
    ).is_file():
        desc_info = json.loads((out_dir / "desc_info.json").read_text())
        conc_info = json.loads((out_dir / "conc_info.json").read_text())
    done = set(desc_info) & set(conc_info)

    todo = [s for s in all_scans if s not in done]
    if args.scan_ids_file:
        wanted = {
            line.strip()
            for line in Path(args.scan_ids_file).read_text().splitlines()
            if line.strip()
        }
        todo = [s for s in todo if s in wanted]
    if args.limit:
        if args.seed is not None:
            import random  # noqa: PLC0415

            random.Random(args.seed).shuffle(todo)
        todo = todo[: args.limit]

    print(
        f"split={args.split} csv={reports_csv}\n"
        f"scans: total={len(all_scans)} already_done={len(done)} todo={len(todo)}\n"
        f"backend={args.backend} model={args.model_id} dtype={args.dtype}"
    )
    if not todo:
        print("Nothing to do.")
        return

    from tqdm import tqdm  # noqa: PLC0415

    backend = build_backend(args)

    for i, scan_id in enumerate(tqdm(todo, desc="decompose")):
        findings, impression = reports[scan_id]
        findings_organs, impression_organs = decompose_scan(
            findings, impression, backend, ORGANS, args.max_new_tokens
        )
        desc_info[scan_id] = {"Findings": findings, **findings_organs}
        conc_info[scan_id] = {"Conclusion": impression, **impression_organs}
        if (i + 1) % args.checkpoint_every == 0:
            _write_outputs(
                out_dir, desc_info, conc_info, _meta(args, reports_csv, desc_info)
            )

    _write_outputs(out_dir, desc_info, conc_info, _meta(args, reports_csv, desc_info))
    print(f"Done. wrote {len(desc_info)} scans to {out_dir}/")


def _meta(args: argparse.Namespace, reports_csv: str, desc_info: dict) -> dict:
    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "dtype": args.dtype,
        "split": args.split,
        "reports_csv": str(reports_csv),
        "seed": args.seed,
        "n_scans": len(desc_info),
        "organs": list(ORGANS),
        "prompts_sha": _prompts_sha(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    main()
