# CT-CLIP Availability (R5 Day 1 check)

**Status (2026-05-26 Day 1)**: ✅ **AVAILABLE**, openly downloadable.

## Source

- Repo: `https://github.com/ibrahimethemhamamci/CT-CLIP` (397★, public)
- Weights: hosted on HuggingFace at `huggingface.co/datasets/ibrahimhamamci/CT-RATE`
- License: **CC-BY-NC-SA** (non-commercial research) — compatible with our academic challenge submission.

## Available checkpoints

| Variant | Use for diagnostic |
|---|---|
| **CT-CLIP (base)** | Default retrieval encoder for `src/diagnostics/retrieval.py` |
| CT-CLIP (VocabFine) | Optional ablation |
| CT-CLIP (ClassFine) | Not needed for retrieval R@K |
| Text Classifier Model | Not needed |

## Implication for plan

- R5 fallback path "train BioBERT+2.5D briefly" is **NOT triggered**. No 3–5 day mini-training of a substitute retrieval encoder.
- The retrieval diagnostic (Phase B B.3 `retrieval.py`) can proceed with the published CT-CLIP base checkpoint.

## Day 2 / Day 3 follow-ups

- Day 2: pin exact HF revision / file hash in `docs/report2ct_external_components.md`.
- Day 3 (5/28): when GenerateCT adapter lands, add `src/diagnostics/retrieval.py` skeleton that loads CT-CLIP via `huggingface_hub` and runs a 5-sample retrieval sanity. Defer real R@1/R@5/R@10 measurement to Phase B B.3.
