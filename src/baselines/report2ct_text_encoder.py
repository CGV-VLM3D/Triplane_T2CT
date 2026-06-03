"""Report2CT text encoder — ported from `third_party/report2ct/vlm3d_inference.ipynb` cell 0.

3 HuggingFace biomedical text encoders concatenated → 2560-d pooled embedding per text.

The notebook saves the **pooled** vector (1D, 2560) for each of findings/impression.
Training script `diff_model_train_vlm3D_2560_multi_text.py:280-283` promotes (B, 2560) to
(B, 1, 2560), then concatenates findings+impression along dim 1 to get (B, 2, 2560).

Parity safety net: `tests/test_report2ct_parity.py::test_text_encoder_parity`.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_IDS: Final[list[str]] = [
    "abhinand/MedEmbed-large-v0.1",
    "medicalai/ClinicalBERT",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
]
MAX_SEQ_LEN: Final[int] = 512  # notebook cell 0


def _mean_pooling(output, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings ignoring padded positions. Source: notebook cell 0."""
    embeddings = output.last_hidden_state  # [B, seq_len, hidden]
    mask_f = mask.unsqueeze(-1).float()
    summed = torch.sum(embeddings * mask_f, dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class Report2CTTextEncoder(nn.Module):
    """Wraps 3 HF biomedical text encoders and returns concatenated pooled embedding.

    Output: 1D tensor of shape (sum_of_hidden_sizes,) per text input.
    Empirical sum for the 3 default models: 1024 + 768 + 768 = 2560.
    """

    def __init__(
        self,
        model_ids: list[str] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model_ids = list(model_ids) if model_ids is not None else list(MODEL_IDS)
        self.max_seq_len = max_seq_len
        self.device_ = torch.device(device)

        self.tokenizers = []
        self.models: list[nn.Module] = []
        for name in self.model_ids:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = (
                AutoModel.from_pretrained(name, trust_remote_code=True)
                .eval()
                .to(self.device_)
            )
            for p in model.parameters():
                p.requires_grad_(False)
            self.tokenizers.append(tokenizer)
            self.models.append(model)

        self.total_dim = sum(m.config.hidden_size for m in self.models)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text → 1D tensor of shape (total_dim,).

        Mirrors `encode_batch_multi` from the notebook for batch=1 input, returning
        only the pooled (c_vec) component (the c_ctx token-level is discarded — upstream
        saves only the pooled vector to JSON).
        """
        pooled_list = []
        for tokenizer, model in zip(self.tokenizers, self.models):
            inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            inputs = {k: v.to(self.device_) for k, v in inputs.items()}
            output = model(**inputs)
            pooled = _mean_pooling(output, inputs["attention_mask"])  # [1, hidden]
            pooled_list.append(pooled)
        c_vec = torch.cat(pooled_list, dim=-1).squeeze(0)  # (total_dim,)
        return c_vec.cpu().float()

    def encode_pair(
        self, findings: str, impression: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode both findings and impression in one call."""
        return self.encode(findings), self.encode(impression)


__all__ = ["Report2CTTextEncoder", "MODEL_IDS", "MAX_SEQ_LEN"]
