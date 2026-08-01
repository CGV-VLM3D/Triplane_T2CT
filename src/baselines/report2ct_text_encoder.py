"""Report2CT text encoder — ported from `third_party/report2ct/vlm3d_inference.ipynb` cell 0.

3 HuggingFace biomedical text encoders concatenated → 2560-d pooled embedding per text.

The notebook saves the **pooled** vector (1D, 2560) for each of findings/impression.
Training script `diff_model_train_vlm3D_2560_multi_text.py:280-283` promotes (B, 2560) to
(B, 1, 2560), then concatenates findings+impression along dim 1 to get (B, 2, 2560).

`encode_tokens` exposes the token-level component (upstream's `c_ctx`) that the pooled
path throws away; it is additive and leaves `encode` / `encode_pair` untouched.

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
    """Mean-pool token embeddings ignoring padded positions.

    Args:
        output: HF model output exposing ``last_hidden_state`` ``(B, seq_len, hidden)``.
        mask: attention mask, ``(B, seq_len)``.

    Returns:
        Pooled embedding, ``(B, hidden)``.

    Source: notebook cell 0.
    """
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
        """Encode a single text to the concatenated pooled embedding.

        Args:
            text: one findings or impression string.

        Returns:
            1D pooled embedding, ``(total_dim,)`` (total_dim = 2560 for the 3
            default models), on CPU float32.

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
        """Encode both findings and impression in one call.

        Args:
            findings: findings section text.
            impression: impression section text.

        Returns:
            Tuple ``(findings_emb, impression_emb)``, each ``(total_dim,)``.
        """
        return self.encode(findings), self.encode(impression)

    @torch.no_grad()
    def encode_tokens(self, text: str) -> list[torch.Tensor]:
        """Encode a single text to per-encoder token sequences (upstream's ``c_ctx``).

        This is the token-level component that `encode` discards. Upstream's
        `encode_batch_multi` builds it as one `(num_models * 512, max_dim)` block by
        right-padding every sequence to `MAX_SEQ_LEN` and zero-padding every hidden dim
        up to `max_dim` (1024), then concatenating along the sequence axis. We keep the
        sequences **separate and unpadded** instead: 2 of the 3 encoders are 768-d, so
        the upstream layout stores 256 zero columns for each of their 1536 rows, and
        real reports use ~250 of the 512 sequence slots. The consumer projects each
        encoder to a common width with its own learned linear layer, which needs the
        native dims anyway.

        Args:
            text: one findings or impression string.

        Returns:
            One tensor per encoder, in ``self.model_ids`` order, each
            ``(n_tokens, hidden_i)`` on CPU float32 (hidden_i = 1024 / 768 / 768 for the
            3 default models). ``n_tokens`` is the true token count including the CLS and
            SEP specials (upstream keeps them too), capped at ``self.max_seq_len``.
        """
        token_seqs: list[torch.Tensor] = []
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
            hidden = output.last_hidden_state[0]  # (seq_len, hidden_i)
            # batch=1 + padding=True means seq_len is already the true length; the mask
            # filter keeps this correct if the call is ever batched.
            keep = inputs["attention_mask"][0].bool()  # (seq_len,)
            token_seqs.append(hidden[keep].cpu().float())  # (n_tokens, hidden_i)
        return token_seqs

    def encode_tokens_pair(
        self, findings: str, impression: str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode findings + impression to token sequences in one call.

        Args:
            findings: findings section text.
            impression: impression section text.

        Returns:
            Tuple ``(findings_seqs, impression_seqs)``, each a per-encoder list of
            ``(n_tokens, hidden_i)`` as described in `encode_tokens`.
        """
        return self.encode_tokens(findings), self.encode_tokens(impression)


__all__ = ["Report2CTTextEncoder", "MODEL_IDS", "MAX_SEQ_LEN"]
