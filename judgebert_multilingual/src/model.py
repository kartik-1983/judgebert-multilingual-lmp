"""
JUDGEBERT-inspired multilingual regression model for Legal Meaning Preservation.

Improvements over basic version:
- Mean pooling over the sequence (masked) instead of only [CLS]
  (often more robust for sentence-pair regression)
- Slightly deeper / better-initialized regression head
- Clean separation of encoder and head for layer-wise learning rates
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class JudgeBERTRegressor(nn.Module):
    """Multilingual JUDGEBERT-style LMP regression model."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.15,
        freeze_base: bool = False,
        use_mean_pooling: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_mean_pooling = use_mean_pooling
        hidden_size = self.config.hidden_size

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Stronger regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Xavier init for the head
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling."""
        # last_hidden_state: (B, T, H), attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        if self.use_mean_pooling:
            pooled = self._mean_pooling(last_hidden, attention_mask)
        else:
            pooled = last_hidden[:, 0, :]  # [CLS]

        pooled = self.dropout(pooled)
        logits = self.regressor(pooled).squeeze(-1)  # (B,)

        # Soft clamp via sigmoid scaling is possible, but hard clamp is fine for training stability
        preds = torch.clamp(logits, 1.0, 10.0)

        loss = None
        if labels is not None:
            # SmoothL1 is robust to outliers
            loss_fct = nn.SmoothL1Loss(beta=1.0)
            loss = loss_fct(preds, labels.float())

        return {
            "loss": loss,
            "logits": logits,          # unclamped for potential ranking losses later
            "predictions": preds,
        }


def get_tokenizer(model_name: str = "xlm-roberta-base"):
    return AutoTokenizer.from_pretrained(model_name)


def prepare_pair(
    tokenizer,
    original: str,
    simplified: str,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Tokenize a single original–simplified pair for the model."""
    encoding = tokenizer(
        original,
        simplified,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }
