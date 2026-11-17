"""
JUDGEBERT-inspired multilingual regression model for Legal Meaning Preservation.

Architecture (following the original JUDGEBERT / MeaningBERT design):
- Base: XLM-RoBERTa-base (multilingual)
- Input: [CLS] original [SEP] simplified [SEP]
- Head: Regression head (linear layer) on the [CLS] representation
- Loss: MSE (or SmoothL1)
- Output: continuous LMP score in [1, 10]
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)


class JudgeBERTRegressor(nn.Module):
    """
    Multilingual JUDGEBERT-style model.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Initialize the regression head
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        # Use [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)
        logits = self.regressor(cls_repr).squeeze(-1)  # (batch,)

        # Clamp predictions to sensible range
        preds = torch.clamp(logits, 1.0, 10.0)

        loss = None
        if labels is not None:
            loss_fct = nn.SmoothL1Loss()
            loss = loss_fct(preds, labels.float())

        return {
            "loss": loss,
            "logits": preds,
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
