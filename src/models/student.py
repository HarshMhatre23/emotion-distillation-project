import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class EmotionStudent(nn.Module):
    def __init__(self, backbone_name="microsoft/deberta-v3-xsmall",
                 num_emotions=27, num_dimensions=2, dropout=0.1,
                 freeze_encoder=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden_dim = self.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.freeze_encoder = freeze_encoder

        # Single-layer head (stable, no double linear)
        self.classifier = nn.Linear(self.hidden_dim, num_emotions)
        self.regressor = nn.Linear(self.hidden_dim, num_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        dims = self.regressor(pooled)

        if return_hidden:
            return logits, dims, pooled
        return logits, dims
