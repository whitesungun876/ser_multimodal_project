# src/models/fusion/transformer_fused.py
import torch
import torch.nn as nn

class FusedTransformer(nn.Module):
    """
    TransformerEncoder on top of earlier fused [speech | text] vectors.
    For simplicity, think of 1536-D as a length=1 “token”, 
    mapped to d_model first, plus a number of virtual learnable tokens to run the TransformerEncoder.
    """
    def __init__(self, in_dim: int = 1536, d_model: int = 256,
                 n_head: int = 4, n_layer: int = 4, n_classes: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # classification token （
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):              # x: [B, in_dim]
        b = x.size(0)
        token = self.cls_token.expand(b, -1, -1)           # [B,1,d_model]
        feats = self.proj(x).unsqueeze(1)                  # [B,1,d_model]
        feats = torch.cat([token, feats], dim=1)           # [B,2,d_model]

        feats = self.enc(feats)                            # [B,2,d_model]
        cls_emb = self.norm(feats[:, 0])                   # [B,d_model]
        logits = self.head(cls_emb)
        return logits
