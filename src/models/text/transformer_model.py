import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, 
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = input_dim
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[-1, :, :]
        return self.classifier(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
