from __future__ import annotations

from torch import nn, Tensor
import torch


class DotProductDecoder(nn.Module):
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        return (z_src[src] * z_dst[dst]).sum(dim=-1)


class FoodOriginPredictionHead(nn.Module):
    """
    Task 1 Head: Food Origin Prediction
    MLP decoder specifically for Sample->Food link prediction.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        # Concatenate source and destination embeddings
        z_concat = torch.cat([z_src[src], z_dst[dst]], dim=-1)
        return self.mlp(z_concat).squeeze(-1)


