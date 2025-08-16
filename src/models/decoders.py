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
    Deep MLP decoder specifically for Sample->Food link prediction.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            # Input layer: concatenated embeddings -> hidden_dim
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer 1: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer 2: hidden_dim -> hidden_dim // 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer 3: hidden_dim // 2 -> hidden_dim // 4
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer 4: hidden_dim // 4 -> hidden_dim // 8
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: hidden_dim // 8 -> 1
            nn.Linear(hidden_dim // 8, 1)
        )
    
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        # Concatenate source and destination embeddings
        z_concat = torch.cat([z_src[src], z_dst[dst]], dim=-1)
        return self.mlp(z_concat).squeeze(-1)


