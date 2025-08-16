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
    Advanced MLP decoder with multiple interaction patterns and attention mechanisms.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Multiple interaction patterns
        self.dot_product = nn.Linear(embedding_dim, 1)  # Element-wise product
        self.difference = nn.Linear(embedding_dim, hidden_dim // 4)  # Difference features
        self.hadamard = nn.Linear(embedding_dim, hidden_dim // 4)  # Hadamard product
        
        # Attention mechanism for feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Main MLP with residual connections
        self.input_proj = nn.Linear(embedding_dim * 2 + hidden_dim // 2, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(3)
        ])
        
        # Final layers with skip connections
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for skip connection
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # More sophisticated activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Layer normalization for input features
        self.input_norm = nn.LayerNorm(embedding_dim * 2 + hidden_dim // 2)
    
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        z_src_emb = z_src[src]  # [num_edges, embedding_dim]
        z_dst_emb = z_dst[dst]  # [num_edges, embedding_dim]
        
        # Multiple interaction patterns
        # 1. Concatenation (original)
        z_concat = torch.cat([z_src_emb, z_dst_emb], dim=-1)
        
        # 2. Element-wise product
        z_product = z_src_emb * z_dst_emb
        product_score = self.dot_product(z_product)
        
        # 3. Difference features
        z_diff = z_src_emb - z_dst_emb
        diff_features = self.difference(z_diff)
        
        # 4. Hadamard product features
        z_hadamard = z_src_emb * z_dst_emb
        hadamard_features = self.hadamard(z_hadamard)
        
        # Combine all interaction patterns
        combined_features = torch.cat([
            z_concat, 
            diff_features, 
            hadamard_features
        ], dim=-1)
        
        # Normalize input
        combined_features = self.input_norm(combined_features)
        
        # Project to hidden dimension
        hidden = self.input_proj(combined_features)
        
        # Apply residual blocks with attention
        skip_connections = []
        for i, block in enumerate(self.residual_blocks):
            hidden = block(hidden)
            if i % 2 == 0:  # Every other block, apply attention
                # Reshape for attention (batch_size=1, seq_len=num_edges, hidden_dim)
                hidden_reshaped = hidden.unsqueeze(0)  # [1, num_edges, hidden_dim]
                attn_out, _ = self.attention(hidden_reshaped, hidden_reshaped, hidden_reshaped)
                hidden = attn_out.squeeze(0)  # [num_edges, hidden_dim]
            skip_connections.append(hidden)
        
        # Final processing with skip connections
        final_input = torch.cat([hidden, skip_connections[-1]], dim=-1)
        output = self.final_layers(final_input)
        
        return output.squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and GELU activation."""
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(hidden_dim, hidden_dim) if hidden_dim != hidden_dim else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + self.shortcut(x)


