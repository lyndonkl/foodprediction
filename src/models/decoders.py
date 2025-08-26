from __future__ import annotations

from torch import nn, Tensor
import torch
import torch.nn.functional as F


class DotProductDecoder(nn.Module):
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        return (z_src[src] * z_dst[dst]).sum(dim=-1)


class EnhancedFoodOriginPredictionHead(nn.Module):
    """
    Enhanced Food Origin Prediction Head with Feature Attention
    Incorporates feature embeddings and edge weights using attention mechanism.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.3, num_attention_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        
        # Feature attention mechanism
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Edge weight projection for attention
        self.edge_weight_proj = nn.Linear(1, embedding_dim)
        
        # Multiple interaction patterns (enhanced)
        self.dot_product = nn.Linear(embedding_dim, 1)
        self.difference = nn.Linear(embedding_dim, hidden_dim // 4)
        self.hadamard = nn.Linear(embedding_dim, hidden_dim // 4)
        
        # Feature context projection
        self.feature_context_proj = nn.Linear(embedding_dim, hidden_dim // 4)
        
        # Main MLP with residual connections
        total_input_dim = embedding_dim * 2 + 1 + hidden_dim // 2 + hidden_dim // 4  # + feature context
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(3)
        ])
        
        # Final layers with skip connections
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Layer normalization for input features
        self.input_norm = nn.LayerNorm(total_input_dim)
    
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor, 
                z_features: Tensor = None, feature_edge_index: Tensor = None, 
                feature_edge_weights: Tensor = None) -> Tensor:
        src, dst = edge_label_index
        z_src_emb = z_src[src]  # [num_edges, embedding_dim]
        z_dst_emb = z_dst[dst]  # [num_edges, embedding_dim]
        
        # Feature attention mechanism
        feature_context = torch.zeros_like(z_src_emb)  # Default to zero if no features
        if z_features is not None and feature_edge_index is not None:
            feature_context = self._compute_feature_attention(
                z_src_emb, z_features, feature_edge_index, feature_edge_weights
            )
        
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
        
        # 5. Feature context features
        feature_context_features = self.feature_context_proj(feature_context)
        
        # Combine all interaction patterns
        combined_features = torch.cat([
            z_concat, 
            product_score,
            diff_features, 
            hadamard_features,
            feature_context_features
        ], dim=-1)
        
        # Normalize input
        combined_features = self.input_norm(combined_features)
        
        # Project to hidden dimension
        hidden = self.input_proj(combined_features)
        
        # Apply residual blocks
        skip_connections = []
        for i, block in enumerate(self.residual_blocks):
            hidden = block(hidden)
            skip_connections.append(hidden)
        
        # Final processing with skip connections
        final_input = torch.cat([hidden, skip_connections[0]], dim=-1)
        output = self.final_layers(final_input)
        
        return output.squeeze(-1)
    
    def _compute_feature_attention(self, z_src_emb: Tensor, z_features: Tensor, 
                                 feature_edge_index: Tensor, feature_edge_weights: Tensor) -> Tensor:
        """
        Compute attention-weighted feature context for each sample.
        
        Args:
            z_src_emb: Sample embeddings [num_edges, embedding_dim]
            z_features: Feature embeddings [num_features, embedding_dim]
            feature_edge_index: Sample->Feature edge indices [2, num_feature_edges]
            feature_edge_weights: Feature edge weights [num_feature_edges, 1]
        
        Returns:
            Feature context for each sample [num_edges, embedding_dim]
        """
        # Group features by sample
        sample_to_features = {}
        for i in range(feature_edge_index.size(1)):
            sample_idx = feature_edge_index[0, i]
            feature_idx = feature_edge_index[1, i]
            weight = feature_edge_weights[i] if feature_edge_weights is not None else 1.0
            
            if sample_idx.item() not in sample_to_features:
                sample_to_features[sample_idx.item()] = []
            sample_to_features[sample_idx.item()].append((feature_idx.item(), weight))
        
        # Compute attention for each sample
        feature_contexts = []
        for i, sample_emb in enumerate(z_src_emb):
            sample_idx = i  # Assuming edge_label_index[0] corresponds to sample indices
            
            if sample_idx in sample_to_features:
                # Get features for this sample
                feature_indices = [f[0] for f in sample_to_features[sample_idx]]
                feature_weights = torch.tensor([f[1] for f in sample_to_features[sample_idx]], 
                                             device=z_src_emb.device, dtype=torch.float32)
                
                # Get feature embeddings
                sample_features = z_features[feature_indices]  # [num_sample_features, embedding_dim]
                
                # Project edge weights
                projected_weights = self.edge_weight_proj(feature_weights.unsqueeze(1))  # [num_sample_features, embedding_dim]
                
                # Add weight information to features
                weighted_features = sample_features + projected_weights
                
                # Apply attention: sample_emb as query, weighted_features as key/value
                query = sample_emb.unsqueeze(0)  # [1, embedding_dim]
                key = value = weighted_features  # [num_sample_features, embedding_dim]
                
                # Multi-head attention
                attn_output, _ = self.feature_attention(query, key, value)
                feature_context = attn_output.squeeze(0)  # [embedding_dim]
            else:
                # No features for this sample
                feature_context = torch.zeros(self.embedding_dim, device=z_src_emb.device)
            
            feature_contexts.append(feature_context)
        
        return torch.stack(feature_contexts)  # [num_edges, embedding_dim]


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
        
        # Removed attention mechanism - each edge prediction is independent
        
        # Main MLP with residual connections
        self.input_proj = nn.Linear(embedding_dim * 2 + 1 + hidden_dim // 2, hidden_dim)
        
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
        self.input_norm = nn.LayerNorm(embedding_dim * 2 + 1 + hidden_dim // 2)
    
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
            product_score,  # Include the learned dot product score
            diff_features, 
            hadamard_features
        ], dim=-1)
        
        # Normalize input
        combined_features = self.input_norm(combined_features)
        
        # Project to hidden dimension
        hidden = self.input_proj(combined_features)
        
        # Apply residual blocks (removed attention - each edge prediction is independent)
        skip_connections = []
        for i, block in enumerate(self.residual_blocks):
            hidden = block(hidden)
            skip_connections.append(hidden)
        
        # Final processing with skip connections (combine early and late features)
        final_input = torch.cat([hidden, skip_connections[0]], dim=-1)
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
        self.shortcut = nn.Identity()  # Identity mapping since input_dim == hidden_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + self.shortcut(x)


