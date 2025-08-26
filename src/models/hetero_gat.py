"""
Heterogeneous GAT link prediction model wrapper.

Training code lives in src/train/.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from .configs import ModelConfig
from .encoders import HeteroGATEncoder
from .decoders import DotProductDecoder, FoodOriginPredictionHead, EnhancedFoodOriginPredictionHead
from torch_geometric.data import HeteroData


class HeteroLinkPredModel(nn.Module):
    def __init__(self, data_metadata: Tuple[List[str], List[Tuple[str, str, str]]], cfg: ModelConfig, supervised_edge_types: List[Tuple[str, str, str]], data: HeteroData = None, use_enhanced_decoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.supervised_edge_types = supervised_edge_types
        self.use_enhanced_decoder = use_enhanced_decoder
        
        # Compute edge dimensions from data if provided
        edge_dims = {}
        if data is not None:
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                    edge_dims[edge_type] = data[edge_type].edge_attr.size(-1)
                else:
                    edge_dims[edge_type] = 0
        
        # Get number of nodes by type for encoder initialization
        num_nodes_by_type = {}
        if data is not None:
            for node_type in data.node_types:
                num_nodes_by_type[node_type] = data[node_type].num_nodes
        
        self.encoder = HeteroGATEncoder(data_metadata, cfg, num_nodes_by_type, edge_dims=edge_dims if data is not None else None)
        
        # Task 1: Food Origin Prediction decoder (MLP for Sample->Food edges)
        self.decoders = nn.ModuleDict()
        for edge_type in supervised_edge_types:
            # All supervised edge types should be Sample->Food for Food Origin Prediction
            if edge_type == ("Sample", "Is_of_type", "Food"):
                if use_enhanced_decoder:
                    self.decoders[str(edge_type)] = EnhancedFoodOriginPredictionHead(
                        embedding_dim=cfg.hidden_dim,
                        hidden_dim=cfg.hidden_dim,
                        dropout=cfg.dropout
                    )
                else:
                    self.decoders[str(edge_type)] = FoodOriginPredictionHead(
                        embedding_dim=cfg.hidden_dim,
                        hidden_dim=cfg.hidden_dim,
                        dropout=cfg.dropout
                    )
            else:
                raise ValueError(f"Only Sample->Food edges should be supervised for Food Origin Prediction, got {edge_type}")

    def init_node_features(self, num_nodes_by_type: Dict[str, int], device: torch.device) -> Dict[str, Tensor]:
        x_dict: Dict[str, Tensor] = {}
        for node_type, num_nodes in num_nodes_by_type.items():
            indices = torch.arange(num_nodes, device=device)
            x = self.encoder.embeddings[node_type](indices)
            x_dict[node_type] = x
        return x_dict

    def predict_edge_scores(self, z_dict: Dict[str, Tensor], edge_type: Tuple[str, str, str], edge_label_index: Tensor, 
                          data: HeteroData = None) -> Tensor:
        src_type, _, dst_type = edge_type
        decoder = self.decoders[str(edge_type)]
        
        if self.use_enhanced_decoder and data is not None:
            # Enhanced decoder with feature attention
            z_features = z_dict.get("Feature", None)
            feature_edge_type = ("Sample", "Contains", "Feature")
            
            if feature_edge_type in data.edge_types:
                feature_edge_index = data[feature_edge_type].edge_index
                feature_edge_weights = data[feature_edge_type].edge_attr if hasattr(data[feature_edge_type], "edge_attr") and data[feature_edge_type].edge_attr is not None else None
                
                return decoder(z_dict[src_type], z_dict[dst_type], edge_label_index, 
                             z_features, feature_edge_index, feature_edge_weights)
            else:
                # Fallback to basic decoder if no feature edges
                return decoder(z_dict[src_type], z_dict[dst_type], edge_label_index)
        else:
            # Basic decoder
            return decoder(z_dict[src_type], z_dict[dst_type], edge_label_index)

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], edge_attr_dict: Dict[Tuple[str, str, str], Tensor] = None) -> Dict[str, Tensor]:
        return self.encoder(x_dict, edge_index_dict, edge_attr_dict)


