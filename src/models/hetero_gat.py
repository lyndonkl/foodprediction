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
from .decoders import DotProductDecoder
from torch_geometric.data import HeteroData


class HeteroLinkPredModel(nn.Module):
    def __init__(self, data_metadata: Tuple[List[str], List[Tuple[str, str, str]]], cfg: ModelConfig, supervised_edge_types: List[Tuple[str, str, str]], data: HeteroData = None):
        super().__init__()
        self.cfg = cfg
        self.supervised_edge_types = supervised_edge_types
        
        # Compute edge dimensions from data if provided
        edge_dims = {}
        if data is not None:
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                    edge_dims[edge_type] = data[edge_type].edge_attr.size(-1)
                else:
                    edge_dims[edge_type] = 0
        
        self.encoder = HeteroGATEncoder(data_metadata, cfg, edge_dims=edge_dims if data is not None else None)
        self.decoder = nn.ModuleDict({
            str(edge_type): DotProductDecoder() for edge_type in supervised_edge_types
        })

    def init_node_features(self, num_nodes_by_type: Dict[str, int], device: torch.device) -> Dict[str, Tensor]:
        x_dict: Dict[str, Tensor] = {}
        for node_type, num_nodes in num_nodes_by_type.items():
            indices = torch.arange(num_nodes, device=device)
            x = self.encoder.embeddings[node_type](indices)
            x_dict[node_type] = x
        return x_dict

    def predict_edge_scores(self, z_dict: Dict[str, Tensor], edge_type: Tuple[str, str, str], edge_label_index: Tensor) -> Tensor:
        src_type, _, dst_type = edge_type
        decoder = self.decoder[str(edge_type)]
        return decoder(z_dict[src_type], z_dict[dst_type], edge_label_index)

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], edge_attr_dict: Dict[Tuple[str, str, str], Tensor] = None) -> Dict[str, Tensor]:
        return self.encoder(x_dict, edge_index_dict, edge_attr_dict)


