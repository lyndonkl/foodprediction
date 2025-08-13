from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

from .configs import ModelConfig, PretrainConfig


class HeteroGATEncoder(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        cfg: ModelConfig | PretrainConfig,
        num_nodes_by_type: Dict[str, int],
        edge_dims: Dict[Tuple[str, str, str], int] = None,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.cfg = cfg

        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(
                num_embeddings=num_nodes_by_type[node_type],
                embedding_dim=cfg.embedding_dim,
            )
            for node_type in node_types
        })

        self.layers = nn.ModuleList()
        in_dim = cfg.embedding_dim
        for _ in range(cfg.num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        in_channels=(in_dim, in_dim),
                        out_channels=cfg.hidden_dim // cfg.num_heads,
                        heads=cfg.num_heads,
                        dropout=cfg.dropout,
                        add_self_loops=False,
                        concat=True,
                        edge_dim=edge_dims.get(edge_type, 0) if edge_dims else 0,
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.layers.append(conv)
            in_dim = cfg.hidden_dim

        self.dropout = nn.Dropout(cfg.dropout)

    def _init_node_features(self, num_nodes_by_type: Dict[str, int], device: torch.device) -> Dict[str, Tensor]:
        x_dict: Dict[str, Tensor] = {}
        for node_type, num_nodes in num_nodes_by_type.items():
            idx = torch.arange(num_nodes, device=device)
            x_dict[node_type] = self.embeddings[node_type](idx)
        return x_dict

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], edge_attr_dict: Dict[Tuple[str, str, str], Tensor] = None) -> Dict[str, Tensor]:
        h_dict = x_dict
        for conv in self.layers:
            h_dict = conv(h_dict, edge_index_dict, edge_attr_dict)
            h_dict = {k: F.elu(self.dropout(v)) for k, v in h_dict.items()}
        return h_dict


