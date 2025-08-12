"""
GraphCL-style self-supervised pretraining for the heterogeneous food graph.

Two augmented "views" of the same graph are created via stochastic transforms:
- Node dropping (mask nodes for selected types by removing their incident edges)
- Edge dropping (randomly remove a fraction of edges per relation)
- Edge feature masking (set edge_attr rows to zero for a fraction)

A heterogeneous GAT encoder is trained with an InfoNCE/NT-Xent objective to
align node embeddings between the two views for selected node types.

After pretraining, save the encoder weights to initialize downstream training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import argparse
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from src.models.configs import PretrainConfig
from src.models.encoders import HeteroGATEncoder
from .ddp_utils import ddp_setup, ddp_wrap_model, ddp_cleanup, is_main_process


class HeteroGATPretrainWrapper(nn.Module):
    def __init__(self, data: HeteroData, cfg: PretrainConfig):
        super().__init__()
        num_nodes_by_type = {nt: data[nt].num_nodes for nt in data.node_types}
        
        # Compute edge dimensions from data
        edge_dims = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                edge_dims[edge_type] = data[edge_type].edge_attr.size(-1)
            else:
                edge_dims[edge_type] = 0
        
        self.encoder = HeteroGATEncoder(data.metadata(), cfg, num_nodes_by_type, edge_dims)

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # Build x_dict from internal embeddings using current graph sizes
        num_nodes_by_type = {nt: data[nt].num_nodes for nt in data.node_types}
        device = next(self.encoder.parameters()).device
        x_dict = self.encoder._init_node_features(num_nodes_by_type, device)
        
        # Extract edge attributes
        edge_attr_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr.to(device)
        
        return self.encoder(x_dict, data.edge_index_dict, edge_attr_dict)


def cosine_nt_xent(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature  # [N, N]
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5
    return loss


@dataclass
class AugmentProb:
    drop_food: float = 0.0
    drop_feature: float = 0.0
    drop_edges: Dict[Tuple[str, str, str], float] = None
    mask_edge_attr: Dict[Tuple[str, str, str], float] = None


def _rand_mask(num: int, keep_prob: float, device: torch.device) -> Tensor:
    if keep_prob >= 1.0:
        return torch.ones(num, dtype=torch.bool, device=device)
    if keep_prob <= 0.0:
        return torch.zeros(num, dtype=torch.bool, device=device)
    return (torch.rand(num, device=device) < keep_prob)


def augment_graph(data: HeteroData, probs: AugmentProb, device: torch.device) -> Tuple[HeteroData, Dict[str, Tensor]]:
    """
    Return an augmented view and node keep masks per type. Nodes are not renumbered;
    we only drop incident edges for masked-out nodes.
    """
    view = HeteroData()
    masks: Dict[str, Tensor] = {}

    # Copy num_nodes
    for nt in data.node_types:
        view[nt].num_nodes = data[nt].num_nodes

    # Node keep masks
    masks["Food"] = _rand_mask(data["Food"].num_nodes, keep_prob=1.0 - probs.drop_food, device=device)
    masks["Feature"] = _rand_mask(data["Feature"].num_nodes, keep_prob=1.0 - probs.drop_feature, device=device)
    # Other node types kept by default
    for nt in data.node_types:
        if nt not in masks:
            masks[nt] = torch.ones(data[nt].num_nodes, dtype=torch.bool, device=device)

    # Edge dropping and attribute masking
    for et in data.edge_types:
        src, rel, dst = et
        ei = data[et].edge_index
        if ei is None:
            continue
        keep_e = torch.ones(ei.size(1), dtype=torch.bool, device=device)

        # Drop by incident masked nodes
        keep_e &= masks[src][ei[0].to(device)]
        keep_e &= masks[dst][ei[1].to(device)]

        # Stochastic edge drop
        p_drop = 0.0
        if probs.drop_edges and et in probs.drop_edges:
            p_drop = probs.drop_edges[et]
        if p_drop > 0.0:
            keep_e &= (_rand_mask(ei.size(1), keep_prob=1.0 - p_drop, device=device))

        new_ei = ei[:, keep_e]
        view[et].edge_index = new_ei

        # Edge attr masking
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            ea = data[et].edge_attr.to(device)
            ea = ea[keep_e]
            p_mask = 0.0
            if probs.mask_edge_attr and et in probs.mask_edge_attr:
                p_mask = probs.mask_edge_attr[et]
            if p_mask > 0.0 and ea.numel() > 0:
                mask_rows = ~_rand_mask(ea.size(0), keep_prob=1.0 - p_mask, device=device)
                ea = ea.clone()
                ea[mask_rows] = 0.0
            view[et].edge_attr = ea

    return view, masks


def pretrain(
    graph_path: str,
    out_path: str = "data/models/pretrain_graphcl_encoder.pt",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.2,
    device: str = "cpu",
    # Augmentations
    drop_food: float = 0.1,
    drop_feature: float = 0.1,
    drop_sample_feature_edges: float = 0.1,
    drop_food_nutrient_edges: float = 0.1,
    mask_intensity: float = 0.1,
    mask_nutrient_amount: float = 0.1,
    # Node types to include in InfoNCE loss
    loss_node_types: Optional[List[str]] = None,
) -> None:
    device_t, rank, world_size, local_rank = ddp_setup()
    data: HeteroData = torch.load(graph_path, map_location=device_t)

    cfg = PretrainConfig(temperature=temperature)
    model = HeteroGATPretrainWrapper(data, cfg)
    model = ddp_wrap_model(model, device_t, world_size, local_rank)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_node_types is None:
        loss_node_types = ["Food", "Sample"]

    drop_edges = {
        ("Sample", "Contains", "Feature"): drop_sample_feature_edges,
        ("Food", "Contains", "Nutrient"): drop_food_nutrient_edges,
        ("Sample", "Is_of_type", "Food"): 0.0,
    }
    mask_edge_attr = {
        ("Sample", "Contains", "Feature"): mask_intensity,
        ("Food", "Contains", "Nutrient"): mask_nutrient_amount,
    }

    probs = AugmentProb(
        drop_food=drop_food,
        drop_feature=drop_feature,
        drop_edges=drop_edges,
        mask_edge_attr=mask_edge_attr,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        view1, masks1 = augment_graph(data, probs, device_t)
        view2, masks2 = augment_graph(data, probs, device_t)

        z1 = model(view1)
        z2 = model(view2)

        total_loss = 0.0
        for nt in loss_node_types:
            m = masks1[nt] & masks2[nt]
            idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            loss_nt = cosine_nt_xent(z1[nt][idx], z2[nt][idx], temperature=temperature)
            total_loss = total_loss + loss_nt

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if is_main_process(rank):
            print(f"Epoch {epoch:03d} | GraphCL loss {float(total_loss):.4f}")

    if is_main_process(rank):
        torch.save({
            "encoder_state": model.state_dict(),
            "config": cfg.__dict__,
            "graph_path": graph_path,
        }, out_path)
        print(f"Saved pretrained encoder to {out_path}")
    ddp_cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GraphCL pretraining for heterogeneous GAT")
    p.add_argument("--graph", type=str, default="data/hetero_graph.pt")
    p.add_argument("--out", type=str, default="data/models/pretrain_graphcl_encoder.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--drop-food", type=float, default=0.1)
    p.add_argument("--drop-feature", type=float, default=0.1)
    p.add_argument("--drop-sample-feature-edges", type=float, default=0.1)
    p.add_argument("--drop-food-nutrient-edges", type=float, default=0.1)
    p.add_argument("--mask-intensity", type=float, default=0.1)
    p.add_argument("--mask-nutrient-amount", type=float, default=0.1)
    p.add_argument("--loss-node-types", type=str, nargs="+", default=["Food", "Sample"])

    a = p.parse_args()
    pretrain(
        graph_path=a.graph,
        out_path=a.out,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        temperature=a.temperature,
        device=a.device,
        drop_food=a.drop_food,
        drop_feature=a.drop_feature,
        drop_sample_feature_edges=a.drop_sample_feature_edges,
        drop_food_nutrient_edges=a.drop_food_nutrient_edges,
        mask_intensity=a.mask_intensity,
        mask_nutrient_amount=a.mask_nutrient_amount,
        loss_node_types=a.loss_node_types,
    )


