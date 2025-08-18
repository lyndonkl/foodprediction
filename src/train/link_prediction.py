"""
Multi-Task Learning (MTL) training for the heterogeneous food metabolomics graph.

Architecture:
- Shared GNN Encoder: HeteroGAT processes all edge types with attention (provides graph structure)
- Task 1 Head (Food Origin Prediction): MLP decoder for Sample->Food link prediction only
- Task 2 Head (Nutritional Organization): Contrastive learning on Food embeddings

Notes:
- Node types have no input features; we use learnable embeddings per node type.
- Only Sample->Food edges are supervised for link prediction (Task 1).
- Other edge types (Food->Nutrient, Sample->Feature) are used only for graph structure in the shared encoder.
- Uses RandomLinkSplit for HeteroData with negative sampling.
- Both tasks contribute equally to the total loss (equal weighting).
- Contrastive learning uses nutritional similarity from graph structure.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import argparse
import json
import math
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from src.models.configs import ModelConfig
from src.models.hetero_gat import HeteroLinkPredModel
from .ddp_utils import ddp_setup, ddp_wrap_model, ddp_cleanup, is_main_process


from src.models.hetero_gat import HeteroLinkPredModel


# -------------------------
# Contrastive helpers
# -------------------------


def _compute_nutritional_similarity_matrix(data: HeteroData, device: torch.device) -> Tensor:
    """
    Build a Food x Food similarity matrix based on shared nutrients.
    Uses the graph structure to find nutritionally similar foods.
    """
    if ("Food", "Contains", "Nutrient") not in data.edge_types:
        return torch.eye(data["Food"].num_nodes, device=device)
    
    et = ("Food", "Contains", "Nutrient")
    ei = data[et].edge_index
    if ei is None or ei.numel() == 0:
        return torch.eye(data["Food"].num_nodes, device=device)
    
    F = data["Food"].num_nodes
    N = data["Nutrient"].num_nodes
    
    # Create binary Food x Nutrient matrix (1 if food contains nutrient, 0 otherwise)
    food_nutrient_matrix = torch.zeros((F, N), dtype=torch.float32, device=device)
    food_nutrient_matrix.index_put_((ei[0].to(device), ei[1].to(device)), torch.ones(ei.size(1), device=device))
    
    # Compute cosine similarity between foods based on shared nutrients
    # Normalize rows
    norms = food_nutrient_matrix.norm(dim=1, keepdim=True).clamp_min(1e-6)
    food_nutrient_norm = food_nutrient_matrix / norms
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(food_nutrient_norm, food_nutrient_norm.t())
    
    # Mask self-similarity
    similarity_matrix.fill_diagonal_(0.0)
    
    return similarity_matrix


def _sample_nutritional_triplets(
    similarity_matrix: Tensor,
    num_triplets: int,
    pos_threshold: float = 0.5,
    neg_threshold: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor] | None:
    """
    Sample (anchor, positive, negative) indices for Food contrastive learning
    based on nutritional similarity from the graph structure.
    
    Args:
        similarity_matrix: Food x Food similarity matrix
        num_triplets: Number of triplets to sample
        pos_threshold: Minimum similarity for positive pairs
        neg_threshold: Maximum similarity for negative pairs
    """
    F = similarity_matrix.size(0)
    if F < 3:
        return None
    
    # Find positive and negative pairs for each food
    pos_mask = similarity_matrix >= pos_threshold
    neg_mask = similarity_matrix <= neg_threshold
    
    anchors = []
    positives = []
    negatives = []
    
    # Sample triplets
    for anchor in range(F):
        # Find positive candidates (nutritionally similar)
        pos_candidates = torch.where(pos_mask[anchor])[0]
        if len(pos_candidates) == 0:
            continue
            
        # Find negative candidates (nutritionally dissimilar)
        neg_candidates = torch.where(neg_mask[anchor])[0]
        if len(neg_candidates) == 0:
            continue
        
        # Sample positive and negative
        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,))]
        neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,))]
        
        anchors.append(anchor)
        positives.append(pos_idx.item())
        negatives.append(neg_idx.item())
        
        if len(anchors) >= num_triplets:
            break
    
    if not anchors:
        return None
    
    return (
        torch.tensor(anchors, dtype=torch.long, device=similarity_matrix.device),
        torch.tensor(positives, dtype=torch.long, device=similarity_matrix.device),
        torch.tensor(negatives, dtype=torch.long, device=similarity_matrix.device),
    )


# -------------------------
# Utils
# -------------------------


def to_device(data: HeteroData, device: torch.device) -> HeteroData:
    return data.to(device)


@torch.no_grad()
def evaluate(model: HeteroLinkPredModel, data: HeteroData, split: str, device: torch.device = None) -> Dict[str, float]:
    model.eval()
    
    # Use the same device as the model
    if device is None:
        device = next(model.parameters()).device
    
    # Extract edge attributes for model forward pass
    edge_attr_dict = {}
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
            edge_attr_dict[edge_type] = data[edge_type].edge_attr.to(device)
    
    # Initialize node features
    num_nodes_by_type = {nt: data[nt].num_nodes for nt in data.node_types}
    x_dict = model.init_node_features(num_nodes_by_type, device)
    
    # Get embeddings from shared encoder
    z_dict = model(x_dict, data.edge_index_dict, edge_attr_dict)
    
    losses: List[float] = []
    aucs: List[float] = []

    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    for edge_type in model.supervised_edge_types:
        et_key = edge_type
        pos_edge_label_index = data[et_key][f"{split}_edge_label_index_pos"]
        neg_edge_label_index = data[et_key][f"{split}_edge_label_index_neg"]

        pos_scores = model.predict_edge_scores(z_dict, edge_type, pos_edge_label_index)
        neg_scores = model.predict_edge_scores(z_dict, edge_type, neg_edge_label_index)

        y_pred = torch.cat([pos_scores, neg_scores], dim=0)
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        losses.append(loss.item())

        if roc_auc_score is not None:
            auc = float(roc_auc_score(y_true.cpu().numpy(), y_pred.detach().cpu().numpy()))
            aucs.append(auc)

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "auc": float(sum(aucs) / max(1, len(aucs))) if aucs else float("nan"),
    }


def attach_split_labels(split_data: Tuple[HeteroData, HeteroData, HeteroData], edge_types: List[Tuple[str, str, str]]) -> Tuple[HeteroData, HeteroData, HeteroData]:
    train_data, val_data, test_data = split_data

    def _extract_labels(data: HeteroData, split_name: str):
        for et in edge_types:
            # RandomLinkSplit creates edge_label (binary labels) and edge_label_index (edge pairs) for each split
            edge_label: Tensor = data[et].edge_label
            edge_label_index: Tensor = data[et].edge_label_index
            # Split into pos/neg by label
            pos_mask = edge_label == 1
            neg_mask = edge_label == 0
            data[et][f"{split_name}_edge_label_index_pos"] = edge_label_index[:, pos_mask]
            data[et][f"{split_name}_edge_label_index_neg"] = edge_label_index[:, neg_mask]

    _extract_labels(train_data, "train")
    _extract_labels(val_data, "val")
    _extract_labels(test_data, "test")
    return train_data, val_data, test_data


# -------------------------
# Training
# -------------------------


def train(
    graph_path: str,
    target_edge_type: Tuple[str, str, str] = ("Sample", "Is_of_type", "Food"),
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    negative_sampling_ratio: float = 1.0,
    device: str = "cpu",
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.2,
    save_path: str = "data/models/linkpred_gat.pt",
    # Sampler params
    use_sampler: bool = True,
    batch_size: int = 2048,
    num_neighbors: List[int] | Tuple[int, ...] = (10, 10),
    # Contrastive objective on Food embeddings
    contrastive_weight: float = 1.0,
    contrastive_margin: float = 0.2,
    contrastive_triplets_per_epoch: int = 8192,
    contrastive_pos_threshold: float = 0.5,
    contrastive_neg_threshold: float = 0.1,
    # Loss weighting
    link_prediction_weight: float = 2.0,
    # Pretrained encoder
    pretrained_encoder_path: str = None,
) -> None:
    device_t = torch.device(device)
    full_data: HeteroData = torch.load(graph_path, map_location=device_t)

    # Prepare splits - only split the target edge type for supervision
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
        edge_types=[target_edge_type],  # Only supervise Sample->Food edges
        rev_edge_types=[(target_edge_type[2], f"rev_{target_edge_type[1]}", target_edge_type[0])],  # Include reverse edges to prevent leakage
        neg_sampling_ratio=negative_sampling_ratio,
    )
    train_data, val_data, test_data = transform(full_data)
    train_data, val_data, test_data = attach_split_labels((train_data, val_data, test_data), [target_edge_type])

    # Build model
    cfg = ModelConfig(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    device_t, rank, world_size, local_rank = ddp_setup()
    model = HeteroLinkPredModel(train_data.metadata(), cfg, supervised_edge_types=[target_edge_type], data=train_data)
    
    # Load pretrained encoder weights if provided
    if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
        if is_main_process(rank):
            print(f"Loading pretrained encoder from {pretrained_encoder_path}")
        checkpoint = torch.load(pretrained_encoder_path, map_location=device_t)
        
        # Extract only encoder weights from the pretrained model
        pretrained_state = checkpoint["encoder_state"]
        encoder_state = {}
        for key, value in pretrained_state.items():
            if key.startswith("encoder."):
                # Remove "encoder." prefix to match our model's encoder
                encoder_key = key[8:]  # Remove "encoder." prefix
                encoder_state[encoder_key] = value
        
        # Load encoder weights into our model's encoder
        model.encoder.load_state_dict(encoder_state, strict=False)
        if is_main_process(rank):
            print(f"Successfully loaded pretrained encoder weights ({len(encoder_state)} parameters)")
            print(f"Loaded keys: {list(encoder_state.keys())[:5]}...")  # Show first 5 keys
    
    model = ddp_wrap_model(model, device_t, world_size, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -math.inf
    best_state = None

    # Optional loader for target edge type using neighbor sampling
    loader = None
    if use_sampler:
        # Concatenate pos/neg indices for training label set
        pos_idx = train_data[target_edge_type]["train_edge_label_index_pos"]
        neg_idx = train_data[target_edge_type]["train_edge_label_index_neg"]
        edge_label_index = torch.cat([pos_idx, neg_idx], dim=1)
        edge_label = torch.cat([
            torch.ones(pos_idx.size(1), dtype=torch.float32),
            torch.zeros(neg_idx.size(1), dtype=torch.float32),
        ], dim=0)
        loader = LinkNeighborLoader(
            train_data,
            num_neighbors=list(num_neighbors) if isinstance(num_neighbors, (list, tuple)) else [num_neighbors],
            batch_size=batch_size,
            edge_label_index=(target_edge_type, edge_label_index),
            edge_label=edge_label,
            shuffle=True,
        )

    # Precompute nutritional similarity matrix for foods based on graph structure
    nutritional_similarity = _compute_nutritional_similarity_matrix(full_data, device_t)

    for epoch in range(1, epochs + 1):
        model.train()

        # Extract edge attributes for model forward pass
        edge_attr_dict = {}
        for edge_type in train_data.edge_types:
            if hasattr(train_data[edge_type], "edge_attr") and train_data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = train_data[edge_type].edge_attr.to(device_t)
        
        # Initialize node features
        num_nodes_by_type = {nt: train_data[nt].num_nodes for nt in train_data.node_types}
        x_dict = model.init_node_features(num_nodes_by_type, device_t)

        # ---- Combined MTL Training: Both tasks contribute equally ----
        optimizer.zero_grad()
        
        # Get embeddings from shared encoder (used by both tasks)
        z_dict = model(x_dict, train_data.edge_index_dict, edge_attr_dict)
        
        # Task 1: Link Prediction Loss (Food Origin Prediction)
        # Only predict Sample->Food edges using MLP decoder
        total_lp_loss = 0.0
        num_lp_steps = 0
        if use_sampler and loader:
            for batch in loader:
                batch = batch.to(device_t)
                edge_label_index = batch[target_edge_type].edge_label_index
                edge_label = batch[target_edge_type].edge_label
                scores = model.predict_edge_scores(z_dict, target_edge_type, edge_label_index)
                loss = F.binary_cross_entropy_with_logits(scores, edge_label)
                total_lp_loss += float(loss)
                num_lp_steps += 1
        else:
            pos_index = train_data[target_edge_type]["train_edge_label_index_pos"]
            neg_index = train_data[target_edge_type]["train_edge_label_index_neg"]
            pos_scores = model.predict_edge_scores(z_dict, target_edge_type, pos_index)
            neg_scores = model.predict_edge_scores(z_dict, target_edge_type, neg_index)
            y_pred = torch.cat([pos_scores, neg_scores], dim=0)
            y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
            total_lp_loss = float(loss)
            num_lp_steps = 1

        avg_lp_loss = total_lp_loss / max(1, num_lp_steps)
        
        # Task 2: Contrastive Loss on Food embeddings (Nutritional Organization)
        # Uses nutritional similarity from graph structure to find similar/dissimilar foods
        food_emb = z_dict["Food"]
        triplets = _sample_nutritional_triplets(
            nutritional_similarity, 
            contrastive_triplets_per_epoch,
            contrastive_pos_threshold,
            contrastive_neg_threshold
        )
        if triplets is not None:
            a, p, n = triplets
            triplet_loss_fn = nn.TripletMarginLoss(margin=contrastive_margin, p=2)
            closs = triplet_loss_fn(food_emb[a], food_emb[p], food_emb[n])
            closs_val = float(closs)
        else:
            closs_val = float('nan')
            closs = torch.tensor(0.0, device=device_t)
        
        # Combine losses with proper weighting
        total_loss = link_prediction_weight * avg_lp_loss + contrastive_weight * closs
        total_loss.backward()
        optimizer.step()

        # ---- Eval ----
        val_metrics = evaluate(model, val_data, split="val", device=device_t)
        if is_main_process(rank):
            print(
                f"Epoch {epoch:03d} | lp_loss {avg_lp_loss:.4f} | closs {closs_val:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | val_auc {val_metrics['auc']:.4f}"
            )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None and is_main_process(rank):
        torch.save({
            "model_state": best_state,
            "config": cfg.__dict__,
            "target_edge_type": target_edge_type,
            "graph_path": graph_path,
        }, save_path)
        print(f"Saved best model to {save_path} (best val AUC={best_val_auc:.4f})")

    test_metrics = evaluate(model, test_data, split="test", device=device_t)
    if is_main_process(rank):
        print(f"Test: loss {test_metrics['loss']:.4f} | auc {test_metrics['auc']:.4f}")
    ddp_cleanup()


def parse_edge_types(edge_types_strs: List[str]) -> List[Tuple[str, str, str]]:
    parsed: List[Tuple[str, str, str]] = []
    for s in edge_types_strs:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid edge type string: {s}. Expected 'Src,Rel,Dst'.")
        parsed.append((parts[0], parts[1], parts[2]))
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Learning (MTL) trainer: Food Origin Prediction (Sample->Food) + Nutritional Organization (contrastive)")
    parser.add_argument("--graph", type=str, default="data/hetero_graph.pt", help="Path to saved HeteroData graph")
    parser.add_argument(
        "--target-edge-type",
        type=str,
        default="Sample,Is_of_type,Food",
        help="Target edge type for Food Origin Prediction (format: Src,Rel,Dst). Default: Sample->Food.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-ratio", type=float, default=1.0, help="Negative sampling ratio")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="data/models/linkpred_gat.pt")
    parser.add_argument("--use-sampler", action="store_true", help="Use LinkNeighborLoader with negative sampling")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-triplets", type=int, default=8192)
    parser.add_argument("--contrastive-pos-threshold", type=float, default=0.5, help="Minimum similarity for positive pairs in contrastive learning")
    parser.add_argument("--contrastive-neg-threshold", type=float, default=0.1, help="Maximum similarity for negative pairs in contrastive learning")
    parser.add_argument("--link-prediction-weight", type=float, default=2.0, help="Weight for link prediction loss (higher = more emphasis)")
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="Path to pretrained encoder weights from GraphCL pretraining")

    args = parser.parse_args()

    target_et = parse_edge_types([args.target_edge_type])[0]  # Parse single edge type
    train(
        graph_path=args.graph,
        target_edge_type=target_et,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        negative_sampling_ratio=args.neg_ratio,
        device=args.device,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        save_path=args.out,
        use_sampler=args.use_sampler,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        contrastive_triplets_per_epoch=args.contrastive_triplets,
        contrastive_pos_threshold=args.contrastive_pos_threshold,
        contrastive_neg_threshold=args.contrastive_neg_threshold,
        link_prediction_weight=args.link_prediction_weight,
        pretrained_encoder_path=args.pretrained_encoder,
    )

