"""
Link prediction training for the heterogeneous food metabolomics graph.

Baseline: Hetero GAT encoder + dot-product decoder.

Notes
- Node types have no input features; we use learnable embeddings per node type.
- We train on forward canonical edge types only; reverse edges are present but
  excluded from supervision to avoid duplicates.
- Uses RandomLinkSplit for HeteroData with negative sampling.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import argparse
import json
import math

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
# Utils
# -------------------------


def to_device(data: HeteroData, device: torch.device) -> HeteroData:
    return data.to(device)


@torch.no_grad()
def evaluate(model: HeteroLinkPredModel, data: HeteroData, split: str) -> Dict[str, float]:
    model.eval()
    z_dict = model(data)
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


def attach_split_labels(data: HeteroData, split_data: Tuple[HeteroData, HeteroData, HeteroData], edge_types: List[Tuple[str, str, str]]) -> Tuple[HeteroData, HeteroData, HeteroData]:
    train_data, val_data, test_data = split_data

    def _extract_labels(src: HeteroData, dest: HeteroData):
        for et in edge_types:
            # RandomLinkSplit stores at dest[(et)].edge_label and edge_label_index
            edge_label: Tensor = dest[et].edge_label
            edge_label_index: Tensor = dest[et].edge_label_index
            # Split into pos/neg by label
            pos_mask = edge_label == 1
            neg_mask = edge_label == 0
            dest[et]["train_edge_label_index_pos" if src is train_data else ("val_edge_label_index_pos" if src is val_data else "test_edge_label_index_pos")] = edge_label_index[:, pos_mask]
            dest[et]["train_edge_label_index_neg" if src is train_data else ("val_edge_label_index_neg" if src is val_data else "test_edge_label_index_neg")] = edge_label_index[:, neg_mask]

    _extract_labels(train_data, train_data)
    _extract_labels(val_data, val_data)
    _extract_labels(test_data, test_data)
    return train_data, val_data, test_data


# -------------------------
# Training
# -------------------------


def train(
    graph_path: str,
    edge_types: List[Tuple[str, str, str]],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    negative_sampling_ratio: float = 1.0,
    device: str = "cpu",
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
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
    contrastive_topk_pos: int = 10,
    contrastive_topk_neg: int = 10,
) -> None:
    device_t = torch.device(device)
    full_data: HeteroData = torch.load(graph_path, map_location=device_t)

    # Prepare splits
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
        edge_types=edge_types,
        neg_sampling_ratio=negative_sampling_ratio,
    )
    train_data, val_data, test_data = transform(full_data)
    train_data, val_data, test_data = attach_split_labels(train_data, (train_data, val_data, test_data), edge_types)

    # Build model
    cfg = ModelConfig(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    device_t, rank, world_size, local_rank = ddp_setup()
    model = HeteroLinkPredModel(train_data.metadata(), cfg, supervised_edge_types=edge_types)
    model = ddp_wrap_model(model, device_t, world_size, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -math.inf
    best_state = None

    # Optional loaders per edge type using neighbor sampling
    loaders: Dict[Tuple[str, str, str], LinkNeighborLoader] = {}
    if use_sampler:
        for et in edge_types:
            # Concatenate pos/neg indices for training label set
            pos_idx = train_data[et]["train_edge_label_index_pos"]
            neg_idx = train_data[et]["train_edge_label_index_neg"]
            edge_label_index = torch.cat([pos_idx, neg_idx], dim=1)
            edge_label = torch.cat([
                torch.ones(pos_idx.size(1), dtype=torch.float32),
                torch.zeros(neg_idx.size(1), dtype=torch.float32),
            ], dim=0)
            loaders[et] = LinkNeighborLoader(
                train_data,
                num_neighbors=list(num_neighbors) if isinstance(num_neighbors, (list, tuple)) else [num_neighbors],
                batch_size=batch_size,
                edge_label_index=(et, edge_label_index),
                edge_label=edge_label,
                shuffle=True,
            )

    # Precompute nutrient vectors for foods (from full graph, Food->Nutrient edge_attr[:,0])
    food_nutr_vecs = _compute_food_nutrient_matrix(full_data, device_t)

    for epoch in range(1, epochs + 1):
        model.train()

        # ---- Link prediction loss (mini-batch with neighbor sampling or full-graph fallback) ----
        total_lp_loss = 0.0
        num_lp_steps = 0
        if use_sampler and loaders:
            for et, loader in loaders.items():
                for batch in loader:
                    batch = batch.to(device_t)
                    optimizer.zero_grad()
                    z_dict = model(batch)
                    edge_label_index = batch[et].edge_label_index
                    edge_label = batch[et].edge_label
                    scores = model.predict_edge_scores(z_dict, et, edge_label_index)
                    loss = F.binary_cross_entropy_with_logits(scores, edge_label)
                    loss.backward()
                    optimizer.step()
                    total_lp_loss += float(loss)
                    num_lp_steps += 1
        else:
            optimizer.zero_grad()
            z_dict = model(train_data)
            for et in edge_types:
                pos_index = train_data[et]["train_edge_label_index_pos"]
                neg_index = train_data[et]["train_edge_label_index_neg"]
                pos_scores = model.predict_edge_scores(z_dict, et, pos_index)
                neg_scores = model.predict_edge_scores(z_dict, et, neg_index)
                y_pred = torch.cat([pos_scores, neg_scores], dim=0)
                y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
                total_lp_loss = total_lp_loss + float(loss)
            loss.backward()
            optimizer.step()
            num_lp_steps = max(1, len(edge_types))

        avg_lp_loss = total_lp_loss / max(1, num_lp_steps)

        # ---- Contrastive loss on Food embeddings (one step per epoch on train split graph) ----
        optimizer.zero_grad()
        z_full = model(train_data)
        food_emb = z_full["Food"]
        triplets = _sample_food_triplets(food_nutr_vecs, contrastive_topk_pos, contrastive_topk_neg, contrastive_triplets_per_epoch)
        if triplets is not None:
            a, p, n = triplets
            triplet_loss_fn = nn.TripletMarginLoss(margin=contrastive_margin, p=2)
            closs = triplet_loss_fn(food_emb[a], food_emb[p], food_emb[n])
            (contrastive_weight * closs).backward()
            optimizer.step()
            closs_val = float(closs)
        else:
            closs_val = float('nan')

        # ---- Eval ----
        val_metrics = evaluate(model, val_data, split="val")
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
            "edge_types": edge_types,
            "graph_path": graph_path,
        }, save_path)
        print(f"Saved best model to {save_path} (best val AUC={best_val_auc:.4f})")

    test_metrics = evaluate(model, test_data, split="test")
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
    parser = argparse.ArgumentParser(description="Hetero GAT link prediction trainer")
    parser.add_argument("--graph", type=str, default="data/hetero_graph.pt", help="Path to saved HeteroData graph")
    parser.add_argument(
        "--edge-types",
        type=str,
        nargs="+",
        default=[
            "Sample,Is_of_type,Food",
            "Food,Contains,Nutrient",
            "Sample,Contains,Feature",
        ],
        help="Canonical edge types to supervise (format: Src,Rel,Dst).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-ratio", type=float, default=1.0, help="Negative sampling ratio")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="data/models/linkpred_gat.pt")
    parser.add_argument("--use-sampler", action="store_true", help="Use LinkNeighborLoader with negative sampling")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-triplets", type=int, default=8192)
    parser.add_argument("--contrastive-topk-pos", type=int, default=10)
    parser.add_argument("--contrastive-topk-neg", type=int, default=10)

    args = parser.parse_args()

    et = parse_edge_types(args.edge_types)
    train(
        graph_path=args.graph,
        edge_types=et,
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
        contrastive_topk_pos=args.contrastive_topk_pos,
        contrastive_topk_neg=args.contrastive_topk_neg,
    )

# -------------------------
# Contrastive helpers
# -------------------------


def _compute_food_nutrient_matrix(data: HeteroData, device: torch.device) -> Tensor:
    """
    Build a dense Food x Nutrient matrix from ('Food','Contains','Nutrient') edge_attr.
    Uses the first column of edge_attr as the standardized amount.
    """
    if ("Food", "Contains", "Nutrient") not in data.edge_types:
        return torch.empty((data["Food"].num_nodes, 0), device=device)
    et = ("Food", "Contains", "Nutrient")
    ei = data[et].edge_index
    if ei is None or ei.numel() == 0:
        return torch.empty((data["Food"].num_nodes, data["Nutrient"].num_nodes), device=device)
    ea = data[et].edge_attr if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None else None
    if ea is None or ea.size(1) == 0:
        # no amounts available
        return torch.zeros((data["Food"].num_nodes, data["Nutrient"].num_nodes), device=device)
    amounts = ea[:, 0].to(device)
    F = data["Food"].num_nodes
    N = data["Nutrient"].num_nodes
    mat = torch.zeros((F, N), dtype=torch.float32, device=device)
    mat.index_put_((ei[0].to(device), ei[1].to(device)), amounts, accumulate=True)
    return mat


def _sample_food_triplets(
    food_vectors: Tensor,
    topk_pos: int,
    topk_neg: int,
    num_triplets: int,
) -> Tuple[Tensor, Tensor, Tensor] | None:
    """
    Sample (anchor, positive, negative) indices for Food contrastive learning
    based on cosine similarity of nutrient vectors.
    """
    F, D = food_vectors.shape
    if F < 3 or D == 0:
        return None
    # Normalize rows
    fv = food_vectors
    norms = fv.norm(dim=1, keepdim=True).clamp_min(1e-6)
    fv_norm = fv / norms
    sim = torch.matmul(fv_norm, fv_norm.t())  # [F, F]
    # Mask self
    sim.fill_diagonal_(-1.0)

    pos_idx = torch.topk(sim, k=min(topk_pos, max(1, F - 1)), dim=1).indices  # most similar
    neg_idx = torch.topk(-sim, k=min(topk_neg, max(1, F - 1)), dim=1).indices  # least similar

    anchors = []
    positives = []
    negatives = []
    rng = torch.Generator(device=food_vectors.device)
    # Sample roughly uniform across anchors
    for a in range(F):
        if pos_idx.size(1) == 0 or neg_idx.size(1) == 0:
            continue
        p = pos_idx[a, torch.randint(low=0, high=pos_idx.size(1), size=(1,), generator=rng)].item()
        n = neg_idx[a, torch.randint(low=0, high=neg_idx.size(1), size=(1,), generator=rng)].item()
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
        if len(anchors) >= num_triplets:
            break

    if not anchors:
        return None
    return (
        torch.tensor(anchors, dtype=torch.long, device=food_vectors.device),
        torch.tensor(positives, dtype=torch.long, device=food_vectors.device),
        torch.tensor(negatives, dtype=torch.long, device=food_vectors.device),
    )


