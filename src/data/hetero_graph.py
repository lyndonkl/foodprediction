"""
Utilities to construct a single large PyTorch Geometric HeteroData graph
from the intermediate JSON produced by `FoodMetabolomicsProcessor`.

Node types:
- Sample: one per sample (no input features; use learnable embeddings in model)
- Feature: one per unique feature id (no input features; learnable embeddings)
- Food: one per unique food type (no input features; learnable embeddings)
- Nutrient: one per unique nutrient id (no input features; learnable embeddings)

Edge types (both directions will be created):
- ("Sample", "Contains", "Feature") with edge_attr: intensity_z (float32, shape [1])
- ("Sample", "Is_of_type", "Food") with no edge_attr
- ("Food", "Contains", "Nutrient") with edge_attr: [amount_z, one_hot_unit...] (float32)

Normalization:
- Sample→Feature intensities are z-score standardized per feature id
- Food→Nutrient amounts are z-score standardized per nutrient id
  (with unit one-hot appended; no unit conversion here)

This module focuses on clean, functional construction with minimal coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import math

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


# ---------------------------
# Public API
# ---------------------------

def load_intermediate_json(json_path: str | Path = "data/intermediate_samples.json") -> Dict[str, Any]:
    """
    Load the intermediate JSON produced by the data processor.

    Args:
        json_path: Path to `intermediate_samples.json`.

    Returns:
        Parsed JSON dictionary.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Intermediate JSON not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    # Basic validation
    if "samples" not in data:
        raise ValueError("Invalid intermediate JSON: missing 'samples'")
    if not isinstance(data["samples"], list) or len(data["samples"]) == 0:
        raise ValueError("No samples found in intermediate JSON")

    return data


def build_hetero_graph(
    intermediate: Dict[str, Any],
    zscore_features: bool = True,
    zscore_nutrients_by: str = "nutrient",  # "nutrient" or "nutrient_unit"
) -> HeteroData:
    """
    Build a single large HeteroData graph from intermediate JSON.

    Args:
        intermediate: Parsed intermediate JSON.
        zscore_features: If True, z-score Sample→Feature intensities per feature id.
        zscore_nutrients_by: Grouping for nutrient z-score. One of:
            - "nutrient": z-score across foods per nutrient id (default)
            - "nutrient_unit": z-score across foods per (nutrient id, unit)

    Returns:
        HeteroData graph with nodes, edges, reverse edges, and normalized edge attrs.
    """
    samples: List[Dict[str, Any]] = intermediate["samples"]
    metadata: Dict[str, Any] = intermediate.get("metadata", {})

    # ---- Build node index mappings ----
    sample_index: Dict[int, int] = {}
    for local_idx, s in enumerate(samples):
        sample_index[int(s["id"])] = local_idx

    # Features
    feature_ids: List[int] = _collect_unique_feature_ids(samples, metadata)
    feature_index: Dict[int, int] = {fid: i for i, fid in enumerate(feature_ids)}

    # Foods
    food_names: List[str] = sorted({s.get("food_name", "unknown") for s in samples})
    food_index: Dict[str, int] = {name: i for i, name in enumerate(food_names)}

    # Nutrients
    nutrient_ids: List[int] = _collect_unique_nutrient_ids(samples, metadata)
    nutrient_index: Dict[int, int] = {nid: i for i, nid in enumerate(nutrient_ids)}

    # Units catalogue for one-hot (stable order)
    unit_values: List[str] = metadata.get("nutrients", {}).get("unique_nutrient_units", [])
    unit_values = list(unit_values) if isinstance(unit_values, list) else []
    unit_index: Dict[str, int] = {u: i for i, u in enumerate(unit_values)}
    num_unit_types: int = len(unit_values)

    # ---- Collect edges ----
    # Sample→Feature with intensity, and Sample→Food
    sf_src: List[int] = []
    sf_dst: List[int] = []
    sf_vals: List[float] = []  # raw intensities; will be z-scored per feature

    sf_group_values: Dict[int, List[float]] = {fid: [] for fid in feature_ids}
    sf_group_indices: Dict[int, List[int]] = {fid: [] for fid in feature_ids}

    sf_count = 0
    s_to_f_src: List[int] = []
    s_to_f_dst: List[int] = []

    for s in samples:
        s_idx = sample_index[int(s["id"])]
        food_name = s.get("food_name", "unknown")
        f_idx = food_index[food_name]
        s_to_f_src.append(s_idx)
        s_to_f_dst.append(f_idx)

        for feat in s.get("features", []):
            fid = int(feat["id"])
            if fid not in feature_index:
                # Skip any feature id that is not in the collected set
                continue
            sf_src.append(s_idx)
            sf_dst.append(feature_index[fid])
            intensity_val = float(feat.get("intensity", 0.0))
            sf_vals.append(intensity_val)

            # Grouping for z-score per feature
            sf_group_values[fid].append(intensity_val)
            sf_group_indices[fid].append(sf_count)
            sf_count += 1

    # Food→Nutrient with amount and unit one-hot; deduplicate per (food, nutrient)
    fn_src: List[int] = []
    fn_dst: List[int] = []
    fn_amounts: List[float] = []  # raw amounts to be z-scored
    fn_units: List[str] = []

    # Keep first occurrence per (food, nutrient)
    seen_food_nutrient: set[Tuple[str, int]] = set()

    for s in samples:
        food_name = s.get("food_name", "unknown")
        f_idx = food_index[food_name]
        for n in s.get("nutrients", []):
            nid = int(n["id"])
            key = (food_name, nid)
            if key in seen_food_nutrient:
                continue
            seen_food_nutrient.add(key)

            if nid not in nutrient_index:
                continue
            amount = float(n.get("amount", 0.0))
            unit = str(n.get("unit", ""))

            fn_src.append(f_idx)
            fn_dst.append(nutrient_index[nid])
            fn_amounts.append(amount)
            fn_units.append(unit)

    # ---- Normalize edge attributes ----
    # 1) Sample→Feature intensities: z-score per feature id
    sf_vals_norm: List[float] = list(sf_vals)
    if zscore_features and len(sf_vals_norm) > 0:
        for fid, values in sf_group_values.items():
            if not values:
                continue
            mean_v = _mean(values)
            std_v = _std(values, mean_v)
            # Apply per-feature normalization
            for pos in sf_group_indices[fid]:
                raw = sf_vals[pos]
                if std_v > 0:
                    sf_vals_norm[pos] = (raw - mean_v) / std_v
                else:
                    sf_vals_norm[pos] = 0.0

    # 2) Food→Nutrient amounts: z-score per nutrient id (or per nutrient+unit)
    if len(fn_amounts) > 0:
        if zscore_nutrients_by not in {"nutrient", "nutrient_unit"}:
            raise ValueError("zscore_nutrients_by must be 'nutrient' or 'nutrient_unit'")

        # Prepare grouping keys aligned with fn_amounts
        # We have fn_dst as nutrient indices; need original nutrient ids for grouping
        inv_nutrient_index: List[int] = [0] * len(nutrient_index)
        for nid, idx in nutrient_index.items():
            inv_nutrient_index[idx] = nid

        group_to_positions: Dict[Tuple[int, str] | Tuple[int], List[int]] = {}
        for i, (dst_idx, unit) in enumerate(zip(fn_dst, fn_units)):
            nid = inv_nutrient_index[dst_idx]
            if zscore_nutrients_by == "nutrient_unit":
                key = (nid, unit)
            else:
                key = (nid,)
            group_to_positions.setdefault(key, []).append(i)

        fn_amounts_norm: List[float] = list(fn_amounts)
        for positions in group_to_positions.values():
            values = [fn_amounts[i] for i in positions]
            mean_v = _mean(values)
            std_v = _std(values, mean_v)
            for i in positions:
                raw = fn_amounts[i]
                if std_v > 0:
                    fn_amounts_norm[i] = (raw - mean_v) / std_v
                else:
                    fn_amounts_norm[i] = 0.0
    else:
        fn_amounts_norm = []

    # One-hot encode units
    fn_unit_onehot: List[List[float]] = []
    for u in fn_units:
        vec = [0.0] * num_unit_types
        if u in unit_index:
            vec[unit_index[u]] = 1.0
        fn_unit_onehot.append(vec)

    # Final edge_attr tensors
    sf_edge_attr: Tensor | None = (
        torch.tensor(sf_vals_norm, dtype=torch.float32).unsqueeze(1)
        if len(sf_vals_norm) > 0
        else None
    )

    if len(fn_amounts_norm) > 0:
        amount_col = torch.tensor(fn_amounts_norm, dtype=torch.float32).unsqueeze(1)
        unit_oh = (
            torch.tensor(fn_unit_onehot, dtype=torch.float32)
            if num_unit_types > 0
            else torch.zeros((len(fn_amounts_norm), 0), dtype=torch.float32)
        )
        fn_edge_attr = torch.cat([amount_col, unit_oh], dim=1)
    else:
        fn_edge_attr = None

    # ---- Assemble HeteroData ----
    data = HeteroData()

    # Set num_nodes per node type (no x features; we will use embeddings in the model)
    data["Sample"].num_nodes = len(sample_index)
    data["Feature"].num_nodes = len(feature_index)
    data["Food"].num_nodes = len(food_index)
    data["Nutrient"].num_nodes = len(nutrient_index)

    # Sample→Feature
    if len(sf_src) > 0:
        data[("Sample", "Contains", "Feature")].edge_index = _to_edge_index(sf_src, sf_dst)
        if sf_edge_attr is not None:
            data[("Sample", "Contains", "Feature")].edge_attr = sf_edge_attr

    # Sample→Food
    if len(s_to_f_src) > 0:
        data[("Sample", "Is_of_type", "Food")].edge_index = _to_edge_index(s_to_f_src, s_to_f_dst)

    # Food→Nutrient
    if len(fn_src) > 0:
        data[("Food", "Contains", "Nutrient")].edge_index = _to_edge_index(fn_src, fn_dst)
        if fn_edge_attr is not None:
            data[("Food", "Contains", "Nutrient")].edge_attr = fn_edge_attr

    # ---- Add reverse edges (bi-directional) ----
    _add_reverse_edges_in_place(data)

    # ---- Persist mappings as graph metadata for convenient access after load ----
    # Build inverse sample id list ordered by internal index
    index_to_sample_id: List[int] = [0] * len(sample_index)
    for sid, idx in sample_index.items():
        index_to_sample_id[idx] = sid

    graph_metadata: Dict[str, Any] = {
        "food_names": food_names,  # index -> food name
        "food_name_to_index": food_index,
        "feature_ids": feature_ids,  # index -> feature id
        "feature_id_to_index": feature_index,
        "nutrient_ids": nutrient_ids,  # index -> nutrient id
        "nutrient_id_to_index": {str(k): v for k, v in nutrient_index.items()},
        "unit_values": unit_values,  # index -> unit name
        "unit_name_to_index": unit_index,
        "sample_ids": index_to_sample_id,  # index -> sample id
        "sample_id_to_index": sample_index,
        "zscore_nutrients_by": zscore_nutrients_by,
        "zscore_features": zscore_features,
    }
    # Attach as attribute; torch.save will pickle this along with the graph
    data.graph_metadata = graph_metadata

    return data


def build_and_save_hetero_graph(
    json_path: str | Path = "data/intermediate_samples.json",
    output_path: str | Path = "data/hetero_graph.pt",
    zscore_features: bool = True,
    zscore_nutrients_by: str = "nutrient",
    mappings_path: str | Path | None = "data/hetero_graph_mappings.json",
) -> Tuple[Path, Path | None]:
    """
    Convenience function to build the hetero graph and persist it as a PyTorch file.

    Args:
        json_path: Path to the intermediate JSON.
        output_path: Destination .pt file path.
        zscore_features: Whether to z-score intensities per feature id.
        zscore_nutrients_by: "nutrient" or "nutrient_unit".

    Returns:
        Path to the saved .pt file.
    """
    intermediate = load_intermediate_json(json_path)
    graph = build_hetero_graph(
        intermediate,
        zscore_features=zscore_features,
        zscore_nutrients_by=zscore_nutrients_by,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out_path)
    mappings_out_path: Path | None = None
    if mappings_path is not None:
        mappings_out_path = Path(mappings_path)
        mappings_out_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist human-friendly JSON for inference tooling
        with open(mappings_out_path, "w") as f:
            json.dump(graph.graph_metadata, f, indent=2)
    return out_path, mappings_out_path


# ---------------------------
# Helpers (private)
# ---------------------------

def _collect_unique_feature_ids(samples: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[int]:
    ids: List[int] = []
    md_ids = (
        metadata.get("features", {}).get("unique_feature_ids", [])
        if isinstance(metadata, dict)
        else []
    )
    if isinstance(md_ids, list) and md_ids:
        ids = [int(x) for x in md_ids]
    else:
        bag = set()
        for s in samples:
            for feat in s.get("features", []):
                bag.add(int(feat["id"]))
        ids = sorted(bag)
    return ids


def _collect_unique_nutrient_ids(samples: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[int]:
    md_unique = metadata.get("nutrients", {}).get("unique_nutrients", [])
    if isinstance(md_unique, list) and md_unique:
        try:
            return sorted(int(item["id"]) for item in md_unique)
        except Exception:
            pass
    bag = set()
    for s in samples:
        for n in s.get("nutrients", []):
            bag.add(int(n["id"]))
    return sorted(bag)


def _to_edge_index(src: List[int], dst: List[int]) -> Tensor:
    if len(src) != len(dst):
        raise ValueError("src and dst must be the same length")
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _add_reverse_edges_in_place(data: HeteroData) -> None:
    """
    Create reverse edges for each relation present in the HeteroData.
    Edge attributes are copied over unchanged.
    """
    for edge_type in list(data.edge_types):
        src_type, rel, dst_type = edge_type
        rev_rel = f"rev_{rel}"
        rev_edge_type = (dst_type, rev_rel, src_type)

        if rev_edge_type in data.edge_types:
            continue  # Already present

        edge_index = data[edge_type].edge_index
        if edge_index is None:
            continue

        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data[rev_edge_type].edge_index = rev_edge_index

        # Copy edge_attr if available
        if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
            data[rev_edge_type].edge_attr = data[edge_type].edge_attr.clone()


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float], mean_value: float) -> float:
    if not values:
        return 0.0
    var = sum((v - mean_value) ** 2 for v in values) / max(1, (len(values) - 1))
    return float(math.sqrt(var))


if __name__ == "__main__":
    # Simple CLI entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Build hetero graph from intermediate JSON")
    parser.add_argument("--json", type=str, default="data/intermediate_samples.json", help="Path to intermediate JSON")
    parser.add_argument("--out", type=str, default="data/hetero_graph.pt", help="Output path for saved graph")
    parser.add_argument(
        "--zscore-nutrients-by",
        type=str,
        default="nutrient",
        choices=["nutrient", "nutrient_unit"],
        help="Grouping for nutrient z-score normalization",
    )
    parser.add_argument("--no-zscore-features", action="store_true", help="Disable z-score for feature intensities")
    parser.add_argument("--mappings-out", type=str, default="data/hetero_graph_mappings.json", help="Path to write JSON mappings (set empty to skip)")

    args = parser.parse_args()

    mappings_out_arg: str | None = args.mappings_out if args.mappings_out else None
    out, mappings_out = build_and_save_hetero_graph(
        json_path=args.json,
        output_path=args.out,
        zscore_features=(not args.no_zscore_features),
        zscore_nutrients_by=args.zscore_nutrients_by,
        mappings_path=mappings_out_arg,
    )
    print(f"Saved hetero graph to: {out}")
    if mappings_out is not None:
        print(f"Saved mappings to: {mappings_out}")


