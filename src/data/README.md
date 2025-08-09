# Food Metabolomics Data Processing

This folder builds the dataset for a single large heterogeneous graph used for training.

## Components

- `processor.py`: Loads CSVs and produces `data/intermediate_samples.json` (samples, features, foods, nutrients + metadata).
- `generate_intermediate.py`: CLI to run the processor.
- `hetero_graph.py`: Builds a PyTorch Geometric `HeteroData` graph and saves graph + index mappings.
- `__init__.py`: Exposes a small public API for convenient imports.

## Environment

Use the project root `environment.yml` (Python 3.11, torch 2.3.1 with PyG wheels). From repo root:
```bash
conda env create -f environment.yml
conda activate foodprediction
```

## Required data files (under `data/`)

- `Metadata_500food.csv`
- `featuretable_reformated - Kushal.csv`
- Optional directory `FoodData_Central_csv_2025-04-24/` containing: `nutrient.csv`, `food_nutrient.csv`, `sr_legacy_food.csv` (for Food→Nutrient edges)

## Build the dataset

### 1) Generate intermediate JSON
```bash
python -m src.data.generate_intermediate
# Outputs: data/intermediate_samples.json
```

### 2) Construct the hetero graph (with z-score normalization)
```bash
python -m src.data.hetero_graph \
  --json data/intermediate_samples.json \
  --out data/hetero_graph.pt \
  --zscore-nutrients-by nutrient_unit

# Outputs:
# - data/hetero_graph.pt                (PyG HeteroData, ready for training)
# - data/hetero_graph_mappings.json     (indices for foods, features, nutrients, units, samples)
```

## Programmatic usage
```python
from src.data import (
    FoodMetabolomicsProcessor,
    load_intermediate_json,
    build_hetero_graph,
    build_and_save_hetero_graph,
)

# Generate intermediate JSON
processor = FoodMetabolomicsProcessor(data_dir="data")
processor.generate_intermediate_format("data/intermediate_samples.json")

# Build and save hetero graph
build_and_save_hetero_graph(
    json_path="data/intermediate_samples.json",
    output_path="data/hetero_graph.pt",
    zscore_features=True,
    zscore_nutrients_by="nutrient_unit",
)
```

## Intermediate JSON structure (simplified)
```json
{
  "samples": [
    {
      "id": 0,
      "food_name": "apple",
      "features": [{ "id": 123, "intensity": 456.7 }, ...],
      "nutrients": [{ "id": 1003, "name": "Protein", "amount": 0.3, "unit": "g" }, ...]
    }
  ],
  "metadata": {
    "features": { "unique_feature_ids": [1, 2, ...] },
    "nutrients": {
      "unique_nutrient_units": ["g", "mg", ...],
      "unique_nutrients": [{ "id": 1003, "name": "Protein" }, ...]
    }
  }
}
```

## Normalization

- `('Sample','Contains','Feature')`: intensity is z-scored per feature id across samples.
- `('Food','Contains','Nutrient')`: amount is z-scored per nutrient (or per nutrient+unit when `--zscore-nutrients-by nutrient_unit`), then unit one-hot is appended to edge_attr.

## Notes

- Reverse edges are added automatically so information propagates both ways during message passing.
- Index mappings are saved both into the graph (`graph.graph_metadata`) and as JSON for inference.