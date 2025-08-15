# Source Overview

This folder contains the end-to-end pipeline to build a single heterogeneous graph and train heterogeneous GNNs for food metabolomics.

## Layout

- `data/`
  - `processor.py`: build `data/intermediate_samples.json` from CSVs
  - `hetero_graph.py`: construct and save `data/hetero_graph.pt` and index mappings
  - `generate_intermediate.py`: CLI to run the processor
- `models/`
  - `configs.py`: `ModelConfig`, `PretrainConfig`
  - `encoders.py`: `HeteroGATEncoder`
  - `decoders.py`: `DotProductDecoder`
  - `hetero_gat.py`: `HeteroLinkPredModel` wrapper
- `train/`
  - `pretrain_graphcl.py`: GraphCL-style self-supervised pretraining
  - `link_prediction.py`: fine-tuning for link prediction (+ sampler + food-contrastive)
  - `ddp_utils.py`: DDP setup for CUDA/MPS/CPU
- `utils/`: general utilities

## Graph schema (what the model consumes)
- Node types: `Sample`, `Feature`, `Food`, `Nutrient` (no input features; use learnable embeddings)
- Edge types (reverse edges auto-added):
  - `('Sample','Contains','Feature')` with edge_attr: intensity z-score per feature
  - `('Sample','Is_of_type','Food')` with no edge_attr
  - `('Food','Contains','Nutrient')` with edge_attr: [amount z-score, unit one-hot]

## Data build (quick)
1) Generate intermediate JSON:
```bash
python -m src.data.generate_intermediate
# -> data/intermediate_samples.json
```
2) Build hetero graph with normalization and mappings:
```bash
python -m src.data.hetero_graph \
  --json data/intermediate_samples.json \
  --out data/hetero_graph.pt \
  --zscore-nutrients-by nutrient_unit
# -> data/hetero_graph.pt, data/hetero_graph_mappings.json
```

## Model structure
- Encoder: `HeteroGATEncoder` (per-edge-type `GATConv` in a `HeteroConv`), learnable node-type embeddings initialized per node ID.
- Decoders: dot-product per supervised relation for link prediction.
- Wrapper: `HeteroLinkPredModel` bundles encoder + decoders.

## Two-stage training

### 1) Self-supervised pretraining (GraphCL)
- Two stochastic graph views per batch via:
  - Node drop (Food/Feature), edge drop per relation, edge-attr masking
- Objective: NT-Xent (InfoNCE) aligns embeddings across the two views for selected node types (default: Food, Sample)
- Run:
```bash
python -m src.train.pretrain_graphcl \
  --graph data/hetero_graph.pt \
  --epochs 100 --lr 1e-3 --temperature 0.2 \
  --drop-food 0.1 --drop-feature 0.1 \
  --drop-sample-feature-edges 0.1 --drop-food-nutrient-edges 0.1 \
  --mask-intensity 0.1 --mask-nutrient-amount 0.1 \
  --loss-node-types Food Sample
# -> data/models/pretrain_graphcl_encoder.pt
```

### 2) Fine-tuning for link prediction (with sampler and auxiliary contrastive)
- Supervise on canonical relations via BCE (positive/negative sampling)
- Use `LinkNeighborLoader` mini-batches with neighbor sampling
- Auxiliary objective: triplet loss on `Food` embeddings using nutrient-vector similarity (cosine) for positives/negatives
- Run:
```bash
python -m src.train.link_prediction \
  --graph data/hetero_graph.pt \
  --edge-types "Sample,Is_of_type,Food" "Food,Contains,Nutrient" "Sample,Contains,Feature" \
  --use-sampler --batch-size 2048 --num-neighbors 10 10 \
  --contrastive-weight 1.0 --contrastive-margin 0.2 --contrastive-triplets 8192 \
  --epochs 50 --lr 1e-3
# -> data/models/linkpred_gat.pt
```

Notes:
- Pretrained encoder weights are saved separately for initialization; loading helper can be added to warm-start fine-tuning.

## Distributed training (DDP)
- Utilities automatically select backend: CUDA→`nccl`; MPS/CPU→`gloo` (per best practices). See: PyTorch in One Hour by Sebastian Raschka.
- Launch on multiple processes using `torchrun`:
```bash
# Pretrain
export PYTORCH_ENABLE_MPS_FALLBACK=1
torchrun -m src.train.pretrain_graphcl \
  --graph data/hetero_graph.pt \
  --epochs 200 \
  --device "cpu" \
  --lr 1e-3 \
  --temperature 0.2 \
  --drop-food 0.1 \
  --drop-feature 0.1 \
  --drop-sample-feature-edges 0.1 \
  --drop-food-nutrient-edges 0.1 \
  --mask-intensity 0.1 \
  --mask-nutrient-amount 0.1
  
torchrun --nproc_per_node=2 -m src.train.pretrain_graphcl --graph data/hetero_graph.pt

# Fine-tune
torchrun --nproc_per_node=2 -m src.train.link_prediction --graph data/hetero_graph.pt --use-sampler
```
- Rank-0 handles printing and checkpointing. Cleanup is automatic at end.

## Environment
Use the project root `environment.yml` (Python 3.11, torch 2.3.1, PyG wheels via pip on macOS/CPU). Example:
```bash
conda env create -f environment.yml
conda activate foodprediction
```

## References
- Link prediction training paradigm and methodology: [Colab notebook](https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing)
- DDP guidance (CPU/MPS/CUDA backends): [PyTorch in One Hour – Sebastian Raschka](https://sebastianraschka.com/teaching/pytorch-1h/)
