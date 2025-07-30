# Food Metabolomics Graph Learning

A Graph Neural Network (GNN) project for food metabolomics analysis, focusing on three core objectives:

1. **Food Origin Prediction** – Identify the origin food of an unknown MS/MS spectrum sample
2. **Salient Molecule Discovery** – Identify the most chemically informative molecules for classification
3. **Nutrition-Aware Embeddings** – Learn embeddings of food types that reflect nutritional similarity

## Project Overview

This project applies heterogeneous Graph Neural Networks to a rich, multi-modal food metabolomics dataset. The system uses a heterogeneous graph to model relationships among:

- **Molecules**: Unique MS/MS features with dreaMS embeddings
- **Foods**: Distinct food categories with USDA nutritional data
- **Samples**: Biological replicates with learnable features
- **Nutrients**: Explicit nutrient concepts with learnable embeddings

## Architecture

### Graph Schema
- **Node Types**: Molecule, Food, Sample, Nutrient
- **Edge Types**: 
  - `(Sample) --contains_molecule--> (Molecule)` with intensity
  - `(Sample) --is_instance_of--> (Food)` (ground truth)
  - `(Food) --has_nutrient--> (Nutrient)` with quantity
  - `(Molecule) --is_similar_to--> (Molecule)` (optional)

### Model Architecture
- **Base Model**: Heterogeneous GAT with attention for interpretability
- **Training**: Two-stage approach (GraphCL pretraining + multi-task fine-tuning)
- **Features**: dreaMS embeddings for molecules, nutritional data for foods

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd foodprediction
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate foodprediction
   ```

3. **Install dreaMS** (if not already installed):
   ```bash
   pip install dreams-embeddings
   ```

## Project Structure

```
foodprediction/
├── src/
│   ├── models/          # GNN and heads (GATConv layers + classifier + projection)
│   ├── data/            # Graph construction logic (single graph)
│   ├── train/           # Training pipeline (pretraining + fine-tuning stages)
│   ├── eval/            # Metrics, explanations (faithfulness, stability, k-NN purity)
│   └── utils/           # Utility functions
├── notebooks/           # EDA, visualization, experimentation
├── data/
│   ├── raw/             # Raw metabolomics and nutritional data
│   ├── processed/       # Processed data and features
│   └── embeddings/      # dreaMS embeddings cache
├── configs/             # Hyperparameters and configuration
└── docs/                # Documentation and implementation plan
```

## Quick Start

1. **Setup the environment** (see Installation above)

2. **Prepare your data**:
   - Place MS/MS spectra in `data/raw/`
   - Add nutritional data for foods
   - Configure data paths in `configs/default.yaml`

3. **Run the pipeline**:
   ```python
   from src.data import build_graph
   from src.train import pretrain, finetune
   from src.eval import evaluate
   
   # Build heterogeneous graph
   graph = build_graph()
   
   # Pretrain with GraphCL
   pretrained_model = pretrain(graph)
   
   # Fine-tune for multi-task learning
   model = finetune(pretrained_model, graph)
   
   # Evaluate all objectives
   results = evaluate(model, graph)
   ```

## Configuration

Edit `configs/default.yaml` to customize:
- Model hyperparameters (hidden dimensions, layers, attention heads)
- Training parameters (learning rates, batch sizes, epochs)
- Data processing settings (augmentation rates, feature dimensions)
- Evaluation metrics and thresholds

## Core Objectives

### 1. Food Origin Prediction
Predict the food source of unknown MS/MS spectra using the heterogeneous graph structure and attention mechanisms.

### 2. Salient Molecule Discovery
Identify the most important molecules for food classification using GAT attention weights and explainability techniques.

### 3. Nutrition-Aware Embeddings
Learn food embeddings that capture nutritional similarity, enabling nutrition-based food recommendations and analysis.

## Implementation Plan

The project follows a structured implementation approach with tickets for each phase:
- **Phase 1**: Foundation (Project Setup, Data Processing, Graph Construction)
- **Phase 2**: Model Development (GNN Model, GraphCL Pretraining, Multi-Task Fine-tuning)
- **Phase 3**: Evaluation & Inference (Evaluation Metrics, Inference Pipeline)

See `docs/tickets/` for detailed implementation tickets.

## Key Features

- **Heterogeneous Graph**: Models complex relationships between molecules, foods, samples, and nutrients
- **dreaMS Integration**: Uses state-of-the-art molecule embeddings
- **Attention Mechanism**: Provides interpretability for salient molecule discovery
- **Multi-Task Learning**: Simultaneously optimizes for all three objectives
- **GraphCL Pretraining**: Self-supervised learning for better representations
- **Dynamic Inference**: Real-time processing of new metabolomics data

## License

[Add your license information here] 