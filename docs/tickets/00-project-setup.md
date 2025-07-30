# Ticket 00: Project Setup

## Phase: Foundation
**Estimated Time**: 2-3 hours  
**Dependencies**: None

## Description
Set up the foundational project structure and environment for the Food Metabolomics Graph Learning project. This establishes the basic infrastructure needed to implement the heterogeneous graph neural network for food origin prediction, salient molecule discovery, and nutrition-aware embeddings.

## Context from Implementation Plan
- **Graph Schema**: Heterogeneous graph with Molecules, Foods, Samples, and Nutrients
- **Key Libraries**: PyTorch Geometric for GNNs, dreaMS for molecule embeddings
- **Architecture**: Single large graph with dynamic node/edge additions
- **Training**: Two-stage approach (GraphCL pretraining + multi-task fine-tuning)

## Tasks

### 1. Environment Setup
- [ ] Create `requirements.txt` with core dependencies:
  - PyTorch Geometric (for heterogeneous GNNs)
  - dreaMS (for molecule embeddings)
  - NumPy, Pandas, SciPy (data processing)
  - Matplotlib, Seaborn (visualization)
  - Jupyter (for notebooks and experimentation)

### 2. Project Structure
- [ ] Create directory structure following the implementation plan:
  ```
  src/
  ├── models/          # GNN and heads (GATConv layers + classifier + projection)
  ├── data/            # Graph construction logic (single graph)
  ├── train/           # Training pipeline (pretraining + fine-tuning stages)
  ├── eval/            # Metrics, explanations (faithfulness, stability, k-NN purity)
  └── utils/           # Utility functions
  notebooks/           # EDA, visualization, experimentation
  data/
  ├── raw/             # Raw metabolomics and nutritional data
  ├── processed/       # Processed data and features
  └── embeddings/      # dreaMS embeddings cache
  configs/             # Hyperparameters and configuration
  ```

### 3. Configuration Management
- [ ] Create `configs/default.yaml` for model hyperparameters
- [ ] Set up basic logging for training monitoring
- [ ] Create `.gitignore` for Python/data science projects

### 4. Basic Documentation
- [ ] Update `README.md` with project overview and objectives
- [ ] Create `docs/setup.md` with installation instructions
- [ ] Document the three core objectives:
  - Food Origin Prediction
  - Salient Molecule Discovery  
  - Nutrition-Aware Embeddings

## Acceptance Criteria
- [ ] All dependencies install without conflicts
- [ ] Project structure follows the implementation plan
- [ ] Basic configuration system is functional
- [ ] Can import all core modules without errors
- [ ] Ready to begin data processing and graph construction

## Notes
- Focus on getting the environment ready for heterogeneous graph construction
- Ensure dreaMS integration is properly set up for molecule embeddings
- Prepare for single large graph handling with PyTorch Geometric
- Keep structure simple but extensible for the three main objectives 