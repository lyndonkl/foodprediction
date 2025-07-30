# Ticket 03: Basic GNN Model

## Phase: Model Development
**Estimated Time**: 4-6 hours  
**Dependencies**: Ticket 02

## Description
Implement the heterogeneous GAT model architecture using PyTorch Geometric, focusing on the core GNN components for food origin prediction, salient molecule discovery, and nutrition-aware embeddings.

## Context from Implementation Plan
- **Base Model**: Heterogeneous GAT using `HeteroConv` with distinct `GATConv` for each edge type
- **Attention Mechanism**: Built-in attention scores enable intrinsic interpretability
- **Edge Attributes**: Handle edge attributes via `edge_dim` parameter
- **Node Types**: Molecule, Food, Sample, Nutrient with different feature dimensions
- **Objectives**: Food classification, molecule importance, nutritional similarity

## Tasks

### 1. Heterogeneous GAT Architecture
- [ ] Create `src/models/hetero_gat.py` for the main heterogeneous GAT model
- [ ] Implement `HeteroConv` with distinct `GATConv` for each edge type:
  - `(Sample) --contains_molecule--> (Molecule)`
  - `(Sample) --is_instance_of--> (Food)`
  - `(Food) --has_nutrient--> (Nutrient)`
  - `(Molecule) --is_similar_to--> (Molecule)` (optional)
- [ ] Add edge attribute handling via `edge_dim` parameter
- [ ] Implement attention mechanism for intrinsic interpretability
- [ ] Create model configuration and hyperparameter management

### 2. Node Type Encoders
- [ ] Create `src/models/encoders.py` for node-specific encoders
- [ ] Implement molecule encoder (dreaMS embeddings with learnable projection)
- [ ] Create food encoder (nutritional features with normalization)
- [ ] Add sample encoder (learnable embeddings or zero initialization)
- [ ] Implement nutrient encoder (learnable embeddings)
- [ ] Add feature dimension alignment and validation

### 3. Multi-Task Learning Heads
- [ ] Create `src/models/heads.py` for task-specific heads
- [ ] Implement food classification head (CrossEntropy loss for food prediction)
- [ ] Create nutrition alignment head (projection layer for nutritional similarity)
- [ ] Add salient molecule discovery head (attention-based importance scoring)
- [ ] Implement multi-task learning setup with task weighting
- [ ] Add attention weight extraction for explainability

### 4. Model Utilities and Attention Analysis
- [ ] Create `src/models/utils.py` for model utilities
- [ ] Implement model initialization functions with proper weight initialization
- [ ] Add model checkpointing and loading utilities
- [ ] Create attention weight extraction and visualization
- [ ] Implement model summary and statistics
- [ ] Add attention analysis for salient molecule discovery

### 5. Model Configuration and Factory
- [ ] Create `src/models/config.py` for model configuration
- [ ] Define model hyperparameters for each component
- [ ] Add model architecture validation
- [ ] Create model factory functions for different configurations
- [ ] Implement model parameter counting and memory estimation

### 6. Forward Pass and Training Setup
- [ ] Create `src/models/forward.py` for forward pass implementation
- [ ] Implement heterogeneous graph forward pass
- [ ] Add multi-task output generation
- [ ] Create attention weight collection for analysis
- [ ] Implement gradient flow validation
- [ ] Add model debugging utilities

## Acceptance Criteria
- [ ] Heterogeneous GAT model implements correctly with all edge types
- [ ] All node types have appropriate encoders with correct feature dimensions
- [ ] Model can handle edge attributes (intensity, quantity) properly
- [ ] Attention weights are accessible for explainability and salient molecule discovery
- [ ] Model can be saved/loaded successfully with all components
- [ ] Multi-task heads work correctly for all three objectives
- [ ] Forward pass generates outputs for food classification and nutrition alignment

## Notes
- Focus on the specific heterogeneous GAT architecture from the implementation plan
- Ensure attention mechanism provides interpretability for salient molecule discovery
- Prepare for GraphCL pretraining and multi-task fine-tuning
- Consider memory efficiency for large heterogeneous graphs
- Maintain clean interfaces for the three main objectives 