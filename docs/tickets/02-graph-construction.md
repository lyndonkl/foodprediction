# Ticket 02: Graph Construction

## Phase: Foundation
**Estimated Time**: 6-8 hours  
**Dependencies**: Ticket 01

## Description
Implement heterogeneous graph construction from processed metabolomics and nutritional data, creating the core data structure for GNN training. This builds the single large graph that models relationships among Molecules, Foods, Samples, and Nutrients.

## Context from Implementation Plan
- **Node Types**: Molecule (dreaMS embeddings), Food (USDA nutritional data), Sample (learnable), Nutrient (learnable)
- **Edge Types**: 
  - `(Sample) --contains_molecule--> (Molecule)` with intensity
  - `(Sample) --is_instance_of--> (Food)` (ground truth)
  - `(Food) --has_nutrient--> (Nutrient)` with quantity
  - `(Molecule) --is_similar_to--> (Molecule)` (optional)
- **Architecture**: Single large graph with dynamic node/edge additions
- **PyG Integration**: Use NeighborLoader/HGTLoader for minibatch training

## Tasks

### 1. Graph Schema Implementation
- [ ] Create `src/data/graph_schema.py` defining the heterogeneous graph structure
- [ ] Implement node type definitions (Molecule, Food, Sample, Nutrient)
- [ ] Define edge type specifications with attributes (intensity, quantity)
- [ ] Create graph metadata and statistics utilities
- [ ] Implement schema validation for graph integrity

### 2. Node Feature Engineering
- [ ] Create `src/data/node_features.py` for feature generation
- [ ] Implement molecule node features using dreaMS embeddings
- [ ] Create food node features from concatenated nutritional data
- [ ] Add sample node features (learnable or zero-initialized)
- [ ] Implement nutrient node features (learnable embeddings)
- [ ] Add feature normalization and validation

### 3. Edge Construction
- [ ] Create `src/data/edge_construction.py` for edge creation
- [ ] Implement `(Sample) --contains_molecule--> (Molecule)` edges with intensity attributes
- [ ] Create `(Sample) --is_instance_of--> (Food)` edges (ground truth for training)
- [ ] Implement `(Food) --has_nutrient--> (Nutrient)` edges with quantity attributes
- [ ] Add optional `(Molecule) --is_similar_to--> (Molecule)` edges for chemical similarity
- [ ] Implement edge attribute normalization and validation

### 4. PyG Heterogeneous Graph Creation
- [ ] Create `src/data/graph_builder.py` for PyG heterogeneous graph construction
- [ ] Implement single large graph creation using PyTorch Geometric
- [ ] Add graph validation and statistics computation
- [ ] Create graph serialization/deserialization utilities
- [ ] Implement dynamic node/edge addition for inference time
- [ ] Add graph partitioning utilities for large datasets

### 5. Graph Utilities and Visualization
- [ ] Create `src/data/graph_utils.py` for graph operations
- [ ] Implement graph visualization utilities for heterogeneous graphs
- [ ] Add graph statistics and analysis functions
- [ ] Create graph sampling utilities for training
- [ ] Implement graph integrity checks and validation
- [ ] Add graph metadata and documentation

### 6. Training Data Preparation
- [ ] Create `src/data/training_data.py` for training data preparation
- [ ] Implement NeighborLoader/HGTLoader setup for minibatch training
- [ ] Add data loading utilities for the three objectives
- [ ] Create train/validation/test splits for heterogeneous graph
- [ ] Implement data augmentation for GraphCL pretraining

## Acceptance Criteria
- [ ] Heterogeneous graph constructed successfully with all node types
- [ ] All node types have appropriate features (dreaMS, nutritional, learnable)
- [ ] All edge types created with correct attributes (intensity, quantity)
- [ ] Graph can be loaded and validated with PyTorch Geometric
- [ ] Graph statistics and visualization work correctly
- [ ] Dynamic node/edge addition works for inference
- [ ] Training data loaders are functional for minibatch training

## Notes
- Focus on the specific heterogeneous graph schema from the implementation plan
- Ensure proper integration with PyTorch Geometric for large graph handling
- Prepare for GraphCL pretraining and multi-task fine-tuning
- Consider memory efficiency for large metabolomics datasets
- Maintain graph integrity for the three main objectives 