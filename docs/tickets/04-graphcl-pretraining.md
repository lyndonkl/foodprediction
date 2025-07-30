# Ticket 04: GraphCL Pretraining

## Phase: Model Development
**Estimated Time**: 6-8 hours  
**Dependencies**: Ticket 03

## Description
Implement GraphCL (Graph Contrastive Learning) for self-supervised pretraining of the heterogeneous graph model. This learns representations invariant to augmentations, preparing the model for multi-task fine-tuning.

## Context from Implementation Plan
- **Pretext Task**: Learn representations invariant to augmentations
- **Augmentations**: Node dropping (random Molecule/Food removal), Edge perturbation (add/remove edges), Feature masking (drop dimensions from dreaMS embeddings)
- **Contrastive Learning**: Maximize similarity between embeddings of same node from two views, minimize with others
- **Two-Stage Training**: GraphCL pretraining followed by multi-task fine-tuning
- **Objective**: Learn generalizable representations for food prediction, molecule discovery, and nutrition alignment

## Tasks

### 1. Graph Augmentation Pipeline
- [ ] Create `src/train/augmentations.py` for graph augmentations
- [ ] Implement node dropping (random Molecule/Food removal with configurable rates)
- [ ] Add edge perturbation (add/remove `contains_molecule` and `has_nutrient` edges)
- [ ] Create feature masking (drop dimensions from dreaMS embeddings and nutrient features)
- [ ] Implement augmentation pipeline with configurable parameters
- [ ] Add augmentation validation and quality checks

### 2. Contrastive Learning Implementation
- [ ] Create `src/train/graphcl.py` for contrastive learning
- [ ] Implement positive pair generation (same node, different augmented views)
- [ ] Add negative pair sampling strategies (in-batch negatives, hard negatives)
- [ ] Create contrastive loss function (InfoNCE with temperature scaling)
- [ ] Implement similarity metrics (cosine similarity, dot product)
- [ ] Add contrastive learning utilities and helpers

### 3. Pretraining Pipeline
- [ ] Create `src/train/pretrain.py` for pretraining pipeline
- [ ] Implement data loading with NeighborLoader/HGTLoader for large graphs
- [ ] Add training loop with validation and checkpointing
- [ ] Create learning rate scheduling and early stopping
- [ ] Implement training monitoring and logging
- [ ] Add gradient clipping and optimization strategies

### 4. Pretraining Utilities and Analysis
- [ ] Create `src/train/pretrain_utils.py` for utilities
- [ ] Implement embedding extraction and analysis
- [ ] Add pretraining evaluation metrics (embedding quality, clustering)
- [ ] Create visualization of learned representations
- [ ] Add pretraining configuration management
- [ ] Implement embedding similarity analysis

### 5. Validation and Monitoring
- [ ] Create `src/train/pretrain_monitoring.py` for monitoring
- [ ] Implement training loss tracking and visualization
- [ ] Add validation metrics (embedding coherence, clustering quality)
- [ ] Create tensorboard logging for training progress
- [ ] Add model checkpoint management and validation
- [ ] Implement embedding quality assessment

### 6. Pretraining Configuration
- [ ] Create `src/train/pretrain_config.py` for configuration
- [ ] Define augmentation parameters and strategies
- [ ] Add contrastive learning hyperparameters
- [ ] Create training schedule and optimization settings
- [ ] Implement configuration validation and defaults
- [ ] Add experiment tracking and reproducibility

## Acceptance Criteria
- [ ] Graph augmentations work correctly and preserve graph structure
- [ ] Contrastive learning loss decreases during training
- [ ] Pretraining pipeline runs without errors on heterogeneous graph
- [ ] Learned embeddings show meaningful structure and clustering
- [ ] Model checkpoints are saved and loadable for fine-tuning
- [ ] Training monitoring and logging work correctly
- [ ] Embeddings are ready for multi-task fine-tuning

## Notes
- Focus on stable contrastive learning for heterogeneous graphs
- Ensure augmentations preserve the semantic meaning of the graph
- Prepare embeddings for the three main objectives (prediction, discovery, nutrition)
- Consider computational efficiency for large heterogeneous graphs
- Maintain reproducibility and experiment tracking 