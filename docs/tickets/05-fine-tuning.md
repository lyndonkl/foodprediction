# Ticket 05: Multi-Task Fine-Tuning

## Phase: Model Development
**Estimated Time**: 5-7 hours  
**Dependencies**: Ticket 04

## Description
Implement multi-task fine-tuning pipeline using pretrained model for food classification and nutrition-aware embedding learning. This builds on the GraphCL pretraining to achieve the three core objectives.

## Context from Implementation Plan
- **Two-Stage Training**: Use pre-trained encoder in Multi-Task Learning (MTL) setup
- **Task 1**: Food classification (CrossEntropy loss for food prediction)
- **Task 2**: Nutritional alignment (Triplet or MSE loss for nutritional similarity)
- **Task 3**: Salient molecule discovery (attention-based importance scoring)
- **Objectives**: Food origin prediction, salient molecule discovery, nutrition-aware embeddings
- **Architecture**: Multi-task heads on top of pretrained heterogeneous GAT

## Tasks

### 1. Multi-Task Learning Setup
- [ ] Create `src/train/multitask.py` for multi-task learning framework
- [ ] Implement task-specific loss functions for all three objectives
- [ ] Add task weighting and balancing strategies (GradNorm, uncertainty weighting)
- [ ] Create gradient accumulation for multiple tasks
- [ ] Implement task-specific evaluation metrics
- [ ] Add multi-task learning utilities and helpers

### 2. Food Classification Task
- [ ] Create `src/train/food_classification.py` for classification task
- [ ] Implement CrossEntropy loss for food origin prediction
- [ ] Add classification metrics (Accuracy, Weighted F1, AUROC)
- [ ] Create confusion matrix visualization and analysis
- [ ] Implement classification head training and validation
- [ ] Add per-class performance analysis

### 3. Nutrition Alignment Task
- [ ] Create `src/train/nutrition_alignment.py` for nutrition task
- [ ] Implement triplet loss for nutritional similarity learning
- [ ] Add MSE loss for nutrient quantity prediction
- [ ] Create nutrition-aware embedding evaluation
- [ ] Implement nutritional similarity metrics and visualization
- [ ] Add nutrition embedding quality assessment

### 4. Salient Molecule Discovery Task
- [ ] Create `src/train/salient_molecules.py` for molecule discovery
- [ ] Implement attention-based molecule importance scoring
- [ ] Add molecule ranking and selection utilities
- [ ] Create molecule importance visualization
- [ ] Implement molecule discovery evaluation metrics
- [ ] Add literature cross-reference validation

### 5. Fine-Tuning Pipeline
- [ ] Create `src/train/finetune.py` for fine-tuning pipeline
- [ ] Implement pretrained model loading and initialization
- [ ] Add task-specific data loading and preprocessing
- [ ] Create fine-tuning training loop with multi-task objectives
- [ ] Implement validation and early stopping for all tasks
- [ ] Add model checkpointing and saving
- [ ] Create fine-tuning configuration management

### 6. Multi-Task Evaluation and Monitoring
- [ ] Create `src/train/finetune_monitoring.py` for monitoring
- [ ] Implement multi-task loss tracking and visualization
- [ ] Add task-specific metric logging and analysis
- [ ] Create tensorboard integration for all tasks
- [ ] Add model performance visualization
- [ ] Implement task balancing analysis

## Acceptance Criteria
- [ ] Multi-task learning pipeline works correctly for all three objectives
- [ ] Food classification achieves reasonable accuracy and performance
- [ ] Nutrition alignment shows meaningful nutritional similarity patterns
- [ ] Salient molecule discovery identifies important molecules
- [ ] Fine-tuning improves over pretrained model for all tasks
- [ ] All metrics are tracked and logged for each task
- [ ] Model checkpoints are saved and loadable with all task heads

## Notes
- Focus on balancing the three tasks effectively
- Ensure each task contributes meaningfully to the overall learning
- Prepare for comprehensive evaluation of all three objectives
- Consider task-specific learning rates and optimization strategies
- Maintain interpretability for salient molecule discovery 