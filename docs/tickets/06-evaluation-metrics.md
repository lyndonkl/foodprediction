# Ticket 06: Evaluation Metrics

## Phase: Evaluation
**Estimated Time**: 4-6 hours  
**Dependencies**: Ticket 05

## Description
Implement comprehensive evaluation metrics for all three project objectives: food prediction, salient molecule discovery, and nutrition-aware embeddings. This provides the evaluation framework to assess model performance across all objectives.

## Context from Implementation Plan
- **Food Prediction**: Accuracy, Weighted F1, AUROC
- **Salient Molecule Discovery**: Fidelity+, Fidelity-, Stability, Qualitative validation
- **Nutrition-Aware Embedding**: k-NN Purity (local coherence), Embedding-Nutrient Correlation (Spearman rank)
- **Explainability**: GAT attention weights, GNNExplainer, faithfulness and stability metrics
- **Validation**: Literature cross-reference for known biomarkers

## Tasks

### 1. Food Prediction Metrics
- [ ] Create `src/eval/food_prediction.py` for classification metrics
- [ ] Implement Accuracy, Weighted F1, AUROC for food origin prediction
- [ ] Add confusion matrix analysis and visualization
- [ ] Create per-class performance metrics and analysis
- [ ] Implement statistical significance testing for model comparisons
- [ ] Add classification confidence analysis

### 2. Salient Molecule Discovery Evaluation
- [ ] Create `src/eval/salient_molecules.py` for molecule analysis
- [ ] Implement Fidelity+ and Fidelity- metrics for molecule importance
- [ ] Add stability analysis (Jaccard/Cosine similarity between explanations)
- [ ] Create molecule importance ranking and selection
- [ ] Implement literature cross-reference validation for known biomarkers
- [ ] Add molecule importance visualization and analysis

### 3. Nutrition-Aware Embedding Evaluation
- [ ] Create `src/eval/nutrition_embeddings.py` for embedding analysis
- [ ] Implement k-NN purity for local coherence assessment
- [ ] Add embedding-nutrient correlation (Spearman rank correlation)
- [ ] Create nutritional similarity visualization and analysis
- [ ] Implement embedding quality metrics and validation
- [ ] Add nutrition-aware clustering analysis

### 4. Model Explainability Analysis
- [ ] Create `src/eval/explainability.py` for model interpretation
- [ ] Implement GAT attention weight analysis and visualization
- [ ] Add GNNExplainer integration for post-hoc explanations
- [ ] Create attention visualization utilities for salient molecules
- [ ] Implement faithfulness and stability metrics for explanations
- [ ] Add explanation quality assessment

### 5. Comprehensive Evaluation Pipeline
- [ ] Create `src/eval/evaluation_pipeline.py` for unified evaluation
- [ ] Implement automated evaluation workflow for all three objectives
- [ ] Add result aggregation and reporting utilities
- [ ] Create evaluation result visualization and analysis
- [ ] Add evaluation configuration management
- [ ] Implement evaluation reproducibility utilities

### 6. Evaluation Utilities and Reporting
- [ ] Create `src/eval/utils.py` for evaluation utilities
- [ ] Implement metric computation helpers and validation
- [ ] Add evaluation result storage and loading
- [ ] Create evaluation report generation
- [ ] Add evaluation visualization utilities
- [ ] Implement evaluation comparison tools

## Acceptance Criteria
- [ ] All evaluation metrics are implemented and tested for the three objectives
- [ ] Evaluation pipeline runs automatically and produces comprehensive reports
- [ ] Results are clearly visualized and analyzed for each objective
- [ ] Statistical significance is properly assessed for model comparisons
- [ ] Explainability analysis provides insights for salient molecule discovery
- [ ] Evaluation results are reproducible and well-documented

## Notes
- Focus on interpretable and actionable metrics for each objective
- Ensure statistical rigor in evaluations and comparisons
- Add comprehensive visualization capabilities for all metrics
- Consider computational efficiency for large evaluation datasets
- Maintain reproducibility and documentation for all evaluation procedures 