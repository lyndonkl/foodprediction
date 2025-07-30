# AI Coding Agent Implementation Plan: Food Metabolomics Graph Learning

## Project Overview

This project applies Graph Neural Networks (GNNs) to a rich, multi-modal food metabolomics dataset to accomplish three core objectives:

1. **Food Origin Prediction** – Identify the origin food of an unknown MS/MS spectrum sample.
2. **Salient Molecule Discovery** – Identify the most chemically informative molecules for classification.
3. **Nutrition-Aware Embeddings** – Learn embeddings of food types that reflect nutritional similarity.

The system will use a heterogeneous graph to model relationships among Molecules, Foods, Samples, and Nutrients.

---

## Graph Schema Design

### Node Types

* **Molecule**: Unique MS/MS features. Feature vector: dreaMS embeddings.
* **Food**: Distinct food category. Feature vector: normalized USDA nutritional data.
* **Sample**: Biological replicates. Feature vector: null or learnable.
* **Nutrient**: Explicit nutrient concept. Feature vector: learnable or ontology-derived.

### Edge Types

* `(Sample) --contains_molecule--> (Molecule)` with intensity as edge attribute
* `(Sample) --is_instance_of--> (Food)` (ground truth for training)
* `(Food) --has_nutrient--> (Nutrient)` with quantity as edge attribute
* `(Molecule) --is_similar_to--> (Molecule)` (optional chemical similarity edge)

The graph is constructed as a **single large graph** with dynamic node/edge additions at inference time.

---

## Feature Engineering

* **Molecule Node Features**: Use `dreaMS` embeddings instead of Spec2Vec. Refer to: [dreaMS Docs](https://dreams-docs.readthedocs.io/en/latest/).
* **Food Node Features**: Concatenate USDA nutrient values.
* **Nutrient Node Features**: Learnable embeddings or use nutritional ontology.
* **Sample Node Features**: Initialized to zeros or learned.
* **Edge Features**:

  * `contains_molecule`: peak intensity (scalar)
  * `has_nutrient`: quantity (scalar)

---

## Model Architecture

### Base Model: **Heterogeneous GAT**

* Use `HeteroConv` with distinct `GATConv` for each edge type.
* Handles edge attributes via `edge_dim`.
* Built-in attention scores enable intrinsic interpretability.

### Advanced Option (Future Work): **Heterogeneous Graph Transformer (HGT)**

* For long-range dependencies and complex meta-relations.
* High computational cost.

---

## Mechanism of Action: Self-Supervised Pretraining with GraphCL

### Pretext Task

* Learn representations invariant to augmentations:

  * **Node Dropping**: Random Molecule/Food removal
  * **Edge Perturbation**: Add/remove `found_in` edges
  * **Feature Masking**: Drop dimensions from `dreaMS` or nutrient features

### Contrastive Learning Objective

* Maximize similarity between embeddings of same node from two views
* Minimize similarity with all others in batch (negative pairs)

### Two-Stage Training

1. **Pretraining**: Use GraphCL on the large heterogeneous graph
2. **Fine-Tuning**: Use pre-trained encoder in a Multi-Task Learning (MTL) setup:

   * Task 1: Food classification (CrossEntropy)
   * Task 2: Nutritional alignment (Triplet or MSE loss)

---

## Dataset and Code Structure

### Followed Conventions

* Inspired by: [https://github.com/lyndonkl/cographnet](https://github.com/lyndonkl/cographnet)
* Organize code into modular folders:

  * `src/models/`: GNN and heads (GATConv layers + classifier + projection)
  * `src/data/`: Graph construction logic (single graph)
  * `src/train/`: Training pipeline (including pretraining + fine-tuning stages)
  * `src/eval/`: Metrics, explanations (faithfulness, stability, k-NN purity, etc.)
  * `notebooks/`: EDA, visualization

### PyG Reference

* For single large graph handling: [PyG large graph training](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)
* Use `NeighborLoader` or `HGTLoader` for minibatch training

---

## Explainability Strategy

* **Intrinsic**: GAT attention weights
* **Post-hoc**: `GNNExplainer`
* **Validation**:

  * Fidelity+ / Fidelity-
  * Stability (Jaccard/Cosine between perturbed explanation masks)
  * Literature cross-reference for known biomarkers

---

## Evaluation Metrics

### Food Prediction

* Accuracy, Weighted F1, AUROC

### Salient Molecule Discovery

* Fidelity+, Fidelity-, Stability, Qualitative validation

### Nutrition-Aware Embedding

* k-NN Purity (local coherence)
* Embedding-Nutrient Correlation (Spearman rank)

---

## Final Notes for Agent

* Use dreaMS for molecule embeddings.
* Treat entire dataset as **one large heterogeneous graph**.
* Incorporate **GraphCL pretraining**, followed by **multi-task fine-tuning**.
* Focus on modular, scalable code (see cographnet for inspiration but improve clarity).

---

## References

* [dreaMS embeddings](https://dreams-docs.readthedocs.io/en/latest/)
* [cographnet](https://github.com/lyndonkl/cographnet)
* [PyG Colabs](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)
* [GraphCL paper](https://arxiv.org/abs/2010.13902) 