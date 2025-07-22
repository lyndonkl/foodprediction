# Project Plan: A Graph-Based Learning Framework for Food and Metabolomic Data

This document outlines the step-by-step plan to implement a graph-based learning framework for analyzing multi-modal food and metabolomic data.

---

### Part I: Foundational Data Analysis and Graph Construction

This phase focuses on understanding the data, enriching it with external sources, and designing the core graph structure.

**1. Data Deconstruction & Analysis:**
-   [ ] **Analyze `Metadata_500food.csv`**:
    -   Identify food samples, descriptions (`description`), and common names (`sample_type_common`).
    -   Map out the food ontology (e.g., `sample_type_group1`, `botanical_family`).
    -   Extract the `ndb_number` as the primary key for linking to external nutritional data.
-   [ ] **Analyze `Untargeted_biomarkers_level5.csv`**:
    -   Map the relationship between molecular features (`feature`) and food sources (`category`).
    -   These will form the primary `('Molecule', 'found_in', 'Food')` edges.
-   [ ] **Analyze Feature Intensity Matrix**:
    -   Use the intensity values to assign weights to the `('Molecula', 'found_in', 'Food')` edges, representing the abundance of a molecule in a food.

**2. USDA FoodData Central Integration Pipeline:**
-   [ ] **Acquire Data**: Download the full USDA FoodData Central dataset (Foundation Foods & SR Legacy) in CSV format.
-   [ ] **Link Data**: For each food in our metadata:
    1.  Use `ndb_number` to find the corresponding `fdc_id` in the USDA `food.csv`.
    2.  Use the `fdc_id` to query `food_nutrient.csv` for all nutrient data.
    3.  Use `nutrient_id` to look up nutrient names and units in `nutrient.csv`.
-   [ ] **Feature Engineering**:
    -   Select a curated list of key nutrients (macro and micro).
    -   Construct a standardized (per 100g) and dense numerical feature vector for each `Food` node.
    -   Handle missing values via zero-imputation.

**3. Heterogeneous Graph Schema Design:**
-   [ ] **Define Node Types**:
    -   **`Molecule`**: One node for each unique molecular feature ID.
    -   **`Food`**: One node for each unique food sample.
-   [ ] **Define Edge Types**:
    -   `('Molecule', 'found_in', 'Food')`: Primary predictive edge.
    -   `('Food', 'contains', 'Molecule')`: Inverse edge for undirected message passing.
-   [ ] **Define Node Features**:
    -   **`Molecule` Node Features**: Generate dense vector embeddings from MS/MS spectra using **Spec2Vec**.
    -   **`Food` Node Features**: Use the nutritional vectors from the USDA integration pipeline. Concatenate with one-hot encoded ontological categories.
-   [ ] **Construct Graph**: Use a library like PyTorch Geometric to build a `HeteroData` object based on the defined schema.

---

### Part II: GNN Architectures for Learning

This phase involves selecting and implementing the GNN model.

**1. Select GNN Architecture:**
-   [ ] **Primary Recommendation (`HAN`)**:
    -   Implement a **Heterogeneous Graph Attention Network (HAN)**.
    -   **Reasoning**: Natively handles heterogeneous graphs and provides interpretability through attention weights. It's the best balance of performance and insight.
-   [ ] **Alternative for Performance (`GIN`)**:
    -   Implement a **Graph Isomorphism Network (GIN)**.
    -   **Reasoning**: Offers maximum expressive power for the food origin prediction task. Requires converting the graph to a homogeneous format first.
-   [ ] **Alternative for Embeddings (`Graph Diffusion`)**:
    -   Explore **Graph Diffusion Models**.
    -   **Reasoning**: A research-oriented approach to learn highly robust and generalizable embeddings in a self-supervised manner.

---

### Part III: Advanced Training and Evaluation Paradigms

This phase details the training methodology and how the model's success will be measured.

**1. Implement Two-Stage Training Strategy:**
-   [ ] **Stage 1: Self-Supervised Pre-training (GraphCL)**:
    -   Implement Graph Contrastive Learning (GraphCL).
    -   Create stochastic augmentations: node dropping, edge perturbation, feature masking.
    -   Train the GNN encoder on the self-supervised contrastive objective without labels to learn robust initial representations.
-   [ ] **Stage 2: Multi-Task Supervised Fine-tuning**:
    -   Initialize the GNN encoder with pre-trained weights.
    -   Add two task-specific "heads":
        1.  **Prediction Head**: A link prediction model for the `found_in` edge (Food Origin). Use **Binary Cross-Entropy Loss**.
        2.  **Organization Head**: A contrastive learning model on `Food` node embeddings to group them by nutritional similarity. Use **Triplet Loss** or **NT-Xent Loss**.
    -   Define total loss: `L_total = α * L_prediction + (1 - α) * L_contrastive`.

**2. Develop a Comprehensive Evaluation Framework:**
-   [ ] **Metrics for Food Origin Prediction**:
    -   Accuracy, Precision, Recall, F1-Score
    -   AUC-ROC
    -   Top-k Accuracy (k=3, 5)
-   [ ] **Metrics for Nutritional Organization**:
    -   **Clustering Metrics**: Silhouette Score, Davies-Bouldin Index on learned `Food` embeddings.
    -   **Qualitative Visualization**: Generate 2D t-SNE/UMAP plots of `Food` embeddings, colored by nutritional properties, to visually validate the organization.

---

### Part IV: Synthesis and Implementation Roadmap

This section provides the end-to-end workflow and final strategic choices.

**1. End-to-End Implementation Workflow:**
1.  [ ] Execute Part I: Data Preprocessing, Feature Generation (Spec2Vec), and Graph Construction.
2.  [ ] Execute Part III (Stage 1): Implement GraphCL for self-supervised pre-training of the chosen GNN (HAN).
3.  [ ] Execute Part III (Stage 2): Implement the multi-task framework for fine-tuning.
4.  [ ] Execute Part III (Evaluation): Run the comprehensive evaluation framework.
5.  [ ] **Final Application**:
    -   Develop an inference pipeline to take an unknown MS/MS spectrum, generate its embedding, and predict its food origin.
    -   Store the learned food embeddings for downstream nutritional analysis.

**2. Final Strategic Recommendation:**
-   **Primary Architecture**: **Heterogeneous Graph Attention Network (HAN)**.
-   **Training Strategy**: **Two-Stage Pre-training and Fine-tuning**. This combination offers the best path to achieving both high performance and robust, generalizable, and interpretable results. 