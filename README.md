# Food Prediction: Graph-Based Learning Framework

A graph-based learning framework for analyzing multi-modal food and metabolomic data using Graph Neural Networks (GNNs).

## Project Overview

This project implements a heterogeneous graph-based machine learning system to:
1. **Predict food origin** from unknown MS/MS spectral fingerprints
2. **Organize food embeddings** by nutritional similarity for downstream analysis

The system leverages the relationships between molecular features (metabolites) and food samples to create a rich, interconnected graph structure that captures both chemical and nutritional properties.

## Data Sources

The project uses three core datasets:

### 1. Metadata_500food.csv
- **Purpose**: Central repository for food sample metadata
- **Key Fields**:
  - `filename`: Unique identifier for each food sample (e.g., P3_E8_G72464.mzML)
  - `description` & `sample_type_common`: Human-readable food labels
  - `ndb_number`: National Nutrient Database number for USDA linking
  - Hierarchical ontology fields: `sample_type_group1`, `botanical_family`, etc.

### 2. Untargeted_biomarkers_level5.csv
- **Purpose**: Defines molecular feature to food relationships
- **Structure**:
  - `feature`: Unique numerical identifier for MS/MS spectral fingerprints
  - `category`: Comma-separated list of food labels where the molecule is a predictive biomarker
- **Key Insight**: These are high-signal, statistically validated relationships

### 3. Feature Intensity Matrix
- **Purpose**: Quantitative abundance data for molecular features in each food sample
- **Structure**: Matrix where rows = molecular features, columns = food samples
- **Use**: Provides edge weights for the graph structure

## Graph Architecture

The system constructs a heterogeneous graph with:

### Node Types
- **Molecule Nodes**: Each unique molecular feature with Spec2Vec embeddings
- **Food Nodes**: Each food sample with nutritional feature vectors from USDA data

### Edge Types
- `('Molecule', 'found_in', 'Food')`: Primary predictive relationships
- `('Food', 'contains', 'Molecule')`: Inverse edges for bidirectional message passing

### Node Features
- **Molecule Features**: Dense vector embeddings from MS/MS spectra (Spec2Vec)
- **Food Features**: Nutritional profiles from USDA FoodData Central + ontological categories

## Machine Learning Approach

### Primary Architecture: Heterogeneous Graph Attention Network (HAN)
- **Advantages**: Native handling of heterogeneous graphs, interpretable attention weights
- **Use Case**: Best balance of performance and scientific insight

### Alternative Architectures
- **Graph Isomorphism Network (GIN)**: Maximum expressive power for food origin prediction
- **Graph Diffusion Models**: Research-oriented approach for robust embeddings

### Training Strategy: Two-Stage Learning
1. **Self-Supervised Pre-training (GraphCL)**: Learn robust representations without labels
2. **Multi-Task Fine-tuning**: Simultaneous food origin prediction and nutritional organization

## Project Structure

```
foodprediction/
├── data/                          # Core datasets
│   ├── Metadata_500food.csv
│   ├── Untargeted_biomarkers_level5.csv
│   ├── featuretable_reformated - Kushal.csv
│   └── FoodData_Central_csv_2025-04-24/  # USDA nutritional data
├── docs/                          # Project documentation
│   └── project-plan.md
├── analysis/                      # Jupyter notebooks for exploration
├── src/                          # Source code (to be implemented)
├── environment.yml               # Conda environment specification
└── README.md                     # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate foodprediction

# Install PyTorch Geometric
pip install torch-geometric
```

### 2. Data Preparation
1. Ensure all CSV files are in the `data/` directory
2. Download USDA FoodData Central dataset (if not already present)
3. Run data exploration notebook: `jupyter notebook analysis/data_exploration.ipynb`

### 3. Development Workflow
1. **Data Analysis**: Explore relationships between datasets
2. **USDA Integration**: Link nutritional data using `ndb_number`
3. **Graph Construction**: Build heterogeneous graph with PyTorch Geometric
4. **Model Training**: Implement HAN with two-stage training
5. **Evaluation**: Assess both prediction accuracy and nutritional organization

## Key Features

- **Multi-Modal Integration**: Combines metabolomic, nutritional, and ontological data
- **Interpretable AI**: Attention mechanisms provide biochemical insights
- **Robust Embeddings**: Self-supervised pre-training for generalization
- **Scalable Architecture**: Designed for large-scale food databases

## Research Applications

- **Food Authentication**: Identify food origin from spectral fingerprints
- **Nutritional Analysis**: Discover foods with similar nutritional profiles
- **Biochemical Discovery**: Understand molecular drivers of food identity
- **Quality Control**: Detect adulteration or mislabeling in food products

## Technical Stack

- **Python 3.12**
- **PyTorch 2.5** & **PyTorch Geometric**
- **Pandas, NumPy, SciPy** for data manipulation
- **Scikit-learn** for evaluation metrics
- **Matplotlib, Seaborn** for visualization
- **Spec2Vec** for molecular feature embeddings

## Contributing

This project follows a structured development approach:
1. Follow the project plan in `docs/project-plan.md`
2. Use minimal, clean code following functional programming principles
3. Prioritize interpretability and scientific insight
4. Maintain comprehensive documentation

## License

[Add your license information here]

## Contact

[Add your contact information here] 