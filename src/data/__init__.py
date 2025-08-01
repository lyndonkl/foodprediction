"""
Streamlined data processing package for Food Metabolomics Graph Learning.

This package handles:
- Raw data loading and validation
- Intermediate JSON format generation
- dreaMS embedding generation
- Data preprocessing pipeline
"""

from .processor import FoodMetabolomicsProcessor
from .dreams_embeddings import DreaMSEmbeddingGenerator

__version__ = "0.1.0"

__all__ = [
    'FoodMetabolomicsProcessor',
    'DreaMSEmbeddingGenerator'
] 