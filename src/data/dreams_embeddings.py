"""
dreaMS embedding generation for molecule node features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import pickle
from dreams.api import dreams_embeddings

logger = logging.getLogger(__name__)


class DreaMSEmbeddingGenerator:
    """
    Generate dreaMS embeddings for MS/MS spectra.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = "data/embeddings"):
        """
        Initialize the embedding generator.
        
        Args:
            cache_dir: Directory to cache embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"dreaMS embedding cache directory: {self.cache_dir}")
    
    def generate_embeddings(self, mgf_path: Union[str, Path], 
                           batch_size: int = 32,
                           force_regenerate: bool = False) -> pd.DataFrame:
        """
        Generate dreaMS embeddings for MS/MS spectra.
        
        Args:
            mgf_path: Path to MGF file
            batch_size: Batch size for processing
            force_regenerate: Force regeneration even if cached
            
        Returns:
            DataFrame with embeddings
        """
        mgf_path = Path(mgf_path)
        cache_file = self.cache_dir / f"{mgf_path.stem}_embeddings.pkl"
        
        # Check if cached embeddings exist
        if not force_regenerate and cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    embeddings_df = pickle.load(f)
                logger.info(f"Loaded {len(embeddings_df)} cached embeddings")
                return embeddings_df
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Generate new embeddings
        logger.info(f"Generating dreaMS embeddings for {mgf_path}")
        
        try:
            # Use dreaMS API to generate embeddings
            embeddings = dreams_embeddings(str(mgf_path))
            
            # Convert to DataFrame
            embeddings_df = pd.DataFrame(embeddings)
            
            # Add spectrum index as identifier
            embeddings_df['spectrum_idx'] = range(len(embeddings_df))
            
            # Cache the embeddings
            logger.info(f"Caching {len(embeddings_df)} embeddings to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_df, f)
            
            logger.info(f"Generated {len(embeddings_df)} embeddings with {embeddings_df.shape[1]-1} dimensions")
            return embeddings_df
            
        except Exception as e:
            logger.error(f"Error generating dreaMS embeddings: {e}")
            raise
    
    def get_embedding_dimension(self, embeddings_df: pd.DataFrame) -> int:
        """
        Get the dimension of dreaMS embeddings.
        
        Args:
            embeddings_df: DataFrame with embeddings
            
        Returns:
            Embedding dimension
        """
        # Exclude non-embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col not in ['spectrum_idx']]
        return len(embedding_cols)
    
    def validate_embeddings(self, embeddings_df: pd.DataFrame) -> bool:
        """
        Validate dreaMS embeddings quality.
        
        Args:
            embeddings_df: DataFrame with embeddings
            
        Returns:
            True if embeddings are valid
        """
        if embeddings_df.empty:
            logger.error("Embeddings DataFrame is empty")
            return False
        
        # Check for NaN values
        nan_count = embeddings_df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings")
        
        # Check embedding dimension
        embedding_dim = self.get_embedding_dimension(embeddings_df)
        if embedding_dim == 0:
            logger.error("No embedding dimensions found")
            return False
        
        logger.info(f"Embedding validation passed: {len(embeddings_df)} embeddings, {embedding_dim} dimensions")
        return True
    
    def get_embedding_summary(self, embeddings_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for embeddings.
        
        Args:
            embeddings_df: DataFrame with embeddings
            
        Returns:
            Dictionary with embedding summary
        """
        embedding_dim = self.get_embedding_dimension(embeddings_df)
        
        summary = {
            'total_embeddings': len(embeddings_df),
            'embedding_dimension': embedding_dim,
            'memory_usage_mb': embeddings_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'nan_count': embeddings_df.isna().sum().sum(),
            'mean_norm': np.linalg.norm(embeddings_df.select_dtypes(include=[np.number]), axis=1).mean()
        }
        
        return summary


def test_dreams_integration():
    """
    Test dreaMS integration with sample data.
    """
    mgf_path = Path("data/500_foods.mgf")
    
    if not mgf_path.exists():
        logger.warning(f"MGF file not found: {mgf_path}")
        return False
    
    try:
        generator = DreaMSEmbeddingGenerator()
        embeddings_df = generator.generate_embeddings(mgf_path, force_regenerate=False)
        
        if generator.validate_embeddings(embeddings_df):
            summary = generator.get_embedding_summary(embeddings_df)
            logger.info(f"dreaMS integration test passed: {summary}")
            return True
        else:
            logger.error("dreaMS integration test failed: invalid embeddings")
            return False
            
    except Exception as e:
        logger.error(f"dreaMS integration test failed: {e}")
        return False 