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
                           force_regenerate: bool = False) -> pd.DataFrame:
        """
        Generate dreaMS embeddings for MS/MS spectra.
        
        Args:
            mgf_path: Path to MGF file
            batch_size: Batch size for processing
            force_regenerate: Force regeneration even if cached
            
        Returns:
            DataFrame with embeddings indexed by Feature ID
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
            
            # Extract Feature IDs from MGF file
            feature_ids = self._extract_feature_ids_from_mgf(mgf_path)
            
            if len(feature_ids) == len(embeddings_df):
                embeddings_df.index = feature_ids
                logger.info(f"Mapped {len(embeddings_df)} embeddings to Feature IDs from MGF")
            else:
                logger.warning(f"Mismatch: {len(feature_ids)} features vs {len(embeddings_df)} embeddings")
                # Fallback to spectrum indices
                embeddings_df.index = [f"spectrum_{i}" for i in range(len(embeddings_df))]
            
            # Cache the embeddings
            logger.info(f"Caching {len(embeddings_df)} embeddings to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_df, f)
            
            logger.info(f"Generated {len(embeddings_df)} embeddings with {embeddings_df.shape[1]} dimensions")
            return embeddings_df
            
        except Exception as e:
            logger.error(f"Error generating dreaMS embeddings: {e}")
            raise
    
    def _extract_feature_ids_from_mgf(self, mgf_path: Path) -> List[str]:
        """
        Extract Feature IDs from MGF file.
        
        Args:
            mgf_path: Path to MGF file
            
        Returns:
            List of Feature IDs in order of appearance
        """
        feature_ids = []
        
        try:
            with open(mgf_path, 'r') as f:
                content = f.read()
            
            # Split by BEGIN IONS
            spectra = content.split('BEGIN IONS')
            
            for spectrum in spectra[1:]:  # Skip first empty part
                # Look for FEATURE_ID line
                lines = spectrum.split('\n')
                for line in lines:
                    if line.startswith('FEATURE_ID='):
                        feature_id = line.split('=')[1].strip()
                        feature_ids.append(feature_id)
                        break
                else:
                    # If no FEATURE_ID found, use index
                    feature_ids.append(f"spectrum_{len(feature_ids)}")
            
            logger.info(f"Extracted {len(feature_ids)} Feature IDs from MGF file")
            return feature_ids
            
        except Exception as e:
            logger.error(f"Error extracting Feature IDs from MGF: {e}")
            return []
    
    def get_embedding_dimension(self, embeddings_df: pd.DataFrame) -> int:
        """
        Get the dimension of dreaMS embeddings.
        
        Args:
            embeddings_df: DataFrame with embeddings
            
        Returns:
            Embedding dimension
        """
        # All columns are embedding dimensions now
        return len(embeddings_df.columns)
    
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