"""
dreaMS embedding generation for molecule node features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
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