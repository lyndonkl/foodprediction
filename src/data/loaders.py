"""
Data loading utilities for MS/MS spectra and nutritional data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def validate_mgf_file(mgf_path: Union[str, Path]) -> bool:
    """
    Validate MGF file format for dreaMS processing.
    
    Args:
        mgf_path: Path to MGF file
        
    Returns:
        True if valid, False otherwise
    """
    mgf_path = Path(mgf_path)
    
    if not mgf_path.exists():
        logger.error(f"MGF file not found: {mgf_path}")
        return False
    
    try:
        with open(mgf_path, 'r') as f:
            content = f.read()
        
        # Basic MGF format validation
        if 'BEGIN IONS' not in content or 'END IONS' not in content:
            logger.error(f"Invalid MGF format: missing BEGIN/END IONS markers")
            return False
        
        # Check for at least one spectrum
        spectrum_count = content.count('BEGIN IONS')
        if spectrum_count == 0:
            logger.error(f"No spectra found in MGF file")
            return False
        
        logger.info(f"MGF file validation passed: {spectrum_count} spectra found")
        return True
        
    except Exception as e:
        logger.error(f"Error reading MGF file: {e}")
        return False


def load_all_data_files(data_dir: Union[str, Path] = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all required data files for food metabolomics analysis.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with loaded DataFrames
    """
    data_dir = Path(data_dir)
    
    required_files = {
        'metadata': 'Metadata_500food.csv',
        'intensity': 'featuretable_reformated - Kushal.csv', 
        'nutrient': 'nutrient.csv',
        'food_nutrient': 'food_nutrient.csv',
        'sr_legacy_food': 'sr_legacy_food.csv',
        'food': 'food.csv'
    }
    
    data_dict = {}
    
    for name, filename in required_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                data_dict[name] = pd.read_csv(file_path)
                logger.info(f"Loaded {name}: {len(data_dict[name])} rows, {len(data_dict[name].columns)} columns")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                raise
        else:
            logger.warning(f"File not found: {file_path}")
    
    return data_dict


def get_food_intensities(food_name: str, intensity_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get all feature intensities for all samples of a given food.
    
    Args:
        food_name: Name of the food from sample_type_group5
        intensity_df: Feature intensity matrix
        metadata_df: Food metadata
    
    Returns:
        DataFrame with Feature, filename, intensity columns
    """
    # Get all samples for this food
    food_samples = metadata_df[metadata_df['sample_type_group5'] == food_name]
    
    if food_samples.empty:
        logger.warning(f"No samples found for food: {food_name}")
        return pd.DataFrame()
    
    all_intensities = []
    
    for _, sample in food_samples.iterrows():
        filename = sample['filename']
        
        # Look for the correct column pattern: "{sample_id} Peak area"
        intensity_col = f"{filename} Peak area"
        
        if intensity_col in intensity_df.columns:
            # Get intensities for this sample
            sample_intensities = intensity_df[['Feature', intensity_col]].copy()
            sample_intensities = sample_intensities.rename(columns={intensity_col: 'intensity'})
            sample_intensities['filename'] = filename
            
            # Filter non-zero intensities
            sample_intensities = sample_intensities[
                (sample_intensities['intensity'] > 0) & 
                (sample_intensities['intensity'].notna())
            ]
            
            all_intensities.append(sample_intensities)
        else:
            logger.warning(f"No intensity column found for sample {filename}: {intensity_col}")
    
    if all_intensities:
        result = pd.concat(all_intensities, ignore_index=True)
        logger.info(f"Found {len(result)} feature-intensity pairs for {food_name}")
        return result
    else:
        logger.warning(f"No intensity data found for {food_name}")
        return pd.DataFrame()


def get_food_nutrients_with_sr_legacy(food_name: str, metadata_df: pd.DataFrame, 
                                     sr_legacy_food_df: pd.DataFrame, 
                                     food_nutrient_df: pd.DataFrame, 
                                     nutrient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get nutritional data for a food using NDB number -> SR Legacy -> fdc_id linking.
    
    Args:
        food_name: Name of the food
        metadata_df: Food metadata with NDB numbers
        sr_legacy_food_df: SR Legacy food mapping
        food_nutrient_df: USDA food_nutrient.csv
        nutrient_df: USDA nutrient.csv
    
    Returns:
        DataFrame with nutrient information
    """
    # Get NDB number for this food
    food_metadata = metadata_df[metadata_df['sample_type_group5'] == food_name]
    ndb_numbers = food_metadata['ndb_number'].dropna().unique()
    
    if len(ndb_numbers) == 0:
        logger.warning(f"No NDB numbers found for {food_name}")
        return pd.DataFrame()
    
    # Use the first NDB number
    ndb_number = ndb_numbers[0]
    logger.info(f"Looking for NDB number {ndb_number} for {food_name}")
    
    # Step 1: Find fdc_id in SR Legacy food (convert to string for comparison)
    sr_legacy_match = sr_legacy_food_df[sr_legacy_food_df['NDB_number'].astype(str) == str(ndb_number)]
    
    if sr_legacy_match.empty:
        logger.warning(f"No SR Legacy match for NDB {ndb_number}")
        return pd.DataFrame()
    
    fdc_id = sr_legacy_match.iloc[0]['fdc_id']
    logger.info(f"✓ Found SR Legacy match: fdc_id {fdc_id}")
    
    # Step 2: Get nutrient data for this fdc_id
    food_nutrients = food_nutrient_df[food_nutrient_df['fdc_id'] == fdc_id]
    
    if food_nutrients.empty:
        logger.warning(f"No nutrient data found for fdc_id {fdc_id}")
        return pd.DataFrame()
    
    logger.info(f"✓ Found {len(food_nutrients)} nutrient entries")
    
    # Step 3: Get nutrient names
    nutrient_ids = food_nutrients['nutrient_id'].unique()
    nutrient_names = nutrient_df[nutrient_df['id'].isin(nutrient_ids)]
    
    # Merge nutrient data with names
    result = food_nutrients.merge(
        nutrient_names[['id', 'name', 'unit_name']], 
        left_on='nutrient_id', 
        right_on='id'
    )
    
    logger.info(f"✓ Found {len(result)} nutrients with names")
    return result[['name', 'amount', 'unit_name']]


class FoodMetabolomicsDataManager:
    """
    Manages food metabolomics data for graph construction.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dict = {}
        self.nutrient_features = {}
        self.sample_features = {}
        self.molecule_features = {}
        
    def load_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data_dict = load_all_data_files(self.data_dir)
            logger.info("Data loading completed")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_nutrient_features(self, food_name: str) -> Dict:
        """
        Get nutrient features for a food.
        
        Args:
            food_name: Name of the food
            
        Returns:
            Dictionary with nutrient features {nutrient_id: {name, normalized_value}}
        """
        if food_name in self.nutrient_features:
            return self.nutrient_features[food_name]
        
        # Get nutrient data for this food
        if 'metadata' not in self.data_dict or 'sr_legacy_food' not in self.data_dict:
            logger.warning("Required data not loaded")
            return {}
        
        nutrients_df = get_food_nutrients_with_sr_legacy(
            food_name, 
            self.data_dict['metadata'],
            self.data_dict['sr_legacy_food'],
            self.data_dict.get('food_nutrient', pd.DataFrame()),
            self.data_dict.get('nutrient', pd.DataFrame())
        )
        
        if nutrients_df.empty:
            return {}
        
        # Create nutrient features
        nutrient_features = {}
        for _, row in nutrients_df.iterrows():
            nutrient_id = f"{row['name']}_{row['unit_name']}"
            nutrient_features[nutrient_id] = {
                'name': row['name'],
                'unit': row['unit_name'],
                'value': row['amount']
            }
        
        self.nutrient_features[food_name] = nutrient_features
        return nutrient_features
    
    def get_sample_features(self, sample_id: str) -> Dict:
        """
        Get sample features including food assignment and molecule intensities.
        
        Args:
            sample_id: Sample identifier (filename)
            
        Returns:
            Dictionary with sample features
        """
        if sample_id in self.sample_features:
            return self.sample_features[sample_id]
        
        # Get sample metadata
        if 'metadata' not in self.data_dict:
            logger.warning("Metadata not loaded")
            return {}
        
        sample_metadata = self.data_dict['metadata'][self.data_dict['metadata']['filename'] == sample_id]
        if sample_metadata.empty:
            logger.warning(f"Sample {sample_id} not found in metadata")
            return {}
        
        food_name = sample_metadata.iloc[0]['sample_type_group5']
        
        # Get intensities for this sample
        if 'intensity' not in self.data_dict:
            logger.warning("Intensity data not loaded")
            return {}
        
        intensity_df = self.data_dict['intensity']
        
        # Find intensity column for this sample
        intensity_col = f"{sample_id} Peak area"
        if intensity_col not in intensity_df.columns:
            logger.warning(f"No intensity column found for sample {sample_id}: {intensity_col}")
            return {}
        
        # Get molecule intensities
        sample_intensities = intensity_df[['Feature', intensity_col]].copy()
        sample_intensities = sample_intensities.rename(columns={intensity_col: 'intensity'})
        sample_intensities = sample_intensities[
            (sample_intensities['intensity'] > 0) & 
            (sample_intensities['intensity'].notna())
        ]
        
        # Create sample features
        sample_features = {
            'food_name': food_name,
            'molecule_intensities': sample_intensities.set_index('Feature')['intensity'].to_dict()
        }
        
        self.sample_features[sample_id] = sample_features
        return sample_features
    
    def get_molecule_features(self, feature_id: str, dreams_embeddings: pd.DataFrame) -> Dict:
        """
        Get molecule features including dreaMS embeddings.
        
        Args:
            feature_id: Feature identifier
            dreams_embeddings: DataFrame with dreaMS embeddings
            
        Returns:
            Dictionary with molecule features
        """
        if feature_id in self.molecule_features:
            return self.molecule_features[feature_id]
        
        # Get dreaMS embedding for this feature
        if feature_id in dreams_embeddings.index:
            embedding = dreams_embeddings.loc[feature_id].to_dict()
            molecule_features = {
                'embedding': embedding,
                'embedding_dim': len(embedding)
            }
        else:
            logger.warning(f"No dreaMS embedding found for feature {feature_id}")
            molecule_features = {}
        
        self.molecule_features[feature_id] = molecule_features
        return molecule_features
    

    
    def get_data_summary(self) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Returns:
            Dictionary with dataset summary
        """
        if not self.data_dict:
            return {}
        
        metadata_df = self.data_dict.get('metadata', pd.DataFrame())
        intensity_df = self.data_dict.get('intensity', pd.DataFrame())
        
        summary = {
            'total_samples': len(metadata_df),
            'unique_foods': metadata_df['sample_type_group5'].nunique() if not metadata_df.empty else 0,
            'total_features': len(intensity_df),
            'loaded_files': list(self.data_dict.keys()),
            'cached_nutrient_features': len(self.nutrient_features),
            'cached_sample_features': len(self.sample_features),
            'cached_molecule_features': len(self.molecule_features)
        }
        
        return summary


def get_data_summary(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        data_dict: Dictionary with all loaded data files
        
    Returns:
        Dictionary with dataset summary
    """
    metadata_df = data_dict.get('metadata', pd.DataFrame())
    intensity_df = data_dict.get('intensity', pd.DataFrame())
    
    summary = {
        'total_samples': len(metadata_df),
        'unique_foods': metadata_df['sample_type_group5'].nunique() if not metadata_df.empty else 0,
        'total_features': len(intensity_df),
        'loaded_files': list(data_dict.keys())
    }
    
    return summary 