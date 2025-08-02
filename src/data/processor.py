"""
Streamlined data processor for food metabolomics graph learning.
Generates intermediate JSON format from raw data files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Feature(BaseModel):
    """Feature with intensity data."""
    id: int
    intensity: float


class Nutrient(BaseModel):
    """Nutrient information."""
    id: int
    name: str
    amount: float
    unit: str


class Sample(BaseModel):
    """Sample with features and nutrients."""
    id: int
    food_name: str
    features: List[Feature]
    nutrients: List[Nutrient]


class IntermediateOutput(BaseModel):
    """Complete intermediate output format."""
    samples: List[Sample]


class FoodMetabolomicsProcessor:
    """
    Streamlined processor for food metabolomics data.
    Generates intermediate JSON format for graph construction.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        
    def load_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            True if successful, False otherwise
        """
        # Core files in main data directory
        core_files = {
            'metadata': 'Metadata_500food.csv',
            'intensity': 'featuretable_reformated - Kushal.csv'
        }
        
        # Nutrient files in FoodData_Central directory
        nutrient_dir = self.data_dir / "FoodData_Central_csv_2025-04-24"
        nutrient_files = {
            'nutrient': 'nutrient.csv',
            'food_nutrient': 'food_nutrient.csv',
            'sr_legacy_food': 'sr_legacy_food.csv'
        }
        
        try:
            # Load core files
            for name, filename in core_files.items():
                file_path = self.data_dir / filename
                if file_path.exists():
                    self.data[name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {name}: {len(self.data[name])} rows")
                else:
                    logger.error(f"Required file not found: {file_path}")
                    return False
            
            # Load nutrient files (optional for basic functionality)
            if nutrient_dir.exists():
                for name, filename in nutrient_files.items():
                    file_path = nutrient_dir / filename
                    if file_path.exists():
                        self.data[name] = pd.read_csv(file_path)
                        logger.info(f"Loaded {name}: {len(self.data[name])} rows")
                    else:
                        logger.warning(f"Nutrient file not found: {file_path}")
            else:
                logger.warning(f"Nutrient directory not found: {nutrient_dir}")
            
            logger.info("Data loading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_sample_intensities(self, sample_id: str) -> Dict[str, float]:
        """
        Get all feature intensities for a sample.
        
        Args:
            sample_id: Sample identifier (filename)
            
        Returns:
            Dictionary mapping feature_id to intensity
        """
        if 'intensity' not in self.data:
            logger.warning("Intensity data not loaded")
            return {}
        
        # Get filename from metadata using sample_id
        if 'metadata' not in self.data:
            logger.warning("Metadata not loaded")
            return {}
        
        sample_metadata = self.data['metadata'][self.data['metadata']['filename'] == sample_id]
        if sample_metadata.empty:
            logger.warning(f"Sample {sample_id} not found in metadata")
            return {}
        
        filename = sample_metadata.iloc[0]['filename']
        intensity_df = self.data['intensity']
        intensity_col = f"{filename} Peak area"
        
        if intensity_col not in intensity_df.columns:
            logger.warning(f"No intensity column found for sample {filename}")
            return {}
        
        # Get intensities for this sample
        sample_intensities = intensity_df[['Feature', intensity_col]].copy()
        sample_intensities = sample_intensities.rename(columns={intensity_col: 'intensity'})
        
        # Filter non-zero intensities
        sample_intensities = sample_intensities[
            (sample_intensities['intensity'] > 0) & 
            (sample_intensities['intensity'].notna())
        ]
        
        # Convert to dictionary
        intensities = sample_intensities.set_index('Feature')['intensity'].to_dict()
        
        logger.debug(f"Found {len(intensities)} non-zero intensities for sample {filename}")
        return intensities
    
    def get_food_for_sample(self, sample_id: str) -> Optional[str]:
        """
        Get the food name associated with a sample.
        
        Args:
            sample_id: Sample identifier (filename)
            
        Returns:
            Food name or None if not found
        """
        if 'metadata' not in self.data:
            logger.warning("Metadata not loaded")
            return None
        
        sample_metadata = self.data['metadata'][self.data['metadata']['filename'] == sample_id]
        if sample_metadata.empty:
            logger.warning(f"Sample {sample_id} not found in metadata")
            return None
        
        food_name = sample_metadata.iloc[0]['sample_type_group5']
        return food_name
    
    def get_nutrients_for_food(self, food_name: str) -> Dict[str, Dict]:
        """
        Get nutrient data for a food using NDB number linking.
        
        Args:
            food_name: Name of the food
            
        Returns:
            Dictionary mapping nutrient_id to nutrient info
        """
        # Check if nutrient data is available
        if not all(key in self.data for key in ['metadata', 'sr_legacy_food', 'food_nutrient', 'nutrient']):
            logger.warning("Nutrient data not available - skipping nutrient lookup")
            return {}
        
        # Get NDB number for this food
        food_metadata = self.data['metadata'][self.data['metadata']['sample_type_group5'] == food_name]
        ndb_numbers = food_metadata['ndb_number'].dropna().unique()
        
        if len(ndb_numbers) == 0:
            logger.warning(f"No NDB numbers found for {food_name}")
            return {}
        
        # Use the first NDB number
        ndb_number = ndb_numbers[0]
        
        # Find fdc_id in SR Legacy food
        sr_legacy_match = self.data['sr_legacy_food'][
            self.data['sr_legacy_food']['NDB_number'].astype(str) == str(ndb_number)
        ]
        
        if sr_legacy_match.empty:
            logger.warning(f"No SR Legacy match for NDB {ndb_number}")
            return {}
        
        fdc_id = sr_legacy_match.iloc[0]['fdc_id']
        
        # Get nutrient data for this fdc_id
        food_nutrients = self.data['food_nutrient'][self.data['food_nutrient']['fdc_id'] == fdc_id]
        
        if food_nutrients.empty:
            logger.warning(f"No nutrient data found for fdc_id {fdc_id}")
            return {}
        
        # Get nutrient names
        nutrient_ids = food_nutrients['nutrient_id'].unique()
        nutrient_names = self.data['nutrient'][self.data['nutrient']['id'].isin(nutrient_ids)]
        
        # Merge nutrient data with names
        result = food_nutrients.merge(
            nutrient_names[['id', 'name', 'unit_name']], 
            left_on='nutrient_id', 
            right_on='id'
        )
        
        # Create nutrient dictionary
        nutrients = {}
        for _, row in result.iterrows():
            nutrient_id = str(row['nutrient_id'])
            nutrients[nutrient_id] = {
                'name': row['name'],
                'unit': row['unit_name'],
                'value': row['amount']
            }
        
        logger.debug(f"Found {len(nutrients)} nutrients for {food_name}")
        return nutrients
    
    def process_all_samples(self) -> IntermediateOutput:
        """
        Process all samples to generate intermediate JSON format.
        
        Returns:
            IntermediateOutput with samples
        """
        if 'metadata' not in self.data:
            logger.error("Metadata not loaded")
            return IntermediateOutput(samples=[])
        
        samples = []
        metadata_df = self.data['metadata']
        
        logger.info(f"Processing {len(metadata_df)} samples...")
        
        for idx, (_, sample) in enumerate(metadata_df.iterrows()):
            sample_id = sample['filename']
            
            # Get sample intensities
            intensities = self.get_sample_intensities(sample_id)
            
            # Get food for this sample
            food_name = self.get_food_for_sample(sample_id)
            
            # Get nutrients for the food
            nutrients = {}
            if food_name:
                nutrients = self.get_nutrients_for_food(food_name)
            
            # Convert intensities to Feature objects
            features = []
            for feature_id, intensity in intensities.items():
                features.append(Feature(id=int(feature_id), intensity=intensity))
            
            # Convert nutrients to Nutrient objects
            nutrient_objects = []
            for nutrient_id, nutrient_data in nutrients.items():
                nutrient_objects.append(Nutrient(
                    id=int(nutrient_id),
                    name=nutrient_data['name'],
                    amount=nutrient_data['value'],
                    unit=nutrient_data['unit']
                ))
            
            # Create sample object
            sample_obj = Sample(
                id=idx,
                food_name=food_name or "unknown",
                features=features,
                nutrients=nutrient_objects
            )
            
            samples.append(sample_obj)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(metadata_df)} samples")
        
        logger.info(f"Completed processing {len(samples)} samples")
        return IntermediateOutput(samples=samples)
    
    def save_intermediate_json(self, output_path: Union[str, Path], output_data: IntermediateOutput) -> bool:
        """
        Save processed samples to intermediate JSON file.
        
        Args:
            output_path: Path to save JSON file
            output_data: IntermediateOutput object
            
        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data.model_dump(), f, indent=2)
            
            logger.info(f"Saved intermediate JSON to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False
    
    def generate_intermediate_format(self, output_path: Union[str, Path] = "data/intermediate_samples.json") -> bool:
        """
        Generate intermediate JSON format from raw data.
        
        Args:
            output_path: Path to save intermediate JSON file
            
        Returns:
            True if successful, False otherwise
        """
        # Load data
        if not self.load_data():
            return False
        
        # Process all samples
        output_data = self.process_all_samples()
        
        if not output_data.samples:
            logger.error("No samples processed")
            return False
        
        # Save to JSON
        if not self.save_intermediate_json(output_path, output_data):
            return False
        
        # Print summary
        total_features = sum(len(sample.features) for sample in output_data.samples)
        total_nutrients = sum(len(sample.nutrients) for sample in output_data.samples)
        unique_foods = len(set(sample.food_name for sample in output_data.samples if sample.food_name != "unknown"))
        
        logger.info(f"Intermediate format generated:")
        logger.info(f"  - {len(output_data.samples)} samples")
        logger.info(f"  - {total_features} total feature intensities")
        logger.info(f"  - {total_nutrients} total nutrient entries")
        logger.info(f"  - {unique_foods} unique foods")
        
        return True 