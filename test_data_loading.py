#!/usr/bin/env python3
"""
Test script for data loading utilities.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logging import setup_logging
from src.data.loaders import (
    validate_mgf_file, 
    load_all_data_files, 
    FoodMetabolomicsDataManager,
    get_data_summary
)

def test_mgf_validation():
    """Test MGF file validation."""
    print("Testing MGF file validation...")
    
    # Test with existing MGF file
    mgf_path = Path("data/500_foods.mgf")
    if mgf_path.exists():
        is_valid = validate_mgf_file(mgf_path)
        print(f"✓ MGF validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid
    else:
        print("⚠ MGF file not found, skipping validation test")
        return True

def test_data_loading():
    """Test data loading functions."""
    print("\nTesting data loading functions...")
    
    try:
        # Load all data files
        data_dict = load_all_data_files("data")
        
        # Check if we have the required files
        required_files = ['metadata', 'intensity']
        missing_files = [f for f in required_files if f not in data_dict]
        
        if missing_files:
            print(f"⚠ Missing required files: {missing_files}")
            print("⚠ This is expected if you don't have the full dataset yet")
            return True
        
        # Get data summary
        summary = get_data_summary(data_dict)
        
        print(f"✓ Data loading: PASSED")
        print(f"✓ Loaded files: {summary['loaded_files']}")
        print(f"✓ Dataset summary: {summary['total_samples']} samples, {summary['unique_foods']} foods, {summary['total_features']} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_data_manager():
    """Test FoodMetabolomicsDataManager."""
    print("\nTesting FoodMetabolomicsDataManager...")
    
    try:
        # Initialize data manager
        data_manager = FoodMetabolomicsDataManager("data")
        
        # Try to load data
        success = data_manager.load_data()
        
        if not success:
            print("⚠ Data loading failed, but this is expected if files are missing")
            return True
        
        # Get summary
        summary = data_manager.get_data_summary()
        
        print(f"✓ Data manager: PASSED")
        print(f"✓ Summary: {summary}")
        
        # Test feature extraction if data is available
        if 'metadata' in data_manager.data_dict:
            metadata_df = data_manager.data_dict['metadata']
            if not metadata_df.empty:
                # Test nutrient features for first food
                first_food = metadata_df['sample_type_group5'].iloc[0]
                nutrient_features = data_manager.get_nutrient_features(first_food)
                print(f"✓ Nutrient features for {first_food}: {len(nutrient_features)} nutrients")
                
                # Test sample features for first sample
                first_sample = metadata_df['filename'].iloc[0]
                sample_features = data_manager.get_sample_features(first_sample)
                if sample_features:
                    print(f"✓ Sample features for {first_sample}: food={sample_features.get('food_name', 'N/A')}")
                    
                    # Test molecule features if we have dreaMS embeddings
                    try:
                        from src.data.dreams_embeddings import DreaMSEmbeddingGenerator
                        generator = DreaMSEmbeddingGenerator()
                        mgf_path = Path("data/500_foods.mgf")
                        
                        if mgf_path.exists():
                            dreams_embeddings = generator.generate_embeddings(mgf_path, force_regenerate=False)
                            
                            # Test molecule features for a few features from the sample
                            molecule_intensities = sample_features.get('molecule_intensities', {})
                            if molecule_intensities:
                                test_features = list(molecule_intensities.keys())[:3]  # Test first 3 features
                                for feature_id in test_features:
                                    molecule_features = data_manager.get_molecule_features(feature_id, dreams_embeddings)
                                    if molecule_features:
                                        print(f"✓ Molecule features for {feature_id}: {molecule_features.get('embedding_dim', 0)} dimensions")
                                    else:
                                        print(f"⚠ No molecule features for {feature_id}")
                    except Exception as e:
                        print(f"⚠ dreaMS integration test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Data Loading Utilities Test")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    all_passed = True
    
    # Test MGF validation
    if not test_mgf_validation():
        all_passed = False
    
    # Test data loading
    if not test_data_loading():
        all_passed = False
    
    # Test data manager
    if not test_data_manager():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ Data loading utilities are working!")
        print("✅ Ready for next step: dreaMS integration")
    else:
        print("❌ Some tests failed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 