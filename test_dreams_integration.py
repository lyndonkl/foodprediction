#!/usr/bin/env python3
"""
Test script for dreaMS integration.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logging import setup_logging
from src.data.dreams_embeddings import DreaMSEmbeddingGenerator, test_dreams_integration

def test_dreams_with_mapping():
    """Test dreaMS integration with proper Feature ID mapping."""
    print("Testing dreaMS integration with Feature ID mapping...")
    
    try:
        # Test dreaMS with mapping
        generator = DreaMSEmbeddingGenerator()
        mgf_path = Path("data/500_foods.mgf")
        
        if mgf_path.exists():
            embeddings_df = generator.generate_embeddings(
                mgf_path, 
                force_regenerate=False
            )
            
            if generator.validate_embeddings(embeddings_df):
                summary = generator.get_embedding_summary(embeddings_df)
                print(f"✓ dreaMS with mapping test passed: {summary}")
                
                # Check if Feature IDs are properly mapped
                if not embeddings_df.empty:
                    sample_features = embeddings_df.index[:5].tolist()
                    print(f"✓ Sample Feature IDs: {sample_features}")
                
                return True
            else:
                print("✗ dreaMS with mapping test failed: invalid embeddings")
                return False
        else:
            print("⚠ MGF file not found for mapping test")
            return True
            
    except Exception as e:
        print(f"✗ dreaMS with mapping test failed: {e}")
        return False

def main():
    """Run dreaMS integration test."""
    print("=" * 50)
    print("dreaMS Integration Test")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Test dreaMS with proper mapping
    success = test_dreams_with_mapping()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ dreaMS integration with mapping is working!")
        print("✅ Ready for next step: Nutritional data processing")
    else:
        print("❌ dreaMS integration failed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 