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

def main():
    """Run dreaMS integration test."""
    print("=" * 50)
    print("dreaMS Integration Test")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Test dreaMS integration
    success = test_dreams_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ dreaMS integration is working!")
        print("✅ Ready for next step: Nutritional data processing")
    else:
        print("❌ dreaMS integration failed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 