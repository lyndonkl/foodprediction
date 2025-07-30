#!/usr/bin/env python3
"""
Comprehensive test script to verify the complete project setup.
"""

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    return True

def test_pytorch_geometric():
    """Test PyTorch Geometric imports."""
    print("\nTesting PyTorch Geometric...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print("✓ PyTorch Geometric imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
        return False
    
    try:
        from torch_geometric.nn import GATConv, HeteroConv
        print("✓ GATConv and HeteroConv imported successfully")
    except ImportError as e:
        print(f"✗ GATConv/HeteroConv import failed: {e}")
        return False
    
    return True

def test_dreams():
    """Test dreaMS imports."""
    print("\nTesting dreaMS...")
    
    try:
        import dreams
        print("✓ dreaMS imported successfully")
    except ImportError as e:
        print(f"✗ dreaMS import failed: {e}")
        return False
    
    try:
        from dreams.api import dreams_embeddings, dreams_predictions
        print("✓ dreaMS API functions imported successfully")
    except ImportError as e:
        print(f"✗ dreaMS API import failed: {e}")
        return False
    
    return True

def test_project_components():
    """Test project-specific components."""
    print("\nTesting project components...")
    
    try:
        from src.utils.config import load_config
        config = load_config()
        print("✓ Configuration loading works")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    try:
        from src.utils.logging import setup_logging
        logger = setup_logging(log_level="INFO")
        print("✓ Logging setup works")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Food Metabolomics Graph Learning - Complete Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test basic imports
    if not test_basic_imports():
        all_passed = False
    
    # Test PyTorch Geometric
    if not test_pytorch_geometric():
        all_passed = False
    
    # Test dreaMS
    if not test_dreams():
        all_passed = False
    
    # Test project components
    if not test_project_components():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ Complete setup is ready!")
        print("✅ Ready to proceed with Ticket 01: Data Processing")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Please check the error messages above")
    print("=" * 60)

if __name__ == "__main__":
    main() 