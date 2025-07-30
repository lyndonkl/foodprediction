#!/usr/bin/env python3
"""
Test script to verify the basic project setup.
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")

def test_project_structure():
    """Test that the project structure is set up correctly."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "src/models",
        "src/data", 
        "src/train",
        "src/eval",
        "src/utils",
        "notebooks",
        "data/raw",
        "data/processed",
        "data/embeddings",
        "configs"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
    
    required_files = [
        "environment.yml",
        "configs/default.yaml",
        ".gitignore",
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/logging.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from src.utils.config import load_config
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  - Model hidden_dim: {config.get('model', {}).get('hidden_dim', 'Not found')}")
        print(f"  - Training epochs: {config.get('training', {}).get('pretrain_epochs', 'Not found')}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")

def test_logging():
    """Test logging setup."""
    print("\nTesting logging setup...")
    
    try:
        from src.utils.logging import setup_logging, get_logger
        logger = setup_logging(log_level="INFO")
        test_logger = get_logger("test")
        test_logger.info("Test log message")
        print("✓ Logging setup successful")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Food Metabolomics Graph Learning - Setup Test")
    print("=" * 50)
    
    test_basic_imports()
    test_project_structure()
    test_config_loading()
    test_logging()
    
    print("\n" + "=" * 50)
    print("Setup test completed!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Install the conda environment: conda env create -f environment.yml")
    print("2. Activate the environment: conda activate foodprediction")
    print("3. Install dreaMS: pip install dreams-embeddings")
    print("4. Run this test again to verify everything works")

if __name__ == "__main__":
    main() 