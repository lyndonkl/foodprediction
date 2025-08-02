#!/usr/bin/env python3
"""
Script to generate intermediate JSON format from raw data files.
"""

import logging
from pathlib import Path
from .processor import FoodMetabolomicsProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Generate intermediate JSON format from raw data."""
    
    # Initialize processor with correct data directory
    processor = FoodMetabolomicsProcessor(data_dir="data")
    
    # Generate intermediate format
    output_path = "data/intermediate_samples.json"
    
    print("Generating intermediate JSON format...")
    success = processor.generate_intermediate_format(output_path)
    
    if success:
        print(f"✓ Successfully generated intermediate format: {output_path}")
        
        # Print file size
        file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
        print(f"  File size: {file_size:.2f} MB")
        
    else:
        print("✗ Failed to generate intermediate format")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 