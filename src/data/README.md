# Food Metabolomics Data Processing

This directory contains the streamlined data processing pipeline for food metabolomics graph learning.

## Overview

The data processing pipeline has been cleaned up and streamlined to focus on generating an intermediate JSON format that contains all the necessary information for graph construction.

## Files

### Core Files

- **`processor.py`** - Main data processor that generates intermediate JSON format
- **`loaders.py`** - Essential data loading utilities
- **`dreams_embeddings.py`** - dreaMS embedding generation for molecule features
- **`__init__.py`** - Package initialization and exports

### Scripts

- **`generate_intermediate.py`** - Script to generate the intermediate JSON format
- **`test_processor.py`** - Test script to verify processor functionality

## Usage

### Generate Intermediate JSON Format

```python
from processor import FoodMetabolomicsProcessor

# Initialize processor
processor = FoodMetabolomicsProcessor(data_dir="data")

# Generate intermediate format
success = processor.generate_intermediate_format("data/intermediate_samples.json")
```

### Run the Generation Script

```bash
cd src/data
python generate_intermediate.py
```

### Test the Processor

```bash
cd src/data
python test_processor.py
```

## Intermediate JSON Format

The intermediate JSON format contains an array of sample objects, where each sample object has the following structure:

```json
{
  "sample_id": "sample_filename",
  "food_name": "food_type",
  "intensities": {
    "feature_id": intensity_value,
    ...
  },
  "nutrients": {
    "nutrient_name_unit": {
      "name": "nutrient_name",
      "unit": "unit_name", 
      "value": nutrient_value
    },
    ...
  }
}
```

## Data Flow

1. **Load Raw Data**: Load metadata, intensity, and nutrient CSV files
2. **Process Samples**: For each sample in metadata:
   - Extract feature intensities for the sample
   - Get associated food name
   - Get nutrient data for the food
3. **Generate JSON**: Save all sample objects to intermediate JSON file
4. **Graph Construction**: Use the JSON file to build the heterogeneous graph (in separate module)

## Required Data Files

The processor expects the following files in the data directory:

- `Metadata_500food.csv` - Sample metadata with food associations
- `featuretable_reformated - Kushal.csv` - Feature intensity matrix
- `nutrient.csv` - USDA nutrient definitions
- `food_nutrient.csv` - USDA food-nutrient relationships
- `sr_legacy_food.csv` - SR Legacy food mapping

## Benefits of Streamlined Approach

1. **Clear Data Flow**: Single responsibility for each component
2. **Reduced Complexity**: Removed redundant functions and classes
3. **Better Testing**: Easier to test individual components
4. **Intermediate Format**: JSON format provides clear data structure for graph construction
5. **Separation of Concerns**: Data processing separate from graph construction

## Next Steps

The intermediate JSON format can be used to:
1. Build heterogeneous graphs with PyTorch Geometric
2. Apply dreaMS embeddings to molecule nodes
3. Create train/validation/test splits
4. Implement graph neural network models 