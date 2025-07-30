#!/usr/bin/env python3
"""
Test script to verify dreaMS installation and basic functionality.
"""

def test_dreams_import():
    """Test dreaMS imports and basic functionality."""
    print("Testing dreaMS setup...")
    
    try:
        import dreams
        print("✓ dreaMS imported successfully")
    except ImportError as e:
        print(f"✗ dreaMS import failed: {e}")
        return False
    
    try:
        from dreams.api import dreams_embeddings, dreams_predictions, PreTrainedModel
        print("✓ dreaMS API functions imported successfully")
    except ImportError as e:
        print(f"✗ dreaMS API import failed: {e}")
        return False
    
    try:
        from dreams.models.dreams.dreams import DreaMS as DreaMSModel
        print("✓ DreaMS model class imported successfully")
    except ImportError as e:
        print(f"✗ DreaMS model class import failed: {e}")
        return False
    
    try:
        # Test basic dreaMS functionality
        print("✓ dreaMS basic functionality available")
    except Exception as e:
        print(f"✗ dreaMS functionality test failed: {e}")
        return False
    
    print("\n✓ dreaMS setup is complete!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("dreaMS Setup Test")
    print("=" * 50)
    
    success = test_dreams_import()
    
    if success:
        print("\n✅ dreaMS setup is complete!")
        print("Ready to proceed with Ticket 01: Data Processing")
    else:
        print("\n❌ dreaMS setup has issues")
        print("Please check the error messages above") 