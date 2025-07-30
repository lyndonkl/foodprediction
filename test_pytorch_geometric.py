#!/usr/bin/env python3
"""
Test script to verify PyTorch Geometric setup.
"""

def test_pytorch_geometric():
    """Test PyTorch Geometric imports and basic functionality."""
    print("Testing PyTorch Geometric setup...")
    
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
        import torch_scatter
        print("✓ torch_scatter imported successfully")
    except ImportError as e:
        print(f"✗ torch_scatter import failed: {e}")
        return False
    
    try:
        import torch_cluster
        print("✓ torch_cluster imported successfully")
    except ImportError as e:
        print(f"✗ torch_cluster import failed: {e}")
        return False
    
    try:
        import torch_spline_conv
        print("✓ torch_spline_conv imported successfully")
    except ImportError as e:
        print(f"✗ torch_spline_conv import failed: {e}")
        return False
    
    # Test basic GAT functionality
    try:
        from torch_geometric.nn import GATConv
        print("✓ GATConv imported successfully")
    except ImportError as e:
        print(f"✗ GATConv import failed: {e}")
        return False
    
    # Test heterogeneous graph functionality
    try:
        from torch_geometric.nn import HeteroConv
        print("✓ HeteroConv imported successfully")
    except ImportError as e:
        print(f"✗ HeteroConv import failed: {e}")
        return False
    
    print("\n✓ All PyTorch Geometric components working!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch Geometric Setup Test")
    print("=" * 50)
    
    success = test_pytorch_geometric()
    
    if success:
        print("\n✅ PyTorch Geometric setup is complete!")
        print("Ready to proceed with Ticket 01: Data Processing")
    else:
        print("\n❌ PyTorch Geometric setup has issues")
        print("Please check the error messages above") 