"""
Basic tests that don't require complex dependencies
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pandas_import():
    """Test that pandas is available"""
    assert pd.__version__ is not None


def test_numpy_import():
    """Test that numpy is available"""
    assert np.__version__ is not None


def test_basic_data_operations():
    """Test basic data operations work"""
    # Create sample data
    data = pd.DataFrame({
        'product_id': ['TEST001', 'TEST002', 'TEST003'],
        'sales': [10, 20, 30],
        'price': [5.0, 10.0, 15.0]
    })
    
    # Test basic operations
    assert len(data) == 3
    assert data['sales'].sum() == 60
    assert data['price'].mean() == 10.0


def test_config_import():
    """Test that config can be imported"""
    try:
        import config
        assert hasattr(config, 'DATA_PATHS')
        assert hasattr(config, 'MODEL_CONFIG')
    except ImportError:
        pytest.skip("Config module not available")


def test_project_structure():
    """Test that project structure exists"""
    project_root = Path(__file__).parent.parent
    
    # Check key directories exist
    assert (project_root / 'data').exists()
    assert (project_root / 'models').exists()
    assert (project_root / 'utils').exists()
    assert (project_root / 'tests').exists()
    
    # Check key files exist
    assert (project_root / 'requirements.txt').exists()
    assert (project_root / 'config.py').exists()


if __name__ == "__main__":
    pytest.main([__file__])
