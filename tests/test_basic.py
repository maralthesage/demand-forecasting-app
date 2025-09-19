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
    data = pd.DataFrame(
        {
            "product_id": ["TEST001", "TEST002", "TEST003"],
            "sales": [10, 20, 30],
            "price": [5.0, 10.0, 15.0],
        }
    )

    # Test basic operations
    assert len(data) == 3
    assert data["sales"].sum() == 60
    assert data["price"].mean() == 10.0


def test_config_import():
    """Test that config can be imported"""
    try:
        import config

        assert hasattr(config, "DATA_PATHS")
        assert hasattr(config, "MODEL_CONFIG")
    except ImportError:
        pytest.skip("Config module not available")


def test_project_structure():
    """Test that project structure exists"""
    project_root = Path(__file__).parent.parent

    # Check key files exist (these are essential)
    assert (project_root / "requirements.txt").exists(), f"requirements.txt not found in {project_root}"
    assert (project_root / "config.py").exists(), f"config.py not found in {project_root}"

    # Check key directories exist (with better error messages)
    directories_to_check = ["models", "utils", "tests"]
    for dir_name in directories_to_check:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory '{dir_name}' not found in {project_root}"

    # Check data directory exists (optional in CI environments)
    data_dir = project_root / "data"
    if data_dir.exists():
        assert data_dir.is_dir(), f"data path exists but is not a directory: {data_dir}"
        print(f"✅ Data directory found at {data_dir}")
    else:
        print(f"ℹ️  Data directory not found at {data_dir} (this is expected in CI environments)")


if __name__ == "__main__":
    pytest.main([__file__])
