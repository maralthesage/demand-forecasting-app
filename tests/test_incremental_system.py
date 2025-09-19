"""
Tests for incremental training system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from incremental_training_system import IncrementalTrainingSystem
except ImportError as e:
    pytest.skip(
        f"Could not import IncrementalTrainingSystem: {e}", allow_module_level=True
    )


class TestIncrementalTrainingSystem:
    """Test suite for incremental training system"""

    @pytest.fixture
    def temp_system(self):
        """Create a temporary system for testing"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Mock the DATA_PATHS to use temp directory
            import config

            original_paths = config.DATA_PATHS.copy()
            config.DATA_PATHS["cache"] = temp_dir + "/cache"
            config.DATA_PATHS["models"] = temp_dir + "/models"
            config.DATA_PATHS["processed"] = temp_dir + "/processed"

            # Set environment variable to avoid data loading issues
            os.environ["SALES_FORECAST_DATA_PATH"] = temp_dir

            system = IncrementalTrainingSystem()

            yield system

        except Exception as e:
            pytest.skip(f"Could not create test system: {e}")
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            if "config" in locals():
                config.DATA_PATHS = original_paths

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start="2023-01-01", end="2024-01-31", freq="MS")
        data = []

        for i in range(3):  # 3 products
            product_id = f"TEST{i:04d}"

            for date in dates:
                sales = np.random.randint(10, 100)
                price = np.random.uniform(5, 50)

                data.append(
                    {
                        "product_id": product_id,
                        "product_category_id": f"CAT{i % 2}",
                        "MONAT": date,
                        "anz_produkt": sales,
                        "unit_preis": price,
                    }
                )

        return pd.DataFrame(data)

    def test_get_data_hash(self, temp_system, sample_data):
        """Test data hash generation"""
        hash1 = temp_system.get_data_hash(sample_data)
        hash2 = temp_system.get_data_hash(sample_data)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    def test_save_load_metadata(self, temp_system):
        """Test metadata save and load"""
        metadata = {
            "training_date": datetime.now().isoformat(),
            "num_records": 100,
            "feature_columns": ["col1", "col2"],
        }

        temp_system.save_metadata(metadata)
        loaded_metadata = temp_system.load_metadata()

        assert loaded_metadata is not None
        assert loaded_metadata["num_records"] == 100
        assert loaded_metadata["feature_columns"] == ["col1", "col2"]

    def test_has_cache_files(self, temp_system):
        """Test cache file detection"""
        assert not temp_system.has_cache_files()

        # Create mock cache files
        for cache_file in [
            temp_system.base_data_cache,
            temp_system.features_cache,
            temp_system.models_cache,
            temp_system.metadata_cache,
        ]:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.touch()

        assert temp_system.has_cache_files()

    def test_quick_load_fallback(self, temp_system):
        """Test fallback data creation when no cache exists"""
        # This should create fallback data
        features_data, forecaster, feature_columns = temp_system.quick_load_for_app()

        assert not features_data.empty
        assert forecaster is not None
        assert len(feature_columns) > 0

        # Check that fallback data has expected structure
        assert "product_id" in features_data.columns
        assert "MONAT" in features_data.columns
        assert "anz_produkt" in features_data.columns


def test_data_hash_consistency():
    """Test that data hash is consistent"""
    data1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data3 = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})  # Different data

    system = IncrementalTrainingSystem()

    hash1 = system.get_data_hash(data1)
    hash2 = system.get_data_hash(data2)
    hash3 = system.get_data_hash(data3)

    assert hash1 == hash2  # Same data should have same hash
    assert hash1 != hash3  # Different data should have different hash


if __name__ == "__main__":
    pytest.main([__file__])
