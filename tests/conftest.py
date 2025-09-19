"""
Pytest configuration and fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ["SALES_FORECAST_DATA_PATH"] = str(project_root / "tests" / "test_data")
os.environ["LOG_LEVEL"] = "ERROR"  # Reduce logging noise during tests


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Create test data directory
    test_data_dir = project_root / "tests" / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    # Create CSV structure for tests
    csv_dir = test_data_dir / "CSV" / "F01"
    csv_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup is handled by individual test fixtures
