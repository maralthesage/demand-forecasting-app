"""
Configuration settings for the Demand Forecasting Application
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DATA_PATH = os.getenv("SALES_FORECAST_DATA_PATH", "/Volumes/MARAL")
PROJECT_ROOT = Path(__file__).parent

# Data source paths
DATA_PATHS = {
    "nachfrage": {
        "F01": f"{BASE_DATA_PATH}/CSV/F01/V2SC1010.csv",
        "F02": f"{BASE_DATA_PATH}/CSV/F02/V2SC1010.csv",
        "F03": f"{BASE_DATA_PATH}/CSV/F03/V2SC1010.csv",
        "F04": f"{BASE_DATA_PATH}/CSV/F04/V2SC1010.csv",
    },
    "marketing_artikel": f"{BASE_DATA_PATH}/CSV/F01/V2AR1001.csv",  # BANUMMER, WARENGR columns
    "product_descriptions": f"{BASE_DATA_PATH}/CSV/F01/V2AR1002.csv",  # NUMMER, BANAME1, BANAME2 columns
    "lager_history": f"{BASE_DATA_PATH}/lager_history/",  # To be defined
    "catalog_data": f"{BASE_DATA_PATH}/catalog_data/",  # To be defined
    "processed": f"{BASE_DATA_PATH}/Data/sales_forecast/",
    "models": PROJECT_ROOT / "models",
    "cache": PROJECT_ROOT / "cache",
}

# Model configuration
MODEL_CONFIG = {
    "horizon_months": 6,
    "min_history_months": 6,
    "covid_period": ("2020-01-01", "2022-12-31"),
    "covid_normalization_factor": 0.8,
    "ensemble_models": ["xgboost", "lightgbm", "prophet"],
    "cross_validation_folds": 3,
    "hyperparameter_trials": 50,
}

# Feature engineering settings
FEATURE_CONFIG = {
    "lag_features": [1, 3, 6, 12],
    "rolling_windows": [3, 6, 12],
    "seasonality_detection": {
        "method": "fft",  # or 'seasonal_decompose'
        "min_periods": 24,
    },
    "price_change_threshold": 0.1,  # 10% price change
    "stock_out_threshold": 0.05,  # 5% of average sales
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "host": "0.0.0.0",  # For server deployment
    "port": 8501,
    "page_title": "Demand Forecasting App",
    "layout": "wide",
    "cache_ttl": 3600,  # 1 hour cache
    "max_upload_size": 200,  # MB
}

# Database configuration (if needed later)
DATABASE_CONFIG = {
    "type": "sqlite",  # or 'postgresql', 'mysql'
    "path": PROJECT_ROOT / "data" / "sales_forecast.db",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log",
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "data_paths": DATA_PATHS,
        "model": MODEL_CONFIG,
        "features": FEATURE_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
        "database": DATABASE_CONFIG,
        "logging": LOGGING_CONFIG,
    }


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs_to_create = [
        DATA_PATHS["models"],
        DATA_PATHS["cache"],
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "data",
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
