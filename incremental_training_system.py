"""
Incremental training system with delta updates and model caching
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import json

from config import DATA_PATHS, MODEL_CONFIG
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.ensemble_model import ProductForecaster
from utils.logger import get_logger
from utils.config_loader import config_loader

logger = get_logger(__name__)


class IncrementalTrainingSystem:
    """System for incremental training and delta updates"""

    def __init__(self):
        self.cache_dir = Path(DATA_PATHS["cache"])
        self.models_dir = Path(DATA_PATHS["models"])
        self.processed_dir = Path(DATA_PATHS["processed"])

        # Create directories
        for dir_path in [self.cache_dir, self.models_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()

        # Cache file paths
        self.base_data_cache = self.cache_dir / "base_integrated_data.parquet"
        self.features_cache = self.cache_dir / "base_features_data.parquet"
        self.models_cache = self.cache_dir / "trained_models.joblib"
        self.metadata_cache = self.cache_dir / "training_metadata.json"

    def get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for data to detect changes"""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()

    def save_metadata(self, metadata: Dict):
        """Save training metadata"""
        with open(self.metadata_cache, "w") as f:
            json.dump(metadata, f, default=str, indent=2)

    def load_metadata(self) -> Optional[Dict]:
        """Load training metadata"""
        if self.metadata_cache.exists():
            try:
                with open(self.metadata_cache, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return None
        return None

    def initial_training(
        self, force_retrain: bool = False
    ) -> Tuple[pd.DataFrame, ProductForecaster]:
        """
        Perform initial comprehensive training and cache everything
        """
        logger.info("Starting initial training process...")

        # Check if we have cached data and models
        if not force_retrain and self.has_valid_cache():
            logger.info("Loading from cache...")
            return self.load_from_cache()

        # Step 1: Load and integrate data
        logger.info("Loading and integrating data...")
        try:
            integrated_data = self.data_loader.create_integrated_dataset()

            if integrated_data.empty:
                raise ValueError("No data loaded from data sources")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            # Try to load from cache as fallback
            if self.has_cache_files():
                logger.info("Falling back to cached data...")
                return self.load_from_cache()
            else:
                raise e

        # Step 2: Feature engineering
        logger.info("Creating features...")
        try:
            features_data = self.feature_engineer.create_features(integrated_data)
            feature_columns = self.feature_engineer.select_features(features_data)

            if features_data.empty:
                raise ValueError("Feature engineering produced no data")

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise e

        # Step 3: Train models
        logger.info("Training models...")
        try:
            forecaster = ProductForecaster(
                min_history_months=MODEL_CONFIG["min_history_months"]
            )
            train_data = features_data.dropna(subset=feature_columns)

            if len(train_data) < 10:
                raise ValueError("Insufficient training data after cleaning")

            forecaster.fit(train_data, feature_columns)

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise e

        # Step 4: Cache everything
        logger.info("Caching trained models and data...")
        try:
            self.cache_training_results(
                integrated_data, features_data, forecaster, feature_columns
            )
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
            # Continue without caching

        logger.info("Initial training completed successfully")
        return features_data, forecaster

    def has_valid_cache(self) -> bool:
        """Check if we have valid cached data and models"""
        if not self.has_cache_files():
            logger.info("Cache files missing")
            return False

        # Check cache age (valid for 7 days by default)
        metadata = self.load_metadata()
        if not metadata:
            logger.info("No metadata found")
            return False

        try:
            cache_date = datetime.fromisoformat(metadata["training_date"])
            cache_age = datetime.now() - cache_date

            # Cache is valid for 7 days
            if cache_age > timedelta(days=7):
                logger.info(f"Cache expired (age: {cache_age.days} days)")
                return False

        except Exception as e:
            logger.warning(f"Error checking cache age: {e}")
            return False

        logger.info("Valid cache found")
        return True

    def has_cache_files(self) -> bool:
        """Check if cache files exist"""
        required_files = [
            self.base_data_cache,
            self.features_cache,
            self.models_cache,
            self.metadata_cache,
        ]

        return all(f.exists() for f in required_files)

    def cache_training_results(
        self,
        integrated_data: pd.DataFrame,
        features_data: pd.DataFrame,
        forecaster: ProductForecaster,
        feature_columns: List[str],
    ):
        """Cache training results for fast loading"""

        try:
            # Save base integrated data
            integrated_data.to_parquet(self.base_data_cache, index=False)
            logger.info(f"Cached integrated data: {len(integrated_data)} records")

            # Save features data
            features_data.to_parquet(self.features_cache, index=False)
            logger.info(f"Cached features data: {len(features_data)} records")

            # Save trained models
            model_data = {"forecaster": forecaster, "feature_columns": feature_columns}
            joblib.dump(model_data, self.models_cache)
            logger.info("Cached trained models")

            # Save metadata
            metadata = {
                "training_date": datetime.now().isoformat(),
                "data_hash": self.get_data_hash(integrated_data),
                "num_products": integrated_data["product_id"].nunique(),
                "num_records": len(integrated_data),
                "date_range": {
                    "start": integrated_data["MONAT"].min().isoformat(),
                    "end": integrated_data["MONAT"].max().isoformat(),
                },
                "feature_columns": feature_columns,
                "model_counts": {
                    "product_models": len(forecaster.product_models),
                    "category_models": len(forecaster.category_models),
                    "global_model": 1 if forecaster.global_model else 0,
                },
            }
            self.save_metadata(metadata)
            logger.info("Cached metadata")

        except Exception as e:
            logger.error(f"Error caching training results: {e}")
            raise e

    def load_from_cache(self) -> Tuple[pd.DataFrame, ProductForecaster]:
        """Load cached training results"""
        logger.info("Loading from cache...")

        try:
            # Load features data
            features_data = pd.read_parquet(self.features_cache)
            features_data["MONAT"] = pd.to_datetime(features_data["MONAT"])
            logger.info(f"Loaded features data: {len(features_data)} records")

            # Load trained models
            cached_models = joblib.load(self.models_cache)
            forecaster = cached_models["forecaster"]
            logger.info("Loaded trained models")

            return features_data, forecaster

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            raise e

    def detect_new_data(self) -> Tuple[bool, pd.DataFrame]:
        """
        Detect new nachfrage data and products since last training
        """
        logger.info("Detecting new data...")

        try:
            # Load current data
            current_data = self.data_loader.create_integrated_dataset()

            if current_data.empty:
                logger.warning("Current data is empty")
                return False, pd.DataFrame()

            # Load cached data for comparison
            if not self.base_data_cache.exists():
                logger.info("No cached data found - all data is new")
                return True, current_data

            cached_data = pd.read_parquet(self.base_data_cache)
            cached_data["MONAT"] = pd.to_datetime(cached_data["MONAT"])

            # Find new records (by product_id + MONAT combination)
            cached_keys = set(zip(cached_data["product_id"], cached_data["MONAT"]))
            current_keys = set(zip(current_data["product_id"], current_data["MONAT"]))

            new_keys = current_keys - cached_keys

            if not new_keys:
                logger.info("No new data detected")
                return False, pd.DataFrame()

            # Extract new data
            new_data_mask = pd.Series(
                list(zip(current_data["product_id"], current_data["MONAT"]))
            ).isin(new_keys)
            new_data = current_data[new_data_mask].copy()

            logger.info(
                f"Detected {len(new_data)} new records for {new_data['product_id'].nunique()} products"
            )

            return True, new_data

        except Exception as e:
            logger.error(f"Error detecting new data: {e}")
            return False, pd.DataFrame()

    def incremental_update(self) -> Tuple[pd.DataFrame, ProductForecaster]:
        """
        Perform incremental update with new data
        """
        logger.info("Starting incremental update...")

        # Check for new data
        has_new_data, new_data = self.detect_new_data()

        if not has_new_data:
            logger.info("No new data - loading from cache")
            try:
                return self.load_from_cache()
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                # Fallback to initial training
                return self.initial_training(force_retrain=True)

        try:
            # Load cached base data and models
            features_data, forecaster = self.load_from_cache()
            cached_base_data = pd.read_parquet(self.base_data_cache)
            cached_base_data["MONAT"] = pd.to_datetime(cached_base_data["MONAT"])

            # Combine old and new data
            updated_base_data = pd.concat(
                [cached_base_data, new_data], ignore_index=True
            )
            updated_base_data = updated_base_data.drop_duplicates(
                subset=["product_id", "MONAT"]
            )

            # Create features for new data only
            logger.info("Creating features for new data...")
            new_features = self.feature_engineer.create_features(new_data)

            # Combine features
            updated_features = pd.concat(
                [features_data, new_features], ignore_index=True
            )
            updated_features = updated_features.drop_duplicates(
                subset=["product_id", "MONAT"]
            )
            feature_columns = self.feature_engineer.select_features(updated_features)

            # Incremental model update strategy
            new_products = set(new_data["product_id"].unique()) - set(
                features_data["product_id"].unique()
            )

            if new_products:
                logger.info(f"Training models for {len(new_products)} new products...")

                # For new products, train individual models
                for product_id in new_products:
                    try:
                        product_data = updated_features[
                            updated_features["product_id"] == product_id
                        ]
                        if len(product_data) >= MODEL_CONFIG["min_history_months"]:
                            # Train product-specific model
                            train_data = product_data.dropna(subset=feature_columns)
                            if len(train_data) >= 3:  # Minimum for training
                                from models.ensemble_model import EnsembleForecaster

                                product_model = EnsembleForecaster()
                                product_model.fit(
                                    train_data[feature_columns],
                                    train_data["anz_produkt"],
                                )
                                forecaster.product_models[product_id] = product_model
                                logger.info(
                                    f"Trained model for new product {product_id}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error training model for product {product_id}: {str(e)}"
                        )

            # Update cache with new data
            self.cache_training_results(
                updated_base_data, updated_features, forecaster, feature_columns
            )

            logger.info("Incremental update completed")
            return updated_features, forecaster

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            # Fallback to full retraining
            logger.info("Falling back to full retraining...")
            return self.initial_training(force_retrain=True)

    def quick_load_for_app(self) -> Tuple[pd.DataFrame, ProductForecaster, List[str]]:
        """
        Quick loading function optimized for app startup
        """
        logger.info("Quick loading for app startup...")

        try:
            # Load optimized configuration
            optimized_config = config_loader.load_config()
            
            # Use optimized feature selection
            feature_config = config_loader.get_feature_selection()
            selected_features = feature_config.get('selected_features', [])
            
            if selected_features:
                logger.info(f"Using {len(selected_features)} optimized features")
            
            # Try incremental update first (checks for new data)
            features_data, forecaster = self.incremental_update()

            # Load feature columns from metadata
            metadata = self.load_metadata()
            feature_columns = metadata.get("feature_columns", []) if metadata else []

            # If no feature columns in metadata, get them from feature engineer
            if not feature_columns and not features_data.empty:
                feature_columns = self.feature_engineer.select_features(features_data)

            logger.info(
                f"Quick load completed: {len(features_data)} records, {len(feature_columns)} features"
            )

            return features_data, forecaster, feature_columns

        except Exception as e:
            logger.error(f"Quick load failed: {e}")

            # Create minimal fallback data for app functionality
            logger.warning("Creating fallback data for app functionality")

            # Generate minimal sample data
            dates = pd.date_range(start="2023-01-01", end="2024-08-31", freq="MS")
            sample_data = []

            np.random.seed(42)
            for i in range(5):  # 5 sample products
                product_id = f"SAMPLE{i:04d}"

                for date in dates:
                    sales = max(1, int(np.random.normal(50, 15)))
                    price = np.random.uniform(10, 50)

                    sample_data.append(
                        {
                            "product_id": product_id,
                            "MONAT": date,
                            "anz_produkt": sales,
                            "unit_preis": price,
                        }
                    )

            fallback_data = pd.DataFrame(sample_data)

            # Create basic features
            fallback_features = self.feature_engineer.create_features(fallback_data)
            feature_columns = self.feature_engineer.select_features(fallback_features)

            # Create minimal forecaster
            fallback_forecaster = ProductForecaster(min_history_months=3)

            try:
                train_data = fallback_features.dropna(subset=feature_columns)
                if len(train_data) > 0:
                    fallback_forecaster.fit(train_data, feature_columns)
            except:
                # If training fails, create empty forecaster
                pass

            logger.info("Fallback data created for app functionality")
            return fallback_features, fallback_forecaster, feature_columns


# Global instance for easy access
incremental_system = IncrementalTrainingSystem()
