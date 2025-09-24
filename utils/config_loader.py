"""
Configuration loader for optimized hyperparameters
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class OptimizedConfigLoader:
    """Load optimized hyperparameters from JSON file"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default path: look for config in notebooks folder
            project_root = Path(__file__).parent.parent
            config_path = project_root / "notebooks" / "optimized_hyperparameters.json"
        
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load optimized configuration from JSON file"""
        if self._config is not None:
            return self._config
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded optimized configuration from {self.config_path}")
                return self._config
            else:
                logger.warning(f"Optimized config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading optimized config: {e}")
            return self._get_default_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get optimized model hyperparameters"""
        config = self.load_config()
        return config.get('model_hyperparameters', {})
    
    def get_feature_selection(self) -> Dict[str, Any]:
        """Get optimized feature selection"""
        config = self.load_config()
        return config.get('feature_selection', {})
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get optimized ensemble weights"""
        config = self.load_config()
        return config.get('ensemble_weights', {})
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from optimization"""
        config = self.load_config()
        return config.get('performance_metrics', {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if optimized config not available"""
        return {
            'model_hyperparameters': {
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                },
                'lightgbm': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbosity': -1,
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1,
                }
            },
            'ensemble_weights': {
                'xgboost': 0.4,
                'lightgbm': 0.4,
                'random_forest': 0.2
            },
            'feature_selection': {
                'selected_features': [],
                'max_features': 15
            }
        }

# Global instance
config_loader = OptimizedConfigLoader()