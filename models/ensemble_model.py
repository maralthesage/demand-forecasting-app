"""
Ensemble forecasting model combining multiple algorithms
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Model imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

from models.base_model import BaseForecaster
from utils.logger import get_logger

logger = get_logger(__name__)

class EnsembleForecaster(BaseForecaster):
    """Ensemble model combining multiple forecasting algorithms"""
    
    def __init__(self, models_config: Optional[Dict] = None):
        super().__init__("EnsembleForecaster")
        
        # Default model configuration
        self.models_config = models_config or {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        self.models = {}
        self.weights = {}
        self.cv_scores = {}
        
    def _initialize_models(self):
        """Initialize individual models"""
        self.models = {}
        
        # XGBoost
        if 'xgboost' in self.models_config:
            self.models['xgboost'] = xgb.XGBRegressor(**self.models_config['xgboost'])
        
        # LightGBM
        if 'lightgbm' in self.models_config:
            self.models['lightgbm'] = lgb.LGBMRegressor(**self.models_config['lightgbm'])
        
        # Random Forest
        if 'random_forest' in self.models_config:
            self.models['random_forest'] = RandomForestRegressor(**self.models_config['random_forest'])
        
        # Linear model (for baseline)
        self.models['linear'] = LinearRegression()
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleForecaster':
        """
        Fit ensemble model with cross-validation for weight determination
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble model on {len(X)} samples with {len(X.columns)} features")
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Initialize models
        self._initialize_models()
        
        # Prepare data
        X_array = X.values
        y_array = y.values
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train individual models and calculate CV scores
        cv_scores = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Cross-validation scores
                scores = cross_val_score(
                    model, X_array, y_array, 
                    cv=tscv, scoring='neg_mean_absolute_error',
                    n_jobs=1  # Avoid nested parallelism
                )
                cv_scores[model_name] = -scores.mean()  # Convert back to positive MAE
                
                # Fit on full dataset
                model.fit(X_array, y_array)
                
                logger.info(f"{model_name} CV MAE: {cv_scores[model_name]:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                # Remove failed model
                del self.models[model_name]
        
        # Calculate ensemble weights based on inverse of CV scores
        self.cv_scores = cv_scores
        total_inverse_error = sum(1/score for score in cv_scores.values() if score > 0)
        
        self.weights = {}
        for model_name, score in cv_scores.items():
            if score > 0:
                self.weights[model_name] = (1/score) / total_inverse_error
            else:
                self.weights[model_name] = 0
        
        logger.info(f"Ensemble weights: {self.weights}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self.validate_input(X)
        X_array = X.values
        
        # Get predictions from each model
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_array)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {str(e)}")
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self.validate_input(X)
        X_array = X.values
        
        # Get predictions from each model
        all_predictions = []
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_array)
                all_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {str(e)}")
        
        if not all_predictions:
            raise ValueError("No models available for prediction")
        
        # Convert to array
        all_predictions = np.array(all_predictions)
        
        # Ensemble prediction (weighted average)
        weights_array = np.array([self.weights.get(name, 0) for name in self.models.keys()])
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights_array)
        
        # Uncertainty estimates based on model agreement
        prediction_std = np.std(all_predictions, axis=0)
        
        # Lower and upper bounds (95% confidence interval approximation)
        lower_bound = ensemble_pred - 1.96 * prediction_std
        upper_bound = ensemble_pred + 1.96 * prediction_std
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        return ensemble_pred, lower_bound, upper_bound
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance across models
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_fitted:
            return {}
        
        feature_importance = {}
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0)
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
            else:
                continue
            
            for i, importance in enumerate(importances):
                feature_name = self.feature_columns[i]
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = 0
                feature_importance[feature_name] += weight * importance
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get individual model CV performance"""
        return self.cv_scores.copy()

class ProductForecaster:
    """Product-specific forecasting with fallback strategies"""
    
    def __init__(self, min_history_months: int = 6):
        self.min_history_months = min_history_months
        self.product_models = {}
        self.category_models = {}
        self.global_model = None
        
    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Fit models at different levels (product, category, global)
        
        Args:
            df: Training data
            feature_columns: List of feature column names
        """
        logger.info("Training product forecasting models...")
        
        # Prepare features and target
        X = df[feature_columns]
        y = df['anz_produkt']
        
        # Train global model (fallback for new products)
        logger.info("Training global model...")
        self.global_model = EnsembleForecaster()
        self.global_model.fit(X, y)
        
        # Train category models
        if 'product_category_id' in df.columns:
            logger.info("Training category models...")
            for category in df['product_category_id'].unique():
                if pd.isna(category):
                    continue
                
                cat_data = df[df['product_category_id'] == category]
                if len(cat_data) >= self.min_history_months:
                    try:
                        cat_model = EnsembleForecaster()
                        cat_model.fit(cat_data[feature_columns], cat_data['anz_produkt'])
                        self.category_models[category] = cat_model
                        logger.info(f"Trained model for category {category}")
                    except Exception as e:
                        logger.error(f"Error training category model for {category}: {str(e)}")
        
        # Train product-specific models
        logger.info("Training product-specific models...")
        product_counts = df.groupby('product_id').size()
        
        for product_id in product_counts[product_counts >= self.min_history_months].index:
            try:
                product_data = df[df['product_id'] == product_id]
                product_model = EnsembleForecaster()
                product_model.fit(product_data[feature_columns], product_data['anz_produkt'])
                self.product_models[product_id] = product_model
            except Exception as e:
                logger.error(f"Error training model for product {product_id}: {str(e)}")
        
        logger.info(f"Trained {len(self.product_models)} product models, "
                   f"{len(self.category_models)} category models, and 1 global model")
    
    def predict(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Make predictions using appropriate model level
        
        Args:
            df: Data to predict on
            feature_columns: Feature column names
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for _, row in df.iterrows():
            product_id = row['product_id']
            category_id = row.get('product_category_id')
            
            # Try product-specific model first
            if product_id in self.product_models:
                model = self.product_models[product_id]
                model_level = 'product'
            # Then try category model
            elif category_id in self.category_models:
                model = self.category_models[category_id]
                model_level = 'category'
            # Finally use global model
            else:
                model = self.global_model
                model_level = 'global'
            
            # Make prediction
            X_row = row[feature_columns].values.reshape(1, -1)
            X_df = pd.DataFrame(X_row, columns=feature_columns)
            
            pred, lower, upper = model.predict_with_uncertainty(X_df)
            
            results.append({
                'product_id': product_id,
                'MONAT': row['MONAT'],
                'predicted_sales': pred[0],
                'lower_bound': lower[0],
                'upper_bound': upper[0],
                'model_level': model_level
            })
        
        return pd.DataFrame(results)
