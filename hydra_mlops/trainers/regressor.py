"""
Regressor Model Trainer for LanceDB MLOps Framework.

This module provides a trainer for regression models, supporting
various sklearn regressors with standardized interfaces for training,
evaluation, and model persistence.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

from . import BaseModelTrainer

logger = logging.getLogger(__name__)

class RegressorTrainer(BaseModelTrainer):
    """
    Trainer for regression models.
    
    Supports various sklearn regressors with a standardized interface.
    """
    
    def __init__(
        self,
        regressor_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the regressor trainer.
        
        Args:
            regressor_type: Type of regressor to use (random_forest, linear, ridge, lasso, elastic_net, svr, gradient_boosting)
            model_params: Parameters for the regressor
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__(regressor_type, model_params)
        self.regressor_type = regressor_type
        self.test_size = test_size
        self.random_state = random_state
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the regressor based on the specified type.
        """
        model_params = self.model_params or {}
        
        # Ensure random_state is set for reproducibility where applicable
        if "random_state" not in model_params and self.regressor_type not in ["linear", "svr"]:
            model_params["random_state"] = self.random_state
            
        if self.regressor_type == "random_forest":
            self.model = RandomForestRegressor(**model_params)
        elif self.regressor_type == "linear":
            self.model = LinearRegression(**model_params)
        elif self.regressor_type == "ridge":
            self.model = Ridge(**model_params)
        elif self.regressor_type == "lasso":
            self.model = Lasso(**model_params)
        elif self.regressor_type == "elastic_net":
            self.model = ElasticNet(**model_params)
        elif self.regressor_type == "svr":
            self.model = SVR(**model_params)
        elif self.regressor_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(**model_params)
        else:
            raise ValueError(f"Unsupported regressor type: {self.regressor_type}")
        
        logger.info(f"Initialized {self.regressor_type} regressor with parameters: {model_params}")
        
    def train(
        self,
        features: Union[np.ndarray, List],
        targets: Union[np.ndarray, List],
        validation_split: bool = True
    ) -> Any:
        """
        Train the regressor on the provided features and targets.
        
        Args:
            features: Feature vectors for training
            targets: Target values for training
            validation_split: Whether to split data for validation
            
        Returns:
            Any: The trained model
        """
        try:
            # Convert inputs to numpy arrays if they aren't already
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if not isinstance(targets, np.ndarray):
                targets = np.array(targets)
                
            # Split data if validation is requested
            if validation_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, targets, test_size=self.test_size, random_state=self.random_state
                )
                
                logger.info(f"Training {self.regressor_type} on {len(X_train)} samples, validating on {len(X_val)} samples")
                
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_metrics = self.evaluate(X_val, y_val)
                logger.info(f"Validation metrics: {val_metrics}")
            else:
                logger.info(f"Training {self.regressor_type} on all {len(features)} samples")
                
                # Train on all data
                self.model.fit(features, targets)
            
            return self.model
        except Exception as e:
            logger.error(f"Error training regressor: {str(e)}")
            raise
            
    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained regressor on the provided features and targets.
        
        Args:
            features: Feature vectors for evaluation
            targets: Target values for evaluation
            
        Returns:
            Dict[str, float]: Metrics for the evaluated regressor
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            # Make predictions
            predictions = self.model.predict(features)
            
            # Calculate metrics
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mean_absolute_error(targets, predictions)),
                "r2": float(r2_score(targets, predictions)),
                "explained_variance": float(explained_variance_score(targets, predictions)),
            }
            
            # Add mean and median absolute percentage error
            if np.any(targets != 0):  # Avoid division by zero
                abs_percentage_error = np.abs((targets - predictions) / np.where(targets != 0, targets, 1))
                metrics["mape"] = float(np.mean(abs_percentage_error) * 100)  # Mean absolute percentage error
                metrics["mdape"] = float(np.median(abs_percentage_error) * 100)  # Median absolute percentage error
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating regressor: {str(e)}")
            raise
            
    def save(self, path: str) -> None:
        """
        Save the trained regressor to disk.
        
        Args:
            path: Path to save the model to
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            if path.endswith('.joblib'):
                joblib.dump(self.model, path)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                    
            logger.info(f"Saved regressor to {path}")
        except Exception as e:
            logger.error(f"Error saving regressor: {str(e)}")
            raise
            
    def load(self, path: str) -> Any:
        """
        Load a trained regressor from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Any: The loaded model
        """
        try:
            # Load the model
            if path.endswith('.joblib'):
                self.model = joblib.load(path)
            else:
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                    
            logger.info(f"Loaded regressor from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading regressor: {str(e)}")
            raise
            
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained regressor.
        
        Args:
            features: Feature vectors for prediction
            
        Returns:
            np.ndarray: Predicted values
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            return self.model.predict(features)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
