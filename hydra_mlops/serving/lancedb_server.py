"""
LanceDB Model Server for MLOps Framework.

This module implements a model server using LanceDB as the backend for
registry access and prediction logging.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import datetime
import hashlib

from . import ModelServer
from hydra_mlops.registry import LanceDBModelRegistry

logger = logging.getLogger(__name__)

class LanceDBModelServer(ModelServer):
    """
    LanceDB implementation of the model server.
    
    Uses LanceDB's vector database capabilities for efficient model
    loading and prediction logging.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the LanceDB model server.
        
        Args:
            db_path: Path to the LanceDB database
        """
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "~/.lancedb")
        self.registry = LanceDBModelRegistry(self.db_path)
        self.loaded_models = {}
        self._db = None
        self._logs_table = None
        
    @property
    def db(self):
        """
        Get the LanceDB database connection.
        
        Returns:
            lancedb.db.LanceDB: Database connection
        """
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    @property
    def logs_table(self):
        """
        Get the prediction logs table.
        
        Returns:
            lancedb.table.Table: Prediction logs table
        """
        if self._logs_table is None:
            # Create table if it doesn't exist
            if "prediction_logs" not in self.db.table_names():
                empty_df = pd.DataFrame({
                    "id": [],
                    "timestamp": [],
                    "model_id": [],
                    "version": [],
                    "features": [],
                    "predictions": [],
                    "ground_truth": [],
                    "metadata": [],
                    "vector": []
                })
                self._logs_table = self.db.create_table("prediction_logs", data=empty_df)
            else:
                self._logs_table = self.db.open_table("prediction_logs")
        return self._logs_table
    
    def load_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model to load
            version: Specific version to load, or None for latest
            
        Returns:
            Dict[str, Any]: Loaded model and metadata
        """
        try:
            # Check if model is already loaded
            model_key = f"{model_id}__{version or 'latest'}"
            if model_key in self.loaded_models:
                logger.info(f"Using cached model {model_id} version {version or 'latest'}")
                return self.loaded_models[model_key]
            
            # Get model metadata from registry
            model_info = self.registry.get_model(model_id, version)
            model_path = model_info["model_path"]
            
            # Load the model from disk
            if model_path.endswith('.joblib'):
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Add model to loaded models cache
            loaded_model = {
                "model": model,
                "model_id": model_id,
                "version": model_info["version"],
                "model_type": model_info["model_type"],
                "loaded_at": datetime.datetime.now().isoformat(),
                "metadata": model_info["metadata"],
            }
            self.loaded_models[model_key] = loaded_model
            
            logger.info(f"Loaded model {model_id} version {model_info['version']} from {model_path}")
            return loaded_model
        except Exception as e:
            logger.error(f"Error loading model from registry: {str(e)}")
            raise
    
    def predict(
        self,
        model_id: str,
        features: Union[np.ndarray, List],
        version: Optional[str] = None,
        log_prediction: bool = True,
    ) -> np.ndarray:
        """
        Make predictions using a model.
        
        Args:
            model_id: ID of the model to use
            features: Features to predict on
            version: Specific version to use, or None for latest
            log_prediction: Whether to log the prediction
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Ensure features is a numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Load the model
            loaded_model = self.load_model(model_id, version)
            model = loaded_model["model"]
            actual_version = loaded_model["version"]
            
            # Make predictions
            predictions = model.predict(features)
            
            # Log prediction if requested
            if log_prediction:
                self.log_prediction(model_id, actual_version, features, predictions)
            
            return predictions
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def log_prediction(
        self,
        model_id: str,
        version: str,
        features: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List],
        ground_truth: Optional[Union[np.ndarray, List]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            model_id: ID of the model used
            version: Version of the model used
            features: Features used for prediction
            predictions: Predictions made
            ground_truth: Optional ground truth values
            metadata: Additional metadata for the prediction
        """
        try:
            # Ensure inputs are numpy arrays
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if ground_truth is not None and not isinstance(ground_truth, np.ndarray):
                ground_truth = np.array(ground_truth)
            
            # Convert features to a vector (use mean if multiple)
            if len(features.shape) > 1 and features.shape[0] > 1:
                vector = np.mean(features, axis=0)
            else:
                vector = features.flatten()
            
            # Generate a unique ID for the prediction
            timestamp = datetime.datetime.now().isoformat()
            id_hash = hashlib.md5(f"{model_id}_{version}_{timestamp}".encode()).hexdigest()
            
            # Convert arrays to JSON strings
            features_json = json.dumps(features.tolist())
            predictions_json = json.dumps(predictions.tolist())
            ground_truth_json = json.dumps(ground_truth.tolist()) if ground_truth is not None else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Create log entry
            log_entry = pd.DataFrame([{
                "id": id_hash,
                "timestamp": timestamp,
                "model_id": model_id,
                "version": version,
                "features": features_json,
                "predictions": predictions_json,
                "ground_truth": ground_truth_json,
                "metadata": metadata_json,
                "vector": vector
            }])
            
            # Add to logs table
            self.logs_table.add(log_entry)
            logger.debug(f"Logged prediction for model {model_id} version {version}")
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            # Don't raise the exception to avoid disrupting the prediction flow
            # Just log the error and continue
    
    def get_prediction_logs(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction logs for a model.
        
        Args:
            model_id: ID of the model to get logs for, or None for all models
            version: Specific version to get logs for, or None for all versions
            limit: Maximum number of logs to return
            
        Returns:
            List[Dict[str, Any]]: Prediction logs
        """
        try:
            # Query the logs table
            logs_df = self.logs_table.to_pandas()
            
            if logs_df.empty:
                return []
            
            # Filter by model_id and version if provided
            if model_id:
                logs_df = logs_df[logs_df["model_id"] == model_id]
            if version:
                logs_df = logs_df[logs_df["version"] == version]
            
            # Sort by timestamp (descending)
            logs_df = logs_df.sort_values("timestamp", ascending=False)
            
            # Limit the number of logs
            logs_df = logs_df.head(limit)
            
            # Build result
            result = []
            for _, log in logs_df.iterrows():
                # Parse JSON fields
                features = json.loads(log["features"])
                predictions = json.loads(log["predictions"])
                ground_truth = json.loads(log["ground_truth"]) if log["ground_truth"] else None
                metadata = json.loads(log["metadata"]) if log["metadata"] else None
                
                # Build result
                log_info = {
                    "id": log["id"],
                    "timestamp": log["timestamp"],
                    "model_id": log["model_id"],
                    "version": log["version"],
                    "features": features,
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "metadata": metadata,
                }
                result.append(log_info)
            
            return result
        except Exception as e:
            logger.error(f"Error getting prediction logs: {str(e)}")
            raise
    
    def predict_proba(
        self,
        model_id: str,
        features: Union[np.ndarray, List],
        version: Optional[str] = None,
        log_prediction: bool = True,
    ) -> np.ndarray:
        """
        Get probability estimates for predictions.
        
        Args:
            model_id: ID of the model to use
            features: Features to predict on
            version: Specific version to use, or None for latest
            log_prediction: Whether to log the prediction
            
        Returns:
            np.ndarray: Probability estimates
        """
        try:
            # Ensure features is a numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Load the model
            loaded_model = self.load_model(model_id, version)
            model = loaded_model["model"]
            actual_version = loaded_model["version"]
            
            # Check if model supports predict_proba
            if not hasattr(model, 'predict_proba'):
                raise ValueError(f"Model {model_id} does not support probability estimates")
            
            # Make predictions
            probabilities = model.predict_proba(features)
            
            # Make class predictions for logging
            predictions = model.predict(features)
            
            # Log prediction if requested
            if log_prediction:
                metadata = {"includes_probabilities": True}
                self.log_prediction(model_id, actual_version, features, predictions, metadata=metadata)
            
            return probabilities
        except Exception as e:
            logger.error(f"Error getting probability estimates: {str(e)}")
            raise
    
    def get_model_performance(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for a model based on logged predictions.
        
        Args:
            model_id: ID of the model to evaluate
            version: Specific version to evaluate, or None for latest
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            # Get prediction logs with ground truth
            logs = self.get_prediction_logs(model_id, version)
            
            # Filter logs with ground truth
            logs_with_truth = [log for log in logs if log["ground_truth"] is not None]
            
            if not logs_with_truth:
                return {"error": "No prediction logs with ground truth available"}
            
            # Extract predictions and ground truth
            all_predictions = []
            all_ground_truth = []
            
            for log in logs_with_truth:
                all_predictions.extend(log["predictions"])
                all_ground_truth.extend(log["ground_truth"])
            
            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            ground_truth = np.array(all_ground_truth)
            
            # Calculate metrics based on model type
            model_info = self.registry.get_model(model_id, version)
            model_type = model_info["model_type"]
            
            if model_type in ["classifier", "random_forest", "logistic_regression", "svm"]:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    "accuracy": float(accuracy_score(ground_truth, predictions)),
                    "precision_macro": float(precision_score(ground_truth, predictions, average='macro', zero_division=0)),
                    "recall_macro": float(recall_score(ground_truth, predictions, average='macro', zero_division=0)),
                    "f1_macro": float(f1_score(ground_truth, predictions, average='macro', zero_division=0)),
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(ground_truth, predictions)
                metrics = {
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(mean_absolute_error(ground_truth, predictions)),
                    "r2": float(r2_score(ground_truth, predictions)),
                }
            
            # Add metadata
            metrics["num_samples"] = len(predictions)
            metrics["updated_at"] = datetime.datetime.now().isoformat()
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating model performance: {str(e)}")
            raise
