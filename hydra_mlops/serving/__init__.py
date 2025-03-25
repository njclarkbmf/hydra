"""
Model serving components for the LanceDB MLOps Framework.

This module provides classes for loading and serving machine learning models,
handling inference requests, and tracking model predictions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import os
import logging
import numpy as np
import datetime
import json

logger = logging.getLogger(__name__)

class ModelServer(ABC):
    """
    Abstract base class for model servers.
    
    Model servers are responsible for loading models from the registry,
    serving them for inference, and tracking predictions.
    """
    
    @abstractmethod
    def load_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model to load
            version: Specific version to load, or None for latest
            
        Returns:
            Dict[str, Any]: Loaded model and metadata
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        model_id: str,
        features: Union[np.ndarray, List],
        version: Optional[str] = None,
    ) -> np.ndarray:
        """
        Make predictions using a model.
        
        Args:
            model_id: ID of the model to use
            features: Features to predict on
            version: Specific version to use, or None for latest
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def log_prediction(
        self,
        model_id: str,
        version: str,
        features: Union[np.ndarray, List],
        predictions: Union[np.ndarray, List],
        ground_truth: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            model_id: ID of the model used
            version: Version of the model used
            features: Features used for prediction
            predictions: Predictions made
            ground_truth: Optional ground truth values
        """
        pass
    
    @abstractmethod
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
        pass

# Import specific server implementations
from .lancedb_server import LanceDBModelServer

# Default server implementation
default_server = LanceDBModelServer
