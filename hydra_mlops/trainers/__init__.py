"""
Model trainers for the LanceDB MLOps Framework.

This module provides base classes and implementations for model trainers
that can be used to train machine learning models on data stored in LanceDB.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import os
import logging
import pandas as pd
import numpy as np
import json
import datetime

logger = logging.getLogger(__name__)

class ModelTrainer(ABC):
    """
    Abstract base class for all model trainers.
    
    Model trainers are responsible for training machine learning models,
    evaluating their performance, and saving them to disk.
    """
    
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> Any:
        """
        Train a model on the provided features and labels.
        
        Args:
            features: Feature vectors for training
            labels: Target labels for training
            
        Returns:
            Any: The trained model
        """
        pass
    
    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model on the provided features and labels.
        
        Args:
            features: Feature vectors for evaluation
            labels: Target labels for evaluation
            
        Returns:
            Dict[str, float]: Metrics for the evaluated model
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model to
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Any: The loaded model
        """
        pass

class BaseModelTrainer(ModelTrainer):
    """
    Base implementation of the ModelTrainer with common functionality.
    """
    
    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
            model_params: Parameters for model initialization
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        
    def register_model(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float],
        model_path: str,
        model_vector: np.ndarray,
        lancedb_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a trained model in the LanceDB model registry.
        
        Args:
            model_id: ID for the model
            version: Version string for the model
            metrics: Evaluation metrics for the model
            model_path: Path where the model is saved
            model_vector: Vector representation of the model
            lancedb_path: Path to the LanceDB database
            metadata: Additional metadata to store with the model
        """
        try:
            import lancedb
            import pandas as pd
            
            db = lancedb.connect(lancedb_path)
            
            # Create model registry table if it doesn't exist
            if "model_registry" not in db.table_names():
                import pandas as pd
                empty_df = pd.DataFrame({
                    "model_id": [],
                    "version": [],
                    "created_at": [],
                    "model_type": [],
                    "metrics": [],
                    "model_path": [],
                    "metadata": [],
                    "vector": []
                })
                table = db.create_table("model_registry", data=empty_df)
            else:
                table = db.open_table("model_registry")
            
            # Add model record
            created_at = datetime.datetime.now().isoformat()
            
            # Convert metrics to string
            metrics_str = json.dumps(metrics)
            
            # Convert metadata to string
            metadata_dict = metadata or {}
            metadata_dict.update({
                "model_type": self.model_type,
                "model_params": self.model_params,
                "created_at": created_at
            })
            metadata_str = json.dumps(metadata_dict)
            
            # Create model record
            model_record = pd.DataFrame([{
                "model_id": model_id,
                "version": version,
                "created_at": created_at,
                "model_type": self.model_type,
                "metrics": metrics_str,
                "model_path": model_path,
                "metadata": metadata_str,
                "vector": model_vector
            }])
            
            # Add to table
            table.add(model_record)
            logger.info(f"Registered model {model_id} version {version} in LanceDB registry")
        except Exception as e:
            logger.error(f"Error registering model in LanceDB: {str(e)}")
            raise

# Import specific trainer implementations
from .classifier import ClassifierTrainer
from .regressor import RegressorTrainer

# Register available trainers
AVAILABLE_TRAINERS = {
    "classifier": ClassifierTrainer,
    "regressor": RegressorTrainer,
}

def get_trainer(trainer_type: str, **params):
    """
    Factory function to get a trainer instance by type.
    
    Args:
        trainer_type: Type of trainer to create.
        **params: Parameters to pass to the trainer constructor.
        
    Returns:
        ModelTrainer: An instance of the requested trainer.
        
    Raises:
        ValueError: If the trainer type is not recognized.
    """
    if trainer_type not in AVAILABLE_TRAINERS:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available types: {list(AVAILABLE_TRAINERS.keys())}")
    
    trainer_class = AVAILABLE_TRAINERS[trainer_type]
    return trainer_class(**params)
