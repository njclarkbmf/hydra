"""
Model registry for the LanceDB MLOps Framework.

This module provides classes for storing and retrieving model metadata
and artifacts using LanceDB as the backend.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import os
import logging
import json
import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelRegistry(ABC):
    """
    Abstract base class for model registries.
    
    Model registries are responsible for storing and retrieving model
    metadata and artifacts.
    """
    
    @abstractmethod
    def register_model(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float],
        model_path: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            version: Version string
            metrics: Evaluation metrics
            model_path: Path to the model artifact
            vector: Vector representation of the model
            metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    def get_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a model from the registry.
        
        Args:
            model_id: ID of the model to retrieve
            version: Specific version to retrieve, or None for latest
            
        Returns:
            Dict[str, Any]: Model metadata and artifact path
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        pass
    
    @abstractmethod
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List[Dict[str, Any]]: List of version metadata
        """
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str, version: Optional[str] = None) -> None:
        """
        Delete a model (or specific version) from the registry.
        
        Args:
            model_id: ID of the model to delete
            version: Specific version to delete, or None for all versions
        """
        pass
    
    @abstractmethod
    def find_similar_models(self, vector: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find models similar to the given vector.
        
        Args:
            vector: Vector to compare against
            limit: Maximum number of models to return
            
        Returns:
            List[Dict[str, Any]]: List of similar models with similarity scores
        """
        pass

# Import specific registry implementations
from .lancedb_registry import LanceDBModelRegistry

# Default registry implementation
default_registry = LanceDBModelRegistry
