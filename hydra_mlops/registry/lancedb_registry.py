"""
LanceDB Model Registry for MLOps Framework.

This module implements a model registry using LanceDB as the backend,
providing efficient vector-based search and retrieval of models.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
import datetime
import numpy as np
import pandas as pd

from . import ModelRegistry

logger = logging.getLogger(__name__)

class LanceDBModelRegistry(ModelRegistry):
    """
    LanceDB implementation of the model registry.
    
    Uses LanceDB's vector database capabilities for efficient storage
    and retrieval of model metadata and artifacts.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the LanceDB model registry.
        
        Args:
            db_path: Path to the LanceDB database
        """
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "~/.lancedb")
        self._db = None
        self._table = None
        
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
    def table(self):
        """
        Get the model registry table.
        
        Returns:
            lancedb.table.Table: Model registry table
        """
        if self._table is None:
            # Create table if it doesn't exist
            if "model_registry" not in self.db.table_names():
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
                self._table = self.db.create_table("model_registry", data=empty_df)
            else:
                self._table = self.db.open_table("model_registry")
        return self._table
    
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
        Register a model in the LanceDB registry.
        
        Args:
            model_id: Unique identifier for the model
            version: Version string
            metrics: Evaluation metrics
            model_path: Path to the model artifact
            vector: Vector representation of the model
            metadata: Additional metadata
        """
        try:
            # Convert metrics to string
            metrics_str = json.dumps(metrics)
            
            # Convert metadata to string
            metadata_dict = metadata or {}
            metadata_str = json.dumps(metadata_dict)
            
            # Create model record
            created_at = datetime.datetime.now().isoformat()
            model_record = pd.DataFrame([{
                "model_id": model_id,
                "version": version,
                "created_at": created_at,
                "model_type": metadata_dict.get("model_type", "unknown"),
                "metrics": metrics_str,
                "model_path": model_path,
                "metadata": metadata_str,
                "vector": vector
            }])
            
            # Add to table
            self.table.add(model_record)
            logger.info(f"Registered model {model_id} version {version} in LanceDB registry")
        except Exception as e:
            logger.error(f"Error registering model in LanceDB: {str(e)}")
            raise
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a model from the registry.
        
        Args:
            model_id: ID of the model to retrieve
            version: Specific version to retrieve, or None for latest
            
        Returns:
            Dict[str, Any]: Model metadata and artifact path
        """
        try:
            # Query the registry
            models_df = self.table.to_pandas()
            
            # Filter by model_id
            model_versions = models_df[models_df["model_id"] == model_id]
            
            if model_versions.empty:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Get specific version or latest
            if version:
                version_record = model_versions[model_versions["version"] == version]
                if version_record.empty:
                    raise ValueError(f"Version {version} of model {model_id} not found in registry")
                model_record = version_record.iloc[0]
            else:
                # Get latest version by creation date
                model_record = model_versions.sort_values("created_at", ascending=False).iloc[0]
            
            # Parse JSON fields
            metrics = json.loads(model_record["metrics"])
            metadata = json.loads(model_record["metadata"])
            
            # Build result
            result = {
                "model_id": model_record["model_id"],
                "version": model_record["version"],
                "created_at": model_record["created_at"],
                "model_type": model_record["model_type"],
                "metrics": metrics,
                "model_path": model_record["model_path"],
                "metadata": metadata,
            }
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving model from LanceDB: {str(e)}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        try:
            # Query the registry
            models_df = self.table.to_pandas()
            
            if models_df.empty:
                return []
            
            # Group by model_id and get latest version
            result = []
            for model_id in models_df["model_id"].unique():
                model_versions = models_df[models_df["model_id"] == model_id]
                latest_version = model_versions.sort_values("created_at", ascending=False).iloc[0]
                
                # Parse JSON fields
                metrics = json.loads(latest_version["metrics"])
                metadata = json.loads(latest_version["metadata"])
                
                # Build result
                model_info = {
                    "model_id": latest_version["model_id"],
                    "version": latest_version["version"],
                    "created_at": latest_version["created_at"],
                    "model_type": latest_version["model_type"],
                    "metrics": metrics,
                    "model_path": latest_version["model_path"],
                    "metadata": metadata,
                }
                result.append(model_info)
            
            return result
        except Exception as e:
            logger.error(f"Error listing models from LanceDB: {str(e)}")
            raise
    
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List[Dict[str, Any]]: List of version metadata
        """
        try:
            # Query the registry
            models_df = self.table.to_pandas()
            
            # Filter by model_id
            model_versions = models_df[models_df["model_id"] == model_id]
            
            if model_versions.empty:
                return []
            
            # Sort by creation date
            model_versions = model_versions.sort_values("created_at", ascending=False)
            
            # Build result
            result = []
            for _, version in model_versions.iterrows():
                # Parse JSON fields
                metrics = json.loads(version["metrics"])
                metadata = json.loads(version["metadata"])
                
                # Build result
                version_info = {
                    "model_id": version["model_id"],
                    "version": version["version"],
                    "created_at": version["created_at"],
                    "model_type": version["model_type"],
                    "metrics": metrics,
                    "model_path": version["model_path"],
                    "metadata": metadata,
                }
                result.append(version_info)
            
            return result
        except Exception as e:
            logger.error(f"Error listing model versions from LanceDB: {str(e)}")
            raise
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> None:
        """
        Delete a model (or specific version) from the registry.
        
        Args:
            model_id: ID of the model to delete
            version: Specific version to delete, or None for all versions
        """
        try:
            # Query the registry
            models_df = self.table.to_pandas()
            
            # Filter by model_id
            if version:
                # Delete specific version
                condition = (models_df["model_id"] != model_id) | (models_df["version"] != version)
                filtered_df = models_df[condition]
                
                if len(filtered_df) == len(models_df):
                    logger.warning(f"Version {version} of model {model_id} not found in registry")
                    return
                
                # Re-create table with filtered data
                self._table = self.db.create_table("model_registry", data=filtered_df, mode="overwrite")
                logger.info(f"Deleted version {version} of model {model_id} from registry")
            else:
                # Delete all versions
                filtered_df = models_df[models_df["model_id"] != model_id]
                
                if len(filtered_df) == len(models_df):
                    logger.warning(f"Model {model_id} not found in registry")
                    return
                
                # Re-create table with filtered data
                self._table = self.db.create_table("model_registry", data=filtered_df, mode="overwrite")
                logger.info(f"Deleted all versions of model {model_id} from registry")
        except Exception as e:
            logger.error(f"Error deleting model from LanceDB: {str(e)}")
            raise
    
    def find_similar_models(self, vector: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find models similar to the given vector.
        
        Args:
            vector: Vector to compare against
            limit: Maximum number of models to return
            
        Returns:
            List[Dict[str, Any]]: List of similar models with similarity scores
        """
        try:
            # Search the registry using vector similarity
            search_results = self.table.search(vector).limit(limit).to_pandas()
            
            if search_results.empty:
                return []
            
            # Build result
            result = []
            for _, model in search_results.iterrows():
                # Parse JSON fields
                metrics = json.loads(model["metrics"])
                metadata = json.loads(model["metadata"])
                
                # Build result
                model_info = {
                    "model_id": model["model_id"],
                    "version": model["version"],
                    "created_at": model["created_at"],
                    "model_type": model["model_type"],
                    "metrics": metrics,
                    "model_path": model["model_path"],
                    "metadata": metadata,
                    "_distance": float(model["_distance"]) if "_distance" in model else None,
                }
                result.append(model_info)
            
            return result
        except Exception as e:
            logger.error(f"Error finding similar models in LanceDB: {str(e)}")
            raise
