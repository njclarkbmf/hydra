"""
LanceDB Model Monitor for MLOps Framework.

This module implements a model monitor using LanceDB as the backend for
tracking performance, detecting drift, and triggering retraining.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import json
import datetime
import requests
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

from . import ModelMonitor
from hydra_mlops.registry import LanceDBModelRegistry
from hydra_mlops.serving import LanceDBModelServer

logger = logging.getLogger(__name__)

class LanceDBModelMonitor(ModelMonitor):
    """
    LanceDB implementation of the model monitor.
    
    Uses LanceDB's vector database capabilities for efficient monitoring
    of model performance and data drift.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the LanceDB model monitor.
        
        Args:
            db_path: Path to the LanceDB database
        """
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "~/.lancedb")
        self.registry = LanceDBModelRegistry(self.db_path)
        self.server = LanceDBModelServer(self.db_path)
        self._db = None
        self._monitor_table = None
        
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
    def monitor_table(self):
        """
        Get the model monitoring table.
        
        Returns:
            lancedb.table.Table: Model monitoring table
        """
        if self._monitor_table is None:
            # Create table if it doesn't exist
            if "model_monitoring" not in self.db.table_names():
                empty_df = pd.DataFrame({
                    "id": [],
                    "timestamp": [],
                    "model_id": [],
                    "version": [],
                    "metric_type": [],
                    "metric_value": [],
                    "threshold": [],
                    "status": [],
                    "metadata": [],
                    "vector": []
                })
                self._monitor_table = self.db.create_table("model_monitoring", data=empty_df)
            else:
                self._monitor_table = self.db.open_table("model_monitoring")
        return self._monitor_table
    
    def detect_drift(
        self,
        model_id: str,
        reference_data: Union[pd.DataFrame, np.ndarray],
        current_data: Union[pd.DataFrame, np.ndarray],
        version: Optional[str] = None,
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            model_id: ID of the model to monitor
            reference_data: Reference data for comparison
            current_data: Current data to check for drift
            version: Specific version to monitor, or None for latest
            drift_threshold: p-value threshold for drift detection
            
        Returns:
            Dict[str, Any]: Drift metrics
        """
        try:
            # Get model metadata
            model_info = self.registry.get_model(model_id, version)
            actual_version = model_info["version"]
            
            # Extract vectors if input is DataFrame
            if isinstance(reference_data, pd.DataFrame) and "vector" in reference_data.columns:
                reference_vectors = np.stack(reference_data["vector"].to_numpy())
            else:
                reference_vectors = reference_data
                
            if isinstance(current_data, pd.DataFrame) and "vector" in current_data.columns:
                current_vectors = np.stack(current_data["vector"].to_numpy())
            else:
                current_vectors = current_data
            
            # Calculate vector statistics
            ref_mean = np.mean(reference_vectors, axis=0)
            current_mean = np.mean(current_vectors, axis=0)
            
            ref_std = np.std(reference_vectors, axis=0)
            current_std = np.std(current_vectors, axis=0)
            
            # Cosine similarity between means
            cos_sim = cosine_similarity([ref_mean], [current_mean])[0][0]
            
            # Calculate distribution drift using KS test on vector norms
            ref_norms = np.linalg.norm(reference_vectors, axis=1)
            current_norms = np.linalg.norm(current_vectors, axis=1)
            
            ks_stat, p_value = ks_2samp(ref_norms, current_norms)
            
            # Determine if drift is significant
            drift_detected = p_value < drift_threshold
            
            # Calculate dimension-wise drift for high-dimensional vectors
            dimension_drift = []
            if ref_mean.shape[0] <= 20:  # Only for reasonably sized vectors
                for i in range(ref_mean.shape[0]):
                    dim_ks_stat, dim_p_value = ks_2samp(reference_vectors[:, i], current_vectors[:, i])
                    dimension_drift.append({
                        "dimension": i,
                        "ks_statistic": float(dim_ks_stat),
                        "p_value": float(dim_p_value),
                        "drift_detected": dim_p_value < drift_threshold
                    })
            
            # Build result
            drift_metrics = {
                "model_id": model_id,
                "version": actual_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "cosine_similarity": float(cos_sim),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
                "drift_threshold": drift_threshold,
                "reference_size": len(reference_vectors),
                "current_size": len(current_vectors),
                "dimension_drift": dimension_drift if dimension_drift else None,
            }
            
            # Log the drift metrics
            self._log_metric(
                model_id=model_id,
                version=actual_version,
                metric_type="drift",
                metric_value=float(ks_stat),
                threshold=drift_threshold,
                status="alert" if drift_detected else "normal",
                metadata=drift_metrics,
                vector=current_mean,
            )
            
            return drift_metrics
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            raise
    
    def monitor_performance(
        self,
        model_id: str,
        threshold: Optional[float] = None,
        version: Optional[str] = None,
        metric_name: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Monitor model performance based on logged predictions.
        
        Args:
            model_id: ID of the model to monitor
            threshold: Performance threshold for alerting
            version: Specific version to monitor, or None for latest
            metric_name: Name of the metric to use for monitoring
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            # Get model metadata
            model_info = self.registry.get_model(model_id, version)
            actual_version = model_info["version"]
            
            # Get model type to determine default threshold
            model_type = model_info["model_type"]
            
            # Determine default threshold based on model type and metric
            if threshold is None:
                if model_type in ["classifier", "random_forest", "logistic_regression", "svm"]:
                    if metric_name == "accuracy":
                        threshold = 0.7
                    elif metric_name in ["precision", "recall", "f1"]:
                        threshold = 0.6
                    else:
                        threshold = 0.7
                else:  # regression models
                    if metric_name == "r2":
                        threshold = 0.7
                    elif metric_name in ["mse", "rmse", "mae"]:
                        # For error metrics, lower is better, so use a negative threshold
                        threshold = -float('inf')  # No alerting by default
                    else:
                        threshold = 0.7
            
            # Calculate performance metrics from prediction logs
            performance = self.server.get_model_performance(model_id, actual_version)
            
            # Extract the requested metric
            if metric_name not in performance and f"{metric_name}_macro" in performance:
                # Try macro-averaged version for classification metrics
                metric_name = f"{metric_name}_macro"
                
            if metric_name not in performance:
                raise ValueError(f"Metric {metric_name} not available for model {model_id}")
                
            metric_value = performance[metric_name]
            
            # Determine status based on threshold
            if metric_name in ["mse", "rmse", "mae"]:
                # For error metrics, lower is better
                status = "alert" if metric_value > -threshold else "normal"
            else:
                # For other metrics, higher is better
                status = "alert" if metric_value < threshold else "normal"
            
            # Build result
            performance_metrics = {
                "model_id": model_id,
                "version": actual_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "threshold": float(threshold),
                "status": status,
                "all_metrics": performance,
            }
            
            # Log the performance metrics
            self._log_metric(
                model_id=model_id,
                version=actual_version,
                metric_type=f"performance_{metric_name}",
                metric_value=float(metric_value),
                threshold=float(threshold),
                status=status,
                metadata=performance_metrics,
                vector=None,  # No vector needed for performance monitoring
            )
            
            return performance_metrics
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            raise
    
    def trigger_retraining(
        self,
        model_id: str,
        version: Optional[str] = None,
        training_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Trigger model retraining when needed.
        
        Args:
            model_id: ID of the model to retrain
            version: Specific version to retrain, or None for latest
            training_data: Optional new training data
            
        Returns:
            Dict[str, Any]: Retraining status
        """
        try:
            # Get model metadata
            model_info = self.registry.get_model(model_id, version)
            actual_version = model_info["version"]
            
            # Check if automatic retraining is enabled
            enable_auto_retraining = os.getenv("ENABLE_AUTOMATIC_RETRAINING", "false").lower() == "true"
            if not enable_auto_retraining:
                return {
                    "model_id": model_id,
                    "version": actual_version,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": "skipped",
                    "message": "Automatic retraining is disabled",
                }
            
            # Get n8n.io webhook URL for retraining
            n8n_base_url = os.getenv("N8N_BASE_URL", "http://localhost:5678/webhook/")
            retrain_webhook = f"{n8n_base_url}retrain-model"
            
            # Prepare request data
            retrain_data = {
                "model_id": model_id,
                "version": actual_version,
                "lancedb_path": self.db_path,
                # Add more parameters as needed for retraining
            }
            
            # Track retraining status
            retraining_status = {
                "model_id": model_id,
                "version": actual_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "requested",
                "message": "Retraining requested via n8n.io webhook",
                "details": retrain_data,
            }
            
            # Log the retraining trigger
            self._log_metric(
                model_id=model_id,
                version=actual_version,
                metric_type="retraining",
                metric_value=0.0,  # No specific value for retraining triggers
                threshold=None,
                status="requested",
                metadata=retraining_status,
                vector=None,  # No vector needed for retraining
            )
            
            # Make the request to n8n.io to trigger retraining
            try:
                response = requests.post(retrain_webhook, json=retrain_data)
                response.raise_for_status()
                
                # Update with response
                retraining_status["status"] = "initiated"
                retraining_status["response"] = response.json()
                
                # Update the log entry
                self._update_metric_log(
                    model_id=model_id,
                    version=actual_version,
                    metric_type="retraining",
                    status="initiated",
                    metadata=retraining_status,
                )
                
            except Exception as req_error:
                # Handle request errors
                retraining_status["status"] = "failed"
                retraining_status["error"] = str(req_error)
                
                # Update the log entry
                self._update_metric_log(
                    model_id=model_id,
                    version=actual_version,
                    metric_type="retraining",
                    status="failed",
                    metadata=retraining_status,
                )
            
            return retraining_status
        except Exception as e:
            logger.error(f"Error triggering retraining: {str(e)}")
            raise
    
    def _log_metric(
        self,
        model_id: str,
        version: str,
        metric_type: str,
        metric_value: float,
        threshold: Optional[float] = None,
        status: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[np.ndarray] = None,
    ) -> None:
        """
        Log a monitoring metric to the monitoring table.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            metric_type: Type of metric (drift, performance, etc.)
            metric_value: Value of the metric
            threshold: Threshold for alerting
            status: Status (normal, alert, etc.)
            metadata: Additional metadata
            vector: Optional vector representation
        """
        try:
            # Generate a unique ID
            import hashlib
            timestamp = datetime.datetime.now().isoformat()
            id_hash = hashlib.md5(f"{model_id}_{version}_{metric_type}_{timestamp}".encode()).hexdigest()
            
            # Convert metadata to string
            metadata_str = json.dumps(metadata) if metadata else None
            
            # Use default vector if not provided
            if vector is None:
                # Use a simple vector based on the metric value and timestamp
                # This allows for time-based queries
                ts = datetime.datetime.now().timestamp()
                vector = np.array([metric_value, ts])
            
            # Create log entry
            log_entry = pd.DataFrame([{
                "id": id_hash,
                "timestamp": timestamp,
                "model_id": model_id,
                "version": version,
                "metric_type": metric_type,
                "metric_value": metric_value,
                "threshold": threshold,
                "status": status,
                "metadata": metadata_str,
                "vector": vector
            }])
            
            # Add to monitoring table
            self.monitor_table.add(log_entry)
            logger.debug(f"Logged {metric_type} metric for model {model_id} version {version}: {metric_value}")
        except Exception as e:
            logger.error(f"Error logging metric: {str(e)}")
            # Don't raise the exception to avoid disrupting the monitoring flow
            # Just log the error and continue
    
    def _update_metric_log(
        self,
        model_id: str,
        version: str,
        metric_type: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update an existing monitoring metric log.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            metric_type: Type of metric (drift, performance, etc.)
            status: New status
            metadata: Updated metadata
        """
        try:
            # This is a simplified implementation since LanceDB doesn't support
            # direct updates to records. In a real implementation, you might want
            # to use a different approach.
            
            # Get the latest log entry for this model/version/metric
            logs_df = self.monitor_table.to_pandas()
            
            # Filter by model_id, version, and metric_type
            filtered_logs = logs_df[
                (logs_df["model_id"] == model_id) &
                (logs_df["version"] == version) &
                (logs_df["metric_type"] == metric_type)
            ]
            
            if filtered_logs.empty:
                logger.warning(f"No log entry found for {model_id} {version} {metric_type}")
                return
            
            # Get the latest log entry
            latest_log = filtered_logs.sort_values("timestamp", ascending=False).iloc[0]
            
            # Create a new log entry with updated status and metadata
            self._log_metric(
                model_id=model_id,
                version=version,
                metric_type=metric_type,
                metric_value=latest_log["metric_value"],
                threshold=latest_log["threshold"],
                status=status,
                metadata=metadata,
                vector=latest_log["vector"],
            )
            
            logger.debug(f"Updated {metric_type} metric for model {model_id} version {version} to status {status}")
        except Exception as e:
            logger.error(f"Error updating metric log: {str(e)}")
            # Don't raise the exception to avoid disrupting the monitoring flow
            # Just log the error and continue
    
    def get_monitoring_logs(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        metric_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get monitoring logs for a model.
        
        Args:
            model_id: ID of the model to get logs for, or None for all models
            version: Specific version to get logs for, or None for all versions
            metric_type: Type of metric to get logs for, or None for all types
            status: Status to filter by, or None for all statuses
            limit: Maximum number of logs to return
            
        Returns:
            List[Dict[str, Any]]: Monitoring logs
        """
        try:
            # Query the monitoring table
            logs_df = self.monitor_table.to_pandas()
            
            if logs_df.empty:
                return []
            
            # Apply filters
            if model_id:
                logs_df = logs_df[logs_df["model_id"] == model_id]
            if version:
                logs_df = logs_df[logs_df["version"] == version]
            if metric_type:
                logs_df = logs_df[logs_df["metric_type"] == metric_type]
            if status:
                logs_df = logs_df[logs_df["status"] == status]
            
            # Sort by timestamp (descending)
            logs_df = logs_df.sort_values("timestamp", ascending=False)
            
            # Limit the number of logs
            logs_df = logs_df.head(limit)
            
            # Build result
            result = []
            for _, log in logs_df.iterrows():
                # Parse metadata
                metadata = json.loads(log["metadata"]) if log["metadata"] else None
                
                # Build result
                log_info = {
                    "id": log["id"],
                    "timestamp": log["timestamp"],
                    "model_id": log["model_id"],
                    "version": log["version"],
                    "metric_type": log["metric_type"],
                    "metric_value": float(log["metric_value"]),
                    "threshold": float(log["threshold"]) if log["threshold"] else None,
                    "status": log["status"],
                    "metadata": metadata,
                }
                result.append(log_info)
            
            return result
        except Exception as e:
            logger.error(f"Error getting monitoring logs: {str(e)}")
            raise
