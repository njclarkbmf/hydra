"""
Monitoring components for the LanceDB MLOps Framework.

This module provides classes for monitoring model performance, detecting drift,
and triggering alerts when models need attention.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import os
import logging
import numpy as np
import pandas as pd
import datetime
import json

logger = logging.getLogger(__name__)

class ModelMonitor(ABC):
    """
    Abstract base class for model monitors.
    
    Model monitors are responsible for tracking model performance,
    detecting drift, and triggering alerts when needed.
    """
    
    @abstractmethod
    def detect_drift(
        self,
        model_id: str,
        reference_data: Union[pd.DataFrame, np.ndarray],
        current_data: Union[pd.DataFrame, np.ndarray],
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            model_id: ID of the model to monitor
            reference_data: Reference data for comparison
            current_data: Current data to check for drift
            version: Specific version to monitor, or None for latest
            
        Returns:
            Dict[str, Any]: Drift metrics
        """
        pass
    
    @abstractmethod
    def monitor_performance(
        self,
        model_id: str,
        threshold: Optional[float] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Monitor model performance based on logged predictions.
        
        Args:
            model_id: ID of the model to monitor
            threshold: Performance threshold for alerting
            version: Specific version to monitor, or None for latest
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        pass
    
    @abstractmethod
    def trigger_retraining(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger model retraining when needed.
        
        Args:
            model_id: ID of the model to retrain
            version: Specific version to retrain, or None for latest
            
        Returns:
            Dict[str, Any]: Retraining status
        """
        pass

# Import specific monitor implementations
from .lancedb_monitor import LanceDBModelMonitor

# Default monitor implementation
default_monitor = LanceDBModelMonitor
