"""
Feature processors for the Hydra MLOps Framework.

This module provides base classes and implementations for feature processors
that transform raw data into features suitable for machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureProcessor(ABC):
    """
    Abstract base class for all feature processors.
    
    Feature processors are responsible for transforming raw data into
    features that can be used by machine learning models.
    """
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input data and generate features.
        
        Args:
            data: Input data to process
            
        Returns:
            pd.DataFrame: Processed data with features
        """
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'FeatureProcessor':
        """
        Fit the processor on the input data.
        
        Args:
            data: Input data to fit on
            
        Returns:
            FeatureProcessor: The fitted processor
        """
        pass
    
    def fit_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the processor on the input data and process it.
        
        Args:
            data: Input data to fit and process
            
        Returns:
            pd.DataFrame: Processed data with features
        """
        self.fit(data)
        return self.process(data)

# Import specific processor implementations
from .text_processor import TextProcessor

# Register available processors
AVAILABLE_PROCESSORS = {
    "text": TextProcessor,
}

def get_processor(processor_type: str, **params):
    """
    Factory function to get a processor instance by type.
    
    Args:
        processor_type: Type of processor to create.
        **params: Parameters to pass to the processor constructor.
        
    Returns:
        FeatureProcessor: An instance of the requested processor.
        
    Raises:
        ValueError: If the processor type is not recognized.
    """
    if processor_type not in AVAILABLE_PROCESSORS:
        raise ValueError(f"Unknown processor type: {processor_type}. Available types: {list(AVAILABLE_PROCESSORS.keys())}")
    
    processor_class = AVAILABLE_PROCESSORS[processor_type]
    return processor_class(**params)
