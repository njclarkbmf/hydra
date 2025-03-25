"""
Data connectors for the LanceDB MLOps Framework.

This module provides base classes and implementations for data connectors
that can be used to ingest data from various sources into LanceDB.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class DataConnector(ABC):
    """
    Abstract base class for all data connectors.
    
    Data connectors are responsible for connecting to data sources,
    fetching data, and transforming it into a format suitable for
    LanceDB (including vector generation).
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish a connection to the data source.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def fetch_data(self, query_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the source based on optional query parameters.
        
        Args:
            query_params: Optional parameters to filter or customize the data fetch.
            
        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.
        """
        pass
    
    @abstractmethod
    def transform_to_vectors(self, data: pd.DataFrame, embedding_model=None) -> pd.DataFrame:
        """
        Transform the fetched data into vector embeddings.
        
        Args:
            data: The data to transform.
            embedding_model: Optional model to use for generating embeddings.
            
        Returns:
            pd.DataFrame: DataFrame with added vector columns.
        """
        pass
    
    def process(self, query_params: Optional[Dict[str, Any]] = None, embedding_model=None) -> pd.DataFrame:
        """
        Execute the full data processing pipeline: connect, fetch, and transform.
        
        Args:
            query_params: Optional parameters for data fetching.
            embedding_model: Optional model for vector generation.
            
        Returns:
            pd.DataFrame: Processed data with vector embeddings.
        """
        if not self.connect():
            raise ConnectionError(f"Failed to connect to data source")
        
        data = self.fetch_data(query_params)
        return self.transform_to_vectors(data, embedding_model)

# Import specific connector implementations
from .csv_connector import CSVConnector
from .db_connector import DatabaseConnector
from .api_connector import APIConnector

# Register available connectors
AVAILABLE_CONNECTORS = {
    "csv": CSVConnector,
    "database": DatabaseConnector,
    "api": APIConnector,
}

def get_connector(connector_type: str, **params):
    """
    Factory function to get a connector instance by type.
    
    Args:
        connector_type: Type of connector to create.
        **params: Parameters to pass to the connector constructor.
        
    Returns:
        DataConnector: An instance of the requested connector.
        
    Raises:
        ValueError: If the connector type is not recognized.
    """
    if connector_type not in AVAILABLE_CONNECTORS:
        raise ValueError(f"Unknown connector type: {connector_type}. Available types: {list(AVAILABLE_CONNECTORS.keys())}")
    
    connector_class = AVAILABLE_CONNECTORS[connector_type]
    return connector_class(**params)
