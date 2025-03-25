"""
CSV Data Connector for LanceDB MLOps Framework.

This module provides a connector for CSV files, allowing data to be
loaded from local or remote CSV files and transformed into vectors.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from . import DataConnector

logger = logging.getLogger(__name__)

class CSVConnector(DataConnector):
    """
    Connector for CSV file data sources.
    
    Supports loading data from local files or URLs and transforming
    specified text columns into vector embeddings.
    """
    
    def __init__(
        self,
        file_path: str,
        text_columns: Optional[Union[str, List[str]]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        embedding_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the CSV connector.
        
        Args:
            file_path: Path to the CSV file (local path or URL)
            text_columns: Column(s) to use for generating vector embeddings
            delimiter: CSV delimiter character
            encoding: File encoding
            embedding_model_name: Name of the sentence transformer model to use
            **kwargs: Additional pandas read_csv parameters
        """
        self.file_path = file_path
        self.text_columns = text_columns
        self.delimiter = delimiter
        self.encoding = encoding
        self.embedding_model_name = embedding_model_name or os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.additional_params = kwargs
        self._embedding_model = None
        
    def connect(self) -> bool:
        """
        Verify that the CSV file is accessible.
        
        Returns:
            bool: True if the file is accessible, False otherwise.
        """
        try:
            # For local files, check if file exists
            if os.path.exists(self.file_path):
                return True
                
            # For URLs, we'll check during fetch_data
            return True
        except Exception as e:
            logger.error(f"Error connecting to CSV file {self.file_path}: {str(e)}")
            return False
            
    def fetch_data(self, query_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load data from the CSV file.
        
        Args:
            query_params: Optional parameters for filtering (e.g., nrows, usecols)
                
        Returns:
            pd.DataFrame: The loaded CSV data
        """
        try:
            # Combine additional params with query params
            params = self.additional_params.copy()
            if query_params:
                params.update(query_params)
                
            # Read the CSV file
            data = pd.read_csv(
                self.file_path,
                delimiter=self.delimiter,
                encoding=self.encoding,
                **params
            )
            
            logger.info(f"Successfully loaded CSV file with {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error fetching data from CSV file: {str(e)}")
            raise
            
    def transform_to_vectors(self, data: pd.DataFrame, embedding_model=None) -> pd.DataFrame:
        """
        Transform text columns into vector embeddings.
        
        Args:
            data: The data to transform
            embedding_model: Optional model to use (if None, will use the configured model)
                
        Returns:
            pd.DataFrame: DataFrame with vector column(s) added
        """
        try:
            # If no text columns specified, try to infer
            columns_to_embed = self.text_columns
            if columns_to_embed is None:
                # Try to find text columns (object dtype with strings)
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    # Use the first text column by default
                    columns_to_embed = [text_cols[0]]
                    logger.info(f"No text_columns specified, using inferred column: {columns_to_embed}")
                else:
                    logger.warning("No text columns found for embedding generation")
                    return data
            
            # Convert to list if single string
            if isinstance(columns_to_embed, str):
                columns_to_embed = [columns_to_embed]
                
            # Validate columns exist
            for col in columns_to_embed:
                if col not in data.columns:
                    logger.warning(f"Column '{col}' not found in data, skipping embedding generation")
                    return data
            
            # Initialize embedding model if not provided
            if embedding_model is None:
                if self._embedding_model is None:
                    logger.info(f"Initializing embedding model: {self.embedding_model_name}")
                    self._embedding_model = SentenceTransformer(self.embedding_model_name)
                embedding_model = self._embedding_model
            
            # Generate embeddings for each text column
            for col in columns_to_embed:
                vector_col = f"{col}_vector"
                
                # Handle NaN values by replacing with empty string
                text_series = data[col].fillna("").astype(str)
                
                logger.info(f"Generating embeddings for column '{col}'")
                data[vector_col] = list(embedding_model.encode(text_series.tolist()))
                
                # If only one column to embed, also add a standard "vector" column
                if len(columns_to_embed) == 1:
                    data["vector"] = data[vector_col]
            
            return data
        except Exception as e:
            logger.error(f"Error transforming data to vectors: {str(e)}")
            raise
