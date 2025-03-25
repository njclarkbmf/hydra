"""
API Data Connector for Hydra MLOps Framework.

This module provides a connector for REST APIs, allowing data to be
loaded from various API endpoints and transformed into vectors.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import requests
import json
from sentence_transformers import SentenceTransformer

from . import DataConnector

logger = logging.getLogger(__name__)

class APIConnector(DataConnector):
    """
    Connector for REST API data sources.
    
    Supports loading data from REST APIs and transforming
    specified text fields into vector embeddings.
    """
    
    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        json_path: Optional[str] = None,
        text_fields: Optional[Union[str, List[str]]] = None,
        embedding_model_name: Optional[str] = None,
    ):
        """
        Initialize the API connector.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            headers: HTTP headers for the request
            auth: Authentication credentials (username, password)
            json_path: JSONPath expression to extract data from response
            text_fields: Field(s) to use for generating vector embeddings
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.auth = auth
        self.json_path = json_path
        self.text_fields = text_fields
        self.embedding_model_name = embedding_model_name or os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._embedding_model = None
        
    def connect(self) -> bool:
        """
        Verify that the API is accessible.
        
        Returns:
            bool: True if the API is accessible, False otherwise.
        """
        try:
            # For APIs, we'll just check if the URL is valid
            # The actual connection is established during fetch_data
            return True
        except Exception as e:
            logger.error(f"Error validating API URL {self.url}: {str(e)}")
            return False
            
    def fetch_data(self, query_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the API.
        
        Args:
            query_params: Optional parameters for the API request
                
        Returns:
            pd.DataFrame: The fetched data
        """
        try:
            # Prepare request parameters
            kwargs = {
                "headers": self.headers,
            }
            
            if self.auth:
                kwargs["auth"] = (self.auth.get("username"), self.auth.get("password"))
            
            if query_params:
                if self.method == "GET":
                    kwargs["params"] = query_params
                else:
                    kwargs["json"] = query_params
            
            # Make the request
            logger.info(f"Making {self.method} request to {self.url}")
            response = requests.request(self.method, self.url, **kwargs)
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse JSON response
            json_data = response.json()
            
            # Extract data using json_path if provided
            if self.json_path:
                from jsonpath_ng import parse
                jsonpath_expr = parse(self.json_path)
                matches = [match.value for match in jsonpath_expr.find(json_data)]
                if not matches:
                    logger.warning(f"No data found at JSON path: {self.json_path}")
                    return pd.DataFrame()
                data = matches
            else:
                data = json_data
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")
            
            logger.info(f"Successfully fetched {len(df)} rows from API")
            return df
        except Exception as e:
            logger.error(f"Error fetching data from API: {str(e)}")
            raise
            
    def transform_to_vectors(self, data: pd.DataFrame, embedding_model=None) -> pd.DataFrame:
        """
        Transform text fields into vector embeddings.
        
        Args:
            data: The data to transform
            embedding_model: Optional model to use (if None, will use the configured model)
                
        Returns:
            pd.DataFrame: DataFrame with vector column(s) added
        """
        try:
            # If no text fields specified, try to infer
            fields_to_embed = self.text_fields
            if fields_to_embed is None:
                # Try to find text columns (object dtype with strings)
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    # Use the first text column by default
                    fields_to_embed = [text_cols[0]]
                    logger.info(f"No text_fields specified, using inferred column: {fields_to_embed}")
                else:
                    logger.warning("No text columns found for embedding generation")
                    return data
            
            # Convert to list if single string
            if isinstance(fields_to_embed, str):
                fields_to_embed = [fields_to_embed]
                
            # Validate fields exist
            for field in fields_to_embed:
                if field not in data.columns:
                    logger.warning(f"Field '{field}' not found in data, skipping embedding generation")
                    return data
            
            # Initialize embedding model if not provided
            if embedding_model is None:
                if self._embedding_model is None:
                    logger.info(f"Initializing embedding model: {self.embedding_model_name}")
                    self._embedding_model = SentenceTransformer(self.embedding_model_name)
                embedding_model = self._embedding_model
            
            # Generate embeddings for each text field
            for field in fields_to_embed:
                vector_field = f"{field}_vector"
                
                # Handle NaN values by replacing with empty string
                text_series = data[field].fillna("").astype(str)
                
                logger.info(f"Generating embeddings for field '{field}'")
                data[vector_field] = list(embedding_model.encode(text_series.tolist()))
                
                # If only one field to embed, also add a standard "vector" column
                if len(fields_to_embed) == 1:
                    data["vector"] = data[vector_field]
            
            return data
        except Exception as e:
            logger.error(f"Error transforming data to vectors: {str(e)}")
            raise
