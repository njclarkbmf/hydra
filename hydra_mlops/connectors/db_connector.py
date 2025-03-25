"""
Database Connector for LanceDB MLOps Framework.

This module provides a connector for SQL databases, allowing data to be
loaded from various database engines and transformed into vectors.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

from . import DataConnector

logger = logging.getLogger(__name__)

class DatabaseConnector(DataConnector):
    """
    Connector for SQL database data sources.
    
    Supports loading data from various database engines using SQLAlchemy
    and transforming specified text columns into vector embeddings.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        text_columns: Optional[Union[str, List[str]]] = None,
        embedding_model_name: Optional[str] = None,
    ):
        """
        Initialize the database connector.
        
        Args:
            connection_string: SQLAlchemy connection string
            host: Database host (if not using connection_string)
            port: Database port (if not using connection_string)
            user: Database user (if not using connection_string)
            password: Database password (if not using connection_string)
            database: Database name (if not using connection_string)
            table: Table to query (if not using custom query)
            query: Custom SQL query to execute
            text_columns: Column(s) to use for generating vector embeddings
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.connection_string = connection_string
        self.host = host or os.getenv("DB_HOST")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")
        self.database = database or os.getenv("DB_NAME")
        self.table = table
        self.query = query
        self.text_columns = text_columns
        self.embedding_model_name = embedding_model_name or os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._engine = None
        self._embedding_model = None
        
    def connect(self) -> bool:
        """
        Establish a connection to the database.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            # Use provided connection string or build one from components
            if self.connection_string:
                conn_string = self.connection_string
            else:
                if not all([self.host, self.user, self.database]):
                    raise ValueError("Database connection requires host, user, and database")
                
                # Build connection string based on parameters
                # This is a simplified version, actual implementation might need to support
                # different database engines (postgres, mysql, etc.)
                conn_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            
            # Create SQLAlchemy engine
            self._engine = create_engine(conn_string)
            
            # Test connection
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            logger.info(f"Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
            
    def fetch_data(self, query_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the database.
        
        Args:
            query_params: Optional parameters for filtering or customizing the query
                
        Returns:
            pd.DataFrame: The fetched data
        """
        try:
            if self._engine is None:
                raise ConnectionError("Database connection not established. Call connect() first.")
            
            # Determine the query to execute
            if self.query:
                # Use custom query
                sql_query = self.query
            elif self.table:
                # Build query from table
                where_clause = ""
                if query_params and "where" in query_params:
                    where_clause = f"WHERE {query_params['where']}"
                
                limit_clause = ""
                if query_params and "limit" in query_params:
                    limit_clause = f"LIMIT {query_params['limit']}"
                
                columns = "*"
                if query_params and "columns" in query_params:
                    if isinstance(query_params["columns"], list):
                        columns = ", ".join(query_params["columns"])
                    else:
                        columns = query_params["columns"]
                
                sql_query = f"SELECT {columns} FROM {self.table} {where_clause} {limit_clause}"
            else:
                raise ValueError("Either table or query must be provided")
            
            # Execute the query
            logger.info(f"Executing SQL query: {sql_query}")
            data = pd.read_sql(sql_query, self._engine)
            
            logger.info(f"Successfully fetched {len(data)} rows from database")
            return data
        except Exception as e:
            logger.error(f"Error fetching data from database: {str(e)}")
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
