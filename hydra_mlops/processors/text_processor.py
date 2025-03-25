"""
Text Feature Processor for Hydra MLOps Framework.

This module provides a processor for text data, transforming raw text
into vector embeddings and other NLP features.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

from . import FeatureProcessor

logger = logging.getLogger(__name__)

class TextProcessor(FeatureProcessor):
    """
    Processor for text data.
    
    Generates vector embeddings and other NLP features from text columns.
    """
    
    def __init__(
        self,
        text_columns: Optional[Union[str, List[str]]] = None,
        embedding_model_name: Optional[str] = None,
        add_text_length: bool = True,
        add_word_count: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[List[str]] = None,
    ):
        """
        Initialize the text processor.
        
        Args:
            text_columns: Column(s) containing text to process
            embedding_model_name: Name of the embedding model to use
            add_text_length: Whether to add text length as a feature
            add_word_count: Whether to add word count as a feature
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            stopwords: List of stopwords to remove (if None, uses common English stopwords)
        """
        self.text_columns = text_columns
        self.embedding_model_name = embedding_model_name or os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.add_text_length = add_text_length
        self.add_word_count = add_word_count
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords
        
        # Lazily loaded resources
        self._embedding_model = None
        self._stopwords_set = None
        
    def fit(self, data: pd.DataFrame) -> 'TextProcessor':
        """
        Fit the processor on the input data.
        
        For text processor, this mainly involves identifying text columns
        if not explicitly specified.
        
        Args:
            data: Input data to fit on
            
        Returns:
            TextProcessor: The fitted processor
        """
        # If no text columns specified, try to infer
        if self.text_columns is None:
            # Try to find text columns (object dtype with strings)
            text_cols = data.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                # Use all text columns by default
                self.text_columns = text_cols
                logger.info(f"No text_columns specified, using inferred columns: {self.text_columns}")
            else:
                logger.warning("No text columns found for processing")
        
        # Convert to list if single string
        if isinstance(self.text_columns, str):
            self.text_columns = [self.text_columns]
            
        # Initialize stopwords if needed
        if self.remove_stopwords and self._stopwords_set is None:
            if self.stopwords:
                self._stopwords_set = set(self.stopwords)
            else:
                # Use a simple set of common English stopwords
                self._stopwords_set = {
                    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                    "when", "where", "how", "who", "which", "this", "that", "these", "those",
                    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
                    "may", "might", "must", "to", "for", "in", "on", "at", "by", "with",
                    "about", "against", "between", "into", "through", "during", "before",
                    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under"
                }
        
        return self
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the text data and generate features.
        
        Args:
            data: Input data to process
            
        Returns:
            pd.DataFrame: Processed data with text features
        """
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Check that text columns exist
        if not self.text_columns:
            logger.warning("No text columns specified or inferred. Returning data unchanged.")
            return processed_data
        
        for column in self.text_columns:
            if column not in processed_data.columns:
                logger.warning(f"Column '{column}' not found in data. Skipping.")
                continue
            
            # Apply text preprocessing
            col_data = processed_data[column].fillna("").astype(str)
            
            if self.lowercase:
                col_data = col_data.str.lower()
            
            if self.remove_punctuation:
                col_data = col_data.apply(lambda x: re.sub(r'[^\w\s]', '', x))
            
            if self.remove_stopwords:
                col_data = col_data.apply(
                    lambda x: ' '.join([word for word in x.split() if word not in self._stopwords_set])
                )
            
            # Store preprocessed text
            preprocessed_col = f"{column}_preprocessed"
            processed_data[preprocessed_col] = col_data
            
            # Add text length feature
            if self.add_text_length:
                processed_data[f"{column}_length"] = col_data.str.len()
            
            # Add word count feature
            if self.add_word_count:
                processed_data[f"{column}_word_count"] = col_data.str.split().str.len()
            
            # Generate embeddings
            if self._embedding_model is None:
                logger.info(f"Initializing embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            
            logger.info(f"Generating embeddings for column '{column}'")
            embeddings = self._embedding_model.encode(col_data.tolist())
            
            # Add embeddings as a vector column
            vector_col = f"{column}_vector"
            processed_data[vector_col] = list(embeddings)
            
            # If only one text column, also add a standard "vector" column
            if len(self.text_columns) == 1:
                processed_data["vector"] = processed_data[vector_col]
        
        return processed_data
