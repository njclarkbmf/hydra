"""
Classifier Model Trainer for LanceDB MLOps Framework.

This module provides a trainer for classification models, supporting
various sklearn classifiers with standardized interfaces for training,
evaluation, and model persistence.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from . import BaseModelTrainer

logger = logging.getLogger(__name__)

class ClassifierTrainer(BaseModelTrainer):
    """
    Trainer for classification models.
    
    Supports various sklearn classifiers with a standardized interface.
    """
    
    def __init__(
        self,
        classifier_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the classifier trainer.
        
        Args:
            classifier_type: Type of classifier to use (random_forest, logistic_regression, svm, gradient_boosting)
            model_params: Parameters for the classifier
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__(classifier_type, model_params)
        self.classifier_type = classifier_type
        self.test_size = test_size
        self.random_state = random_state
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the classifier based on the specified type.
        """
        model_params = self.model_params or {}
        
        # Ensure random_state is set for reproducibility
        if "random_state" not in model_params:
            model_params["random_state"] = self.random_state
            
        if self.classifier_type == "random_forest":
            self.model = RandomForestClassifier(**model_params)
        elif self.classifier_type == "logistic_regression":
            self.model = LogisticRegression(**model_params)
        elif self.classifier_type == "svm":
            self.model = SVC(**model_params)
        elif self.classifier_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        
        logger.info(f"Initialized {self.classifier_type} classifier with parameters: {model_params}")
        
    def train(
        self,
        features: Union[np.ndarray, List],
        labels: Union[np.ndarray, List],
        validation_split: bool = True
    ) -> Any:
        """
        Train the classifier on the provided features and labels.
        
        Args:
            features: Feature vectors for training
            labels: Target labels for training
            validation_split: Whether to split data for validation
            
        Returns:
            Any: The trained model
        """
        try:
            # Convert inputs to numpy arrays if they aren't already
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
                
            # Split data if validation is requested
            if validation_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels, test_size=self.test_size, random_state=self.random_state
                )
                
                logger.info(f"Training {self.classifier_type} on {len(X_train)} samples, validating on {len(X_val)} samples")
                
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_metrics = self.evaluate(X_val, y_val)
                logger.info(f"Validation metrics: {val_metrics}")
            else:
                logger.info(f"Training {self.classifier_type} on all {len(features)} samples")
                
                # Train on all data
                self.model.fit(features, labels)
            
            return self.model
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise
            
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained classifier on the provided features and labels.
        
        Args:
            features: Feature vectors for evaluation
            labels: Target labels for evaluation
            
        Returns:
            Dict[str, float]: Metrics for the evaluated classifier
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            # Make predictions
            predictions = self.model.predict(features)
            
            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(labels, predictions)),
                "precision_macro": float(precision_score(labels, predictions, average='macro', zero_division=0)),
                "recall_macro": float(recall_score(labels, predictions, average='macro', zero_division=0)),
                "f1_macro": float(f1_score(labels, predictions, average='macro', zero_division=0)),
            }
            
            # Add weighted metrics
            metrics.update({
                "precision_weighted": float(precision_score(labels, predictions, average='weighted', zero_division=0)),
                "recall_weighted": float(recall_score(labels, predictions, average='weighted', zero_division=0)),
                "f1_weighted": float(f1_score(labels, predictions, average='weighted', zero_division=0)),
            })
            
            # If binary classification, add binary metrics
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                metrics.update({
                    "precision_binary": float(precision_score(labels, predictions, average='binary', zero_division=0)),
                    "recall_binary": float(recall_score(labels, predictions, average='binary', zero_division=0)),
                    "f1_binary": float(f1_score(labels, predictions, average='binary', zero_division=0)),
                })
                
            # Get detailed classification report
            report = classification_report(labels, predictions, output_dict=True)
            
            # Store per-class metrics
            for class_label, class_metrics in report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    metrics[f"class_{class_label}_precision"] = float(class_metrics['precision'])
                    metrics[f"class_{class_label}_recall"] = float(class_metrics['recall'])
                    metrics[f"class_{class_label}_f1"] = float(class_metrics['f1-score'])
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating classifier: {str(e)}")
            raise
            
    def save(self, path: str) -> None:
        """
        Save the trained classifier to disk.
        
        Args:
            path: Path to save the model to
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            if path.endswith('.joblib'):
                joblib.dump(self.model, path)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                    
            logger.info(f"Saved classifier to {path}")
        except Exception as e:
            logger.error(f"Error saving classifier: {str(e)}")
            raise
            
    def load(self, path: str) -> Any:
        """
        Load a trained classifier from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Any: The loaded model
        """
        try:
            # Load the model
            if path.endswith('.joblib'):
                self.model = joblib.load(path)
            else:
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                    
            logger.info(f"Loaded classifier from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading classifier: {str(e)}")
            raise
            
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained classifier.
        
        Args:
            features: Feature vectors for prediction
            
        Returns:
            np.ndarray: Predicted labels
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            return self.model.predict(features)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            features: Feature vectors for prediction
            
        Returns:
            np.ndarray: Probability estimates
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
                
            # Check if the model supports predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(features)
            else:
                logger.warning(f"Model {self.classifier_type} does not support probability estimates")
                # Return dummy probabilities based on predictions
                predictions = self.model.predict(features)
                n_classes = len(np.unique(predictions))
                proba = np.zeros((len(features), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, pred] = 1.0
                return proba
        except Exception as e:
            logger.error(f"Error generating probability estimates: {str(e)}")
            raise
