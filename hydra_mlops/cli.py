#!/usr/bin/env python
"""
Command-line interface for the LanceDB MLOps Framework.

This module provides a command-line interface for interacting with the
framework, including data ingestion, model training, and inference.
"""

import os
import sys
import argparse
import logging
import json
import datetime
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lancedb-mlops-cli")

# Default API URL
DEFAULT_API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LanceDB MLOps Framework CLI")
    
    # Add version argument
    parser.add_argument("--version", action="version", version="LanceDB MLOps Framework v1.0.0")
    
    # Add common arguments
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Data ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data")
    ingest_parser.add_argument("--connector-type", required=True, help="Connector type (csv, database, api)")
    ingest_parser.add_argument("--table-name", required=True, help="Table name in LanceDB")
    ingest_parser.add_argument("--params", required=True, help="Connector parameters (JSON)")
    ingest_parser.add_argument("--query-params", help="Query parameters (JSON)")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--table-name", required=True, help="Table name in LanceDB")
    train_parser.add_argument("--feature-column", required=True, help="Feature column name")
    train_parser.add_argument("--label-column", required=True, help="Label column name")
    train_parser.add_argument("--model-type", default="classifier", help="Model type (classifier, regressor)")
    train_parser.add_argument("--model-id", required=True, help="Model ID")
    train_parser.add_argument("--model-params", help="Model parameters (JSON)")
    
    # Model inference command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model-id", required=True, help="Model ID")
    predict_parser.add_argument("--features", required=True, help="Features for prediction (JSON)")
    predict_parser.add_argument("--version", help="Model version")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List all models")
    
    # Model monitoring command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor model performance")
    monitor_parser.add_argument("--model-id", required=True, help="Model ID")
    monitor_parser.add_argument("--metric", default="accuracy", help="Metric to monitor")
    monitor_parser.add_argument("--threshold", type=float, help="Alert threshold")
    
    # Drift detection command
    drift_parser = subparsers.add_parser("detect-drift", help="Detect data drift")
    drift_parser.add_argument("--model-id", required=True, help="Model ID")
    drift_parser.add_argument("--reference-table", required=True, help="Reference data table")
    drift_parser.add_argument("--current-table", required=True, help="Current data table")
    
    # A/B testing command
    ab_parser = subparsers.add_parser("create-ab-test", help="Create A/B test")
    ab_parser.add_argument("--model-a", required=True, help="Model A ID")
    ab_parser.add_argument("--model-b", required=True, help="Model B ID")
    ab_parser.add_argument("--test-name", required=True, help="Test name")
    ab_parser.add_argument("--traffic-split", type=float, default=0.5, help="Traffic split (0.0-1.0)")
    
    return parser.parse_args()

def make_api_request(method, endpoint, data=None, headers=None):
    """Make a request to the API."""
    url = f"{args.api_url}/api/{endpoint}"
    
    # Add API key if provided
    request_headers = headers or {}
    if args.api_key:
        request_headers["X-API-Key"] = args.api_key
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=request_headers)
        else:
            response = requests.post(url, json=data, headers=request_headers)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)

def ingest_data(args):
    """Ingest data using the specified connector."""
    # Parse JSON parameters
    try:
        connector_params = json.loads(args.params)
        query_params = json.loads(args.query_params) if args.query_params else None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON parameters: {str(e)}")
        sys.exit(1)
    
    # Prepare request data
    data = {
        "connector_type": args.connector_type,
        "connector_params": connector_params,
        "query_params": query_params,
        "table_name": args.table_name,
        "lancedb_path": os.getenv("LANCEDB_PATH")
    }
    
    # Make API request
    logger.info(f"Ingesting data using {args.connector_type} connector to table '{args.table_name}'")
    result = make_api_request("POST", "data/ingest", data)
    
    # Print result
    if result.get("status") == "success":
        print(f"Successfully ingested {result.get('rows_processed', 0)} rows to table '{args.table_name}'")
    else:
        print(f"Data ingestion failed: {result.get('message', 'Unknown error')}")

def train_model(args):
    """Train a model on the specified data."""
    # Parse JSON parameters
    try:
        model_params = json.loads(args.model_params) if args.model_params else {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON parameters: {str(e)}")
        sys.exit(1)
    
    # Prepare request data
    data = {
        "table_name": args.table_name,
        "feature_column": args.feature_column,
        "label_column": args.label_column,
        "model_type": args.model_type,
        "model_id": args.model_id,
        "model_params": model_params,
        "lancedb_path": os.getenv("LANCEDB_PATH"),
        "models_dir": os.getenv("MODELS_DIR")
    }
    
    # Make API request
    logger.info(f"Training {args.model_type} model '{args.model_id}' on table '{args.table_name}'")
    result = make_api_request("POST", "models/train", data)
    
    # Print result
    if result.get("status") == "success":
        print(f"Successfully trained model '{args.model_id}' version '{result.get('version')}'")
        print("\nMetrics:")
        for metric, value in result.get("metrics", {}).items():
            print(f"- {metric}: {value}")
    else:
        print(f"Model training failed: {result.get('message', 'Unknown error')}")

def predict(args):
    """Make predictions using the specified model."""
    # Parse JSON parameters
    try:
        features = json.loads(args.features)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON parameters: {str(e)}")
        sys.exit(1)
    
    # Prepare request data
    data = {
        "model_id": args.model_id,
        "features": features,
        "lancedb_path": os.getenv("LANCEDB_PATH")
    }
    
    if args.version:
        data["version"] = args.version
    
    # Make API request
    logger.info(f"Making predictions with model '{args.model_id}'")
    result = make_api_request("POST", "models/predict", data)
    
    # Print result
    if result.get("status") == "success":
        print(f"Predictions from model '{args.model_id}' version '{result.get('version')}':")
        predictions = result.get("predictions", [])
        
        # Print predictions in a readable format
        if len(predictions) == 1:
            print(f"Prediction: {predictions[0]}")
        else:
            for i, pred in enumerate(predictions):
                print(f"Sample {i+1}: {pred}")
        
        # Print probabilities if available
        if result.get("probabilities"):
            print("\nProbabilities:")
            probs = result.get("probabilities")
            if len(probs) == 1:
                print(f"Probabilities: {probs[0]}")
            else:
                for i, prob in enumerate(probs):
                    print(f"Sample {i+1}: {prob}")
    else:
        print(f"Prediction failed: {result.get('message', 'Unknown error')}")

def list_models(args):
    """List all models in the registry."""
    # Make API request
    logger.info("Listing all models")
    result = make_api_request("GET", f"models?lancedb_path={os.getenv('LANCEDB_PATH')}")
    
    # Print result
    models = result.get("models", [])
    if models:
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"\nModel ID: {model.get('model_id')}")
            print(f"  Version: {model.get('version')}")
            print(f"  Created: {model.get('created_at')}")
            print(f"  Type: {model.get('model_type')}")
            
            # Print metrics if available
            metrics = model.get('metrics', {})
            if metrics:
                print("  Metrics:")
                for metric, value in metrics.items():
                    print(f"    - {metric}: {value}")
    else:
        print("No models found in the registry")

def monitor_model(args):
    """Monitor model performance."""
    # Check if monitoring is enabled
    if os.getenv("ENABLE_MODEL_MONITORING", "true").lower() != "true":
        print("Model monitoring is disabled. Enable it in the configuration to use this feature.")
        return
    
    # Prepare request data
    data = {
        "model_id": args.model_id,
        "metric_name": args.metric,
        "lancedb_path": os.getenv("LANCEDB_PATH")
    }
    
    if args.threshold:
        data["threshold"] = args.threshold
    
    # Make API request
    logger.info(f"Monitoring model '{args.model_id}' using metric '{args.metric}'")
    
    # This endpoint is only available if monitoring is enabled
    try:
        result = make_api_request("POST", "monitoring/performance", data)
        
        # Print result
        print(f"Performance monitoring for model '{args.model_id}':")
        print(f"  Metric: {result.get('metric_name')}")
        print(f"  Value: {result.get('metric_value')}")
        print(f"  Threshold: {result.get('threshold')}")
        print(f"  Status: {result.get('status')}")
        
        # Print all metrics if available
        all_metrics = result.get('all_metrics', {})
        if all_metrics:
            print("\nAll Metrics:")
            for metric, value in all_metrics.items():
                print(f"  - {metric}: {value}")
    except Exception as e:
        print(f"Model monitoring failed: {str(e)}")

def detect_drift(args):
    """Detect data drift between reference and current data."""
    # Check if drift detection is enabled
    if os.getenv("ENABLE_DRIFT_DETECTION", "false").lower() != "true":
        print("Drift detection is disabled. Enable it in the configuration to use this feature.")
        return
    
    # Prepare request data
    data = {
        "model_id": args.model_id,
        "reference_table": args.reference_table,
        "current_table": args.current_table,
        "lancedb_path": os.getenv("LANCEDB_PATH")
    }
    
    # Make API request
    logger.info(f"Detecting drift for model '{args.model_id}'")
    
    # This endpoint is only available if drift detection is enabled
    try:
        result = make_api_request("POST", "monitoring/drift", data)
        
        # Print result
        print(f"Drift detection for model '{args.model_id}':")
        print(f"  Cosine Similarity: {result.get('cosine_similarity')}")
        print(f"  KS Statistic: {result.get('ks_statistic')}")
        print(f"  P Value: {result.get('p_value')}")
        print(f"  Drift Detected: {result.get('drift_detected')}")
        print(f"  Threshold: {result.get('drift_threshold')}")
        
        # Print dimension drift if available
        dimension_drift = result.get('dimension_drift', [])
        if dimension_drift:
            print("\nDimension-wise Drift:")
            for dim in dimension_drift:
                print(f"  Dimension {dim.get('dimension')}:")
                print(f"    - KS Statistic: {dim.get('ks_statistic')}")
                print(f"    - P Value: {dim.get('p_value')}")
                print(f"    - Drift Detected: {dim.get('drift_detected')}")
    except Exception as e:
        print(f"Drift detection failed: {str(e)}")

def create_ab_test(args):
    """Create an A/B test between two models."""
    # Check if A/B testing is enabled
    if os.getenv("ENABLE_AB_TESTING", "false").lower() != "true":
        print("A/B testing is disabled. Enable it in the configuration to use this feature.")
        return
    
    # Prepare request data
    data = {
        "model_a_id": args.model_a,
        "model_b_id": args.model_b,
        "test_name": args.test_name,
        "traffic_split": args.traffic_split,
        "lancedb_path": os.getenv("LANCEDB_PATH")
    }
    
    # Make API request
    logger.info(f"Creating A/B test '{args.test_name}' between models '{args.model_a}' and '{args.model_b}'")
    
    # This endpoint is only available if A/B testing is enabled
    try:
        result = make_api_request("POST", "testing/ab-test", data)
        
        # Print result
        print(f"A/B test '{args.test_name}' created:")
        print(f"  Test ID: {result.get('test_id')}")
        print(f"  Model A: {result.get('model_a_id')}")
        print(f"  Model B: {result.get('model_b_id')}")
        print(f"  Traffic Split: {result.get('traffic_split')}")
        print(f"  Status: {result.get('status')}")
    except Exception as e:
        print(f"A/B test creation failed: {str(e)}")

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Execute the requested command
    if args.command == "ingest":
        ingest_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "list-models":
        list_models(args)
    elif args.command == "monitor":
        monitor_model(args)
    elif args.command == "detect-drift":
        detect_drift(args)
    elif args.command == "create-ab-test":
        create_ab_test(args)
    else:
        print("No command specified. Use --help to see available commands.")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    main()
