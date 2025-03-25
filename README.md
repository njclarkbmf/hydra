# Hydra MLOps Framework

A comprehensive, pluggable MLOps framework that uses LanceDB as its core component for vector storage and n8n.io for workflow orchestration. This framework provides a modular approach to machine learning operations, allowing you to easily swap components and customize your ML pipeline.

## Features

- **Modular Architecture**: Easily swap out components through well-defined interfaces and plugins
- **LanceDB Integration**: Leverages LanceDB's vector database capabilities for efficient storage of embeddings, model metadata, and features
- **n8n.io Workflow Orchestration**: Uses JSON-based workflows for pipeline automation and scheduling
- **Pluggable Components**: Includes data connectors, feature processors, model trainers, and more
- **FastAPI Backend**: Provides a clean API for interacting with the MLOps pipeline
- **Advanced Features**: 
  - A/B Testing capabilities with traffic splitting and performance tracking
  - Drift detection monitoring with cosine similarity and KS-test statistics
  - Automated retraining based on configurable triggers
  - Model versioning and lineage tracking with vector similarity search
  - Experiment tracking with metadata storage
  - Model serving with monitoring and feedback integration

## Architecture

The framework consists of several key layers that work together to create a complete MLOps platform:

### 1. Data Layer (LanceDB)

LanceDB serves as the core storage and retrieval engine, providing:

- **Vector Storage**: Efficiently stores and indexes high-dimensional vectors for rapid similarity search and retrieval
- **Multi-Table Support**: Organizes different types of data into separate tables:
  - `model_registry`: Stores model metadata, version info, and vector representations
  - `prediction_logs`: Records inference details including features, predictions, and optional ground truth
  - `model_monitoring`: Tracks model performance metrics and drift statistics
  - `ab_tests`: Manages A/B test configurations and results

LanceDB's vector search capabilities enable powerful features like similar model retrieval, embedding-based drift detection, and semantic model organization.

### 2. Workflow Orchestration Layer (n8n.io)

The n8n.io workflows handle the process coordination and event-driven execution:

- **Data Ingestion Workflow**: Imports data from various sources, transforms it, and stores it in LanceDB
- **Model Training Workflow**: Trains models on the prepared data and registers them in LanceDB
- **Model Inference Workflow**: Serves predictions using registered models and logs results
- **Drift Detection Workflow**: Monitors data distributions for changes and triggers alerts/actions
- **A/B Testing Workflow**: Creates and manages experiments to compare model performance

These workflows are defined in JSON files that can be imported directly into n8n.io, edited visually, and extended with additional functionality.

### 3. Component Layer

The pluggable components are Python modules that implement standardized interfaces:

- **Data Connectors**: Import data from various sources:
  - `CSVConnector`: Reads from CSV files with automatic column detection
  - `DatabaseConnector`: Connects to SQL databases with configurable queries
  - `APIConnector`: Retrieves data from REST APIs with authentication support

- **Model Trainers**: Train ML models with standardized interfaces:
  - `ClassifierTrainer`: Implements classification models with sklearn backends
  - `RegressorTrainer`: Implements regression models with sklearn backends

- **Model Registry**: Tracks models and their metadata:
  - `LanceDBModelRegistry`: Stores models with vector embeddings for similarity search
  - Functions for registering, retrieving, and comparing models

- **Model Servers**: Handle model loading and inference:
  - `LanceDBModelServer`: Serves models from the registry and logs predictions
  - Support for both batch and individual predictions

- **Monitoring Tools**: Track model performance and data drift:
  - `LanceDBModelMonitor`: Monitors drift and model performance metrics
  - Configurable alerting and automated retraining triggers

### 4. API Layer

The FastAPI application exposes REST endpoints for interacting with the framework:

- **/api/data/ingest**: Ingest data from various sources
- **/api/models/train**: Train new models
- **/api/models/predict**: Make predictions with models
- **/api/models**: List and retrieve model information
- **/api/monitoring/drift**: Detect and analyze drift
- **/api/testing/ab-test**: Create and manage A/B tests

The API layer includes authentication, validation, and comprehensive documentation via Swagger UI.

## Key Design Principles

The framework follows several important design principles:

1. **Separation of Concerns**: Each component has a well-defined responsibility and interface
2. **Extensibility**: All components can be extended or replaced with custom implementations
3. **Configuration over Code**: Features can be enabled/disabled through configuration
4. **Reproducibility**: All operations are logged and versioned for reproducibility
5. **Vector-Centricity**: The framework leverages vector representations throughout for enhanced capabilities

## Performance Considerations

The framework is optimized for performance in several ways:

- **Vector Indexing**: LanceDB's IVF_PQ indexing for fast similarity search
- **Batch Processing**: Support for batch operations in data ingestion and inference
- **Asynchronous Operations**: Non-blocking API operations for better concurrency
- **Lazy Loading**: Models are loaded on-demand and cached for repeated use
- **Configurable Worker Pools**: Adjustable parallelism for processing-intensive operations

For large-scale deployments, consider:
- Increasing the number of API workers
- Using more powerful hardware for vector operations
- Implementing a load balancer for horizontal scaling
- Monitoring memory usage as vector databases are memory-intensive

## Getting Started

See [INSTALLATION.md](INSTALLATION.md) for detailed installation and setup instructions.

## Basic Usage

```python
# Example: Data ingestion
import requests

response = requests.post(
    "http://localhost:8000/api/data/ingest",
    json={
        "connector_type": "csv",
        "connector_params": {"file_path": "/path/to/data.csv"},
        "lancedb_path": "~/.lancedb",
        "table_name": "my_data"
    }
)
print(response.json())

# Example: Model training
response = requests.post(
    "http://localhost:8000/api/models/train",
    json={
        "table_name": "my_data",
        "feature_column": "vector",
        "label_column": "category",
        "model_type": "classifier",
        "model_id": "my_classifier",
        "lancedb_path": "~/.lancedb",
        "models_dir": "/path/to/models"
    }
)
print(response.json())

# Example: Model inference
response = requests.post(
    "http://localhost:8000/api/models/predict",
    json={
        "model_id": "my_classifier",
        "features": [[0.1, 0.2, 0.3, 0.4]],
        "lancedb_path": "~/.lancedb"
    }
)
print(response.json())
```

## Real-World Examples

### Example 1: Document Classification Pipeline

```bash
# 1. Ingest documents from a CSV file
python cli.py ingest --connector-type csv --table-name documents \
  --params '{"file_path": "documents.csv", "text_columns": ["content"]}'

# 2. Train a classifier
python cli.py train --table-name documents --feature-column vector \
  --label-column category --model-type classifier --model-id doc_classifier

# 3. Make predictions on new documents
python cli.py predict --model-id doc_classifier \
  --features "[[0.1, 0.2, 0.3, ...]]"

# 4. Monitor for drift
python cli.py detect-drift --model-id doc_classifier \
  --reference-table documents --current-table new_documents
```

### Example 2: A/B Testing New Models

```bash
# 1. Create an A/B test between two models
python cli.py create-ab-test --model-a sentiment_model_v1 \
  --model-b sentiment_model_v2 --test-name "Sentiment Analysis Comparison" \
  --traffic-split 0.5

# 2. Make prediction (routing handled automatically)
curl -X POST http://localhost:8000/api/models/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, ...]], "ab_test_id": "Sentiment Analysis Comparison"}'
```

## Feature Toggle Usage

The framework uses environment variables to enable/disable features:

```bash
# Enable A/B testing
ENABLE_AB_TESTING=true

# Enable drift detection
ENABLE_DRIFT_DETECTION=true

# Enable automatic retraining
ENABLE_AUTOMATIC_RETRAINING=true
```

When a feature is enabled, additional API endpoints and functionality become available:

- `ENABLE_AB_TESTING=true` → Adds `/api/testing/ab-test` endpoint and A/B test routing
- `ENABLE_DRIFT_DETECTION=true` → Adds `/api/monitoring/drift` endpoint and drift monitoring
- `ENABLE_AUTOMATIC_RETRAINING=true` → Activates automatic model retraining when drift is detected

## Vector-Based Operations

This framework leverages vector operations for several unique capabilities:

1. **Model Similarity Search**: Find similar models based on their vector representations
   ```python
   # Find models similar to current feature vectors
   similar_models = registry.find_similar_models(feature_vector, limit=5)
   ```

2. **Drift Detection with Vector Similarity**:
   ```python
   # Detect distribution changes using vector statistics
   drift_metrics = monitor.detect_drift(model_id, reference_data, current_data)
   ```

3. **Semantic Model Organization**: Models are organized by their vector representations, creating a semantic space of models that can be queried.

## Configuration

The framework can be configured through environment variables or `.env` files. See [Configuration](configuration.md) for details.

## Docker Deployment

The framework includes Docker support for easy deployment. See [Docker Deployment](docker.md) for instructions.

## Contributing

We welcome contributions to this project. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
