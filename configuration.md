# Configuration Guide

This document describes how to configure the LanceDB MLOps Framework to suit your needs.

## Configuration Methods

The framework can be configured through the following methods:

1. Environment variables
2. `.env` file
3. Configuration file
4. Command-line arguments (for some components)

The order of precedence is: command-line arguments > environment variables > `.env` file > configuration file > defaults.

## Environment Variables

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LANCEDB_PATH` | Path to store LanceDB data | `~/.lancedb` |
| `MODELS_DIR` | Directory to store model files | `./models` |
| `TEMP_DIR` | Directory for temporary files | `./temp` |

### API Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Host to bind the API to | `0.0.0.0` |
| `API_PORT` | Port to expose the API on | `8000` |
| `API_WORKERS` | Number of Uvicorn workers | `4` |
| `LOG_LEVEL` | Logging level | `info` |

### n8n.io Integration

| Variable | Description | Default |
|----------|-------------|---------|
| `N8N_BASE_URL` | URL of n8n.io webhooks | `http://localhost:5678/webhook/` |
| `N8N_WORKFLOWS_DIR` | Directory containing workflow JSON files | `./workflows` |

### Feature Toggles

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_AB_TESTING` | Enable A/B testing | `false` |
| `ENABLE_DRIFT_DETECTION` | Enable drift detection | `false` |
| `ENABLE_EXPERIMENT_TRACKING` | Enable experiment tracking | `false` |
| `ENABLE_AUTOMATIC_RETRAINING` | Enable automatic retraining | `false` |
| `ENABLE_MODEL_MONITORING` | Enable model monitoring | `true` |

### Model Default Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_MODEL_TYPE` | Default model type | `classifier` |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model | `all-MiniLM-L6-v2` |

### Database Settings (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `password` |
| `DB_NAME` | Database name | `mlops` |

### Security Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY_ENABLED` | Enable API key authentication | `false` |
| `API_KEY` | API key for authentication | `` |

### Notification Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_NOTIFICATIONS` | Enable notifications | `false` |
| `SLACK_WEBHOOK_URL` | Slack webhook URL | `` |
| `EMAIL_NOTIFICATIONS` | Email addresses for notifications | `` |

### Performance Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `BATCH_SIZE` | Batch size for processing | `32` |
| `MAX_WORKERS` | Maximum number of parallel workers | `4` |

### Storage Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_S3_STORAGE` | Enable S3 storage for models | `false` |
| `S3_BUCKET` | S3 bucket name | `mlops-models` |
| `S3_REGION` | S3 region | `us-west-2` |
| `AWS_ACCESS_KEY_ID` | AWS access key ID | `` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | `` |

### Docker Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_NETWORK_NAME` | Name for the Docker network | `lancedb-mlops-network` |
| `DOCKER_DATA_VOLUME` | Name for the data volume | `lancedb-mlops-data` |
| `DOCKER_N8N_VOLUME` | Name for the n8n data volume | `lancedb-mlops-n8n` |

## Using a .env File

The framework supports loading configuration from a `.env` file in the root directory. This is a convenient way to set multiple configuration options without modifying the system environment.

Example `.env` file:

```
# Core Settings
LANCEDB_PATH=~/.lancedb
MODELS_DIR=./models
TEMP_DIR=./temp

# API Settings
API_PORT=8000
LOG_LEVEL=info

# Feature Toggles
ENABLE_AB_TESTING=true
ENABLE_DRIFT_DETECTION=true
```

## Configuration File

For more advanced configuration, you can create a configuration file at `config/config.yaml`. This allows for hierarchical configuration and more complex settings.

Example `config.yaml`:

```yaml
core:
  lancedb_path: ~/.lancedb
  models_dir: ./models
  temp_dir: ./temp

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  log_level: info

features:
  ab_testing: true
  drift_detection: true
  experiment_tracking: false
  automatic_retraining: false
  model_monitoring: true

models:
  default_type: classifier
  default_embedding_model: all-MiniLM-L6-v2
  
  classifiers:
    - name: random_forest
      default_params:
        n_estimators: 100
        max_depth: 10
    - name: logistic_regression
      default_params:
        C: 1.0
        max_iter: 1000
        
  regressors:
    - name: random_forest
      default_params:
        n_estimators: 100
        max_depth: 10
    - name: linear_regression
      default_params: {}

connectors:
  csv:
    default_encoding: utf-8
    default_delimiter: ","
  database:
    timeout: 30
  api:
    timeout: 10
    retries: 3

monitoring:
  drift_threshold: 0.05
  performance_check_interval: 3600  # in seconds
  retention_period: 30  # in days
```

## Feature Toggles

The framework uses feature toggles to enable or disable certain functionality. This allows you to customize the framework to your needs without modifying the code.

### AB Testing

When enabled (`ENABLE_AB_TESTING=true`), the framework provides A/B testing capabilities:

- Create A/B tests to compare model performance
- Route traffic between different model versions
- Track and analyze test results

### Drift Detection

When enabled (`ENABLE_DRIFT_DETECTION=true`), the framework monitors for data drift:

- Compare current data distribution to training data
- Alert when significant drift is detected
- Analyze feature-level drift

### Experiment Tracking

When enabled (`ENABLE_EXPERIMENT_TRACKING=true`), the framework tracks model experiments:

- Log hyperparameters, metrics, and artifacts
- Compare experiments
- Visualize results

### Automatic Retraining

When enabled (`ENABLE_AUTOMATIC_RETRAINING=true`), the framework can automatically retrain models:

- Trigger retraining when performance degrades
- Trigger retraining when drift is detected
- Schedule periodic retraining

### Model Monitoring

When enabled (`ENABLE_MODEL_MONITORING=true`), the framework monitors model performance:

- Track inference metrics
- Alert on performance degradation
- Log prediction distribution

## Configuration Best Practices

### Development Environment

For development, we recommend:

```
LOG_LEVEL=debug
ENABLE_MODEL_MONITORING=true
API_KEY_ENABLED=false
```

### Testing Environment

For testing, we recommend:

```
LOG_LEVEL=info
ENABLE_MODEL_MONITORING=true
ENABLE_AB_TESTING=true
ENABLE_DRIFT_DETECTION=true
API_KEY_ENABLED=true
API_KEY=test-api-key
```

### Production Environment

For production, we recommend:

```
LOG_LEVEL=warning
ENABLE_MODEL_MONITORING=true
ENABLE_DRIFT_DETECTION=true
ENABLE_AUTOMATIC_RETRAINING=true
API_KEY_ENABLED=true
API_KEY=your-secure-api-key
```

## Configuring Specific Components

### Connectors

You can configure default settings for connectors:

```
# CSV connector
CSV_DEFAULT_ENCODING=utf-8
CSV_DEFAULT_DELIMITER=","

# Database connector
DB_CONNECTION_TIMEOUT=30

# API connector
API_CONNECTION_TIMEOUT=10
API_CONNECTION_RETRIES=3
```

### Trainers

You can configure default settings for model trainers:

```
# Common settings
MODEL_TEST_SIZE=0.2
MODEL_RANDOM_STATE=42

# Classifier settings
CLASSIFIER_TYPE=random_forest
CLASSIFIER_N_ESTIMATORS=100
CLASSIFIER_MAX_DEPTH=10

# Regressor settings
REGRESSOR_TYPE=random_forest
REGRESSOR_N_ESTIMATORS=100
REGRESSOR_MAX_DEPTH=10
```

### Monitoring

You can configure monitoring settings:

```
# Drift detection
DRIFT_THRESHOLD=0.05
DRIFT_CHECK_INTERVAL=3600  # in seconds

# Performance monitoring
PERFORMANCE_THRESHOLD=0.7
PERFORMANCE_CHECK_INTERVAL=3600  # in seconds

# Alerting
ALERT_CHANNELS=slack,email
```

## Applying Configuration Changes

After changing the configuration, you need to restart the affected components:

- For API changes: Restart the FastAPI application
- For n8n.io workflow changes: Reload the workflows in n8n.io
- For Docker deployment: Restart the containers with `docker-compose restart`

## Validating Configuration

You can validate your configuration with the included validation tool:

```bash
python -m lancedb_mlops.utils.validate_config
```

This will check your configuration for common errors and inconsistencies.
