# Docker Deployment Guide

This guide provides instructions for deploying the LanceDB MLOps Framework using Docker.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker (version 19.03 or higher)
- Docker Compose (version 1.27 or higher)
- Git

## Quick Start

The quickest way to get started with the framework is to use the provided Docker Compose configuration.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lancedb-mlops-framework.git
cd lancedb-mlops-framework
```

### 2. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file to configure your deployment. At a minimum, set:

- `LANCEDB_PATH`: Path to store LanceDB data (inside the container)
- `MODELS_DIR`: Directory to store model files (inside the container)
- `API_PORT`: Port to expose the API on the host

### 3. Start the Services

```bash
docker-compose up -d
```

This will start:
- The FastAPI application
- n8n.io for workflow orchestration
- Volumes for persistent data storage

### 4. Verify the Deployment

Open your browser and navigate to:

- `http://localhost:8000/docs` - FastAPI Swagger UI
- `http://localhost:5678` - n8n.io workflow editor

## Customizing the Deployment

### Using Custom Docker Images

If you need to customize the Docker images, you can modify the `Dockerfile` and rebuild:

```bash
docker-compose build
docker-compose up -d
```

### Environment Variable Reference

The following environment variables can be set in the `.env` file:

#### Core Settings
- `LANCEDB_PATH`: Path to store LanceDB data
- `MODELS_DIR`: Directory to store model files
- `TEMP_DIR`: Directory for temporary files

#### API Settings
- `API_HOST`: Host to bind the API to
- `API_PORT`: Port to expose the API on
- `API_WORKERS`: Number of Uvicorn workers
- `LOG_LEVEL`: Logging level (debug, info, warning, error)

#### Feature Toggles
- `ENABLE_AB_TESTING`: Enable A/B testing (true/false)
- `ENABLE_DRIFT_DETECTION`: Enable drift detection (true/false)
- `ENABLE_EXPERIMENT_TRACKING`: Enable experiment tracking (true/false)
- `ENABLE_AUTOMATIC_RETRAINING`: Enable automatic retraining (true/false)
- `ENABLE_MODEL_MONITORING`: Enable model monitoring (true/false)

#### Docker Settings
- `DOCKER_NETWORK_NAME`: Name for the Docker network
- `DOCKER_DATA_VOLUME`: Name for the data volume
- `DOCKER_N8N_VOLUME`: Name for the n8n data volume

## Advanced Configuration

### Using External PostgreSQL Database

To use an external PostgreSQL database:

1. Uncomment the PostgreSQL service in `docker-compose.yml`
2. Set the database environment variables in `.env`
3. Restart the services:

```bash
docker-compose down
docker-compose up -d
```

### Scaling the API

You can scale the API service horizontally:

```bash
docker-compose up -d --scale api=3
```

### Persistent Storage

The framework uses Docker volumes for persistent storage:

- `data-volume`: Stores LanceDB data and models
- `n8n-data`: Stores n8n.io workflows and credentials

You can back up these volumes using Docker's volume backup features.

## Production Deployment Considerations

### Security

For production deployments, consider the following security measures:

1. Enable API key authentication:
   ```
   API_KEY_ENABLED=true
   API_KEY=your-secure-api-key
   ```

2. Set up HTTPS:
   - Use a reverse proxy like Nginx with SSL certificates
   - Configure the API to use HTTPS

3. Use secrets management for sensitive credentials

### Monitoring

Set up monitoring for your Docker containers:

1. Use Docker's built-in health checks:
   - API service has health checks enabled by default
   - n8n.io service has health checks enabled by default

2. Integrate with container monitoring tools:
   - Prometheus and Grafana
   - Datadog
   - New Relic

### Logging

Configure logging for better observability:

1. Set the log level in `.env`:
   ```
   LOG_LEVEL=info
   ```

2. Collect logs with a logging service:
   ```bash
   docker-compose logs -f
   ```

3. For production, consider using a dedicated logging solution:
   - ELK Stack (Elasticsearch, Logstash, Kibana)
   - Graylog
   - Loki with Grafana

## Troubleshooting

### Common Issues

#### Container Fails to Start

Check the logs:
```bash
docker-compose logs api
```

#### Volume Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix permissions on the host
sudo chown -R 1000:1000 ./data
```

#### Network Connection Issues

If services can't connect to each other:

```bash
# Check the network
docker network inspect lancedb-mlops-network

# Restart the network
docker-compose down
docker-compose up -d
```

### Getting Help

If you encounter issues not covered here, please:

1. Check the GitHub issues to see if your problem has been reported
2. Create a new issue with details about your problem
3. Join our community chat for real-time help

## Updating the Framework

To update to the latest version:

```bash
git pull
docker-compose build
docker-compose up -d
```

## Uninstalling

To remove the deployment:

```bash
# Stop and remove containers
docker-compose down

# Optionally, remove volumes
docker-compose down -v
```
