# Installation Guide

This comprehensive guide will walk you through the process of setting up and installing the Hydra MLOps Framework.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** (Python 3.9 recommended)
- **pip** (Python package manager)
- **Docker** (version 19.03 or higher) and **Docker Compose** (version 1.27 or higher) for containerized deployment
- **Git** (version 2.20 or higher)
- **Node.js** (version 14 or higher) for n8n.io if installing locally

### System Requirements

- **Memory**: Minimum 4GB RAM, 8GB+ recommended (LanceDB vector operations are memory-intensive)
- **Storage**: Minimum 1GB free space, more for large datasets and models
- **CPU**: Multi-core processor recommended for parallel processing

## Installation Options

You can install the framework in several ways:

1. **Local Installation**: Install directly on your machine
2. **Docker Installation**: Run in containers (recommended for production)
3. **Development Installation**: For contributing to the framework

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hydra-mlops.git
cd hydra-mlops
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required Python packages, including:
- FastAPI for the API layer
- LanceDB for vector storage
- PyArrow for data handling
- scikit-learn for ML models
- SentenceTransformers for text embeddings

### 5. Create a .env File

```bash
cp .env.example .env
```

Edit the `.env` file to configure your installation. At a minimum, set:

```
# Core Settings
LANCEDB_PATH=~/.lancedb
MODELS_DIR=./models
TEMP_DIR=./temp

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=info

# n8n.io Integration
N8N_BASE_URL=http://localhost:5678/webhook/
```

### 6. Create Required Directories

```bash
mkdir -p ~/.lancedb ./models ./temp
```

### 7. Install n8n.io

You need to install n8n separately:

```bash
npm install n8n -g
```

### 8. Start the Framework

Start n8n.io:
```bash
n8n start
```

In a new terminal, start the FastAPI application:
```bash
cd hydra-mlops
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m hydra_mlops.api.app
```

### 9. Import n8n.io Workflows

1. Open your browser and navigate to `http://localhost:5678`
2. Create an account or sign in
3. Go to **Workflows** → **Import from File**
4. Select each workflow JSON file from the `hydra_mlops/workflows` directory:
   - `ab_testing.json`
   - `data_ingestion.json`
   - `drift_detection.json`
   - `model_inference.json`
   - `model_training.json`
5. Activate each workflow by clicking the toggle switch

### 10. Verify Installation

Open your browser and navigate to:
- `http://localhost:8000/docs` - FastAPI Swagger UI
- `http://localhost:5678` - n8n.io workflow editor

Try a simple health check:
```bash
curl http://localhost:8000/health
```

Should return something like:
```json
{
  "status": "healthy",
  "timestamp": 1684912345.678,
  "version": "1.0.0",
  "features": {
    "ab_testing": false,
    "drift_detection": false,
    "experiment_tracking": false,
    "auto_retraining": false,
    "model_monitoring": true
  }
}
```

## Docker Installation

Using Docker is the recommended way to deploy the framework in production.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hydra-mlops.git
cd hydra-mlops
```

### 2. Create a .env File

```bash
cp .env.example .env
```

Edit the `.env` file to configure your installation. Key settings for Docker:

```
# Docker Settings
DOCKER_NETWORK_NAME=hydra-mlops-network
DOCKER_DATA_VOLUME=hydra-mlops-data

# API Settings
API_PORT=8000
```

### 3. Build and Start with Docker Compose

```bash
docker-compose up -d
```

This command will:
1. Build the Docker images for the services
2. Create the required Docker volumes for persistence
3. Start all services in the background

### 4. Import n8n.io Workflows

1. Open your browser and navigate to `http://localhost:5678`
2. Create an account or sign in
