import os
from typing import Dict, List, Any, Optional
import time
import logging
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import json
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import optional modules based on environment variables
enable_ab_testing = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
enable_drift_detection = os.getenv("ENABLE_DRIFT_DETECTION", "false").lower() == "true"
enable_experiment_tracking = os.getenv("ENABLE_EXPERIMENT_TRACKING", "false").lower() == "true"
enable_auto_retraining = os.getenv("ENABLE_AUTOMATIC_RETRAINING", "false").lower() == "true"
enable_model_monitoring = os.getenv("ENABLE_MODEL_MONITORING", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LanceDB MLOps Framework API",
    description="API for the LanceDB MLOps Framework",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678/webhook/")
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "~/.lancedb")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")

# Optional API key authentication
API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Middleware for API key authentication
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY_ENABLED:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY and request.url.path != "/health":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"},
            )
    return await call_next(request)

# Request Models
class DataIngestionRequest(BaseModel):
    connector_type: str
    connector_params: Dict[str, Any]
    query_params: Optional[Dict[str, Any]] = None
    lancedb_path: Optional[str] = None
    table_name: str

class ModelTrainingRequest(BaseModel):
    table_name: str
    feature_column: str
    label_column: str
    model_type: Optional[str] = None
    model_id: str
    model_params: Optional[Dict[str, Any]] = None
    lancedb_path: Optional[str] = None
    models_dir: Optional[str] = None

class InferenceRequest(BaseModel):
    model_id: str
    features: List[Any]
    lancedb_path: Optional[str] = None

class DriftDetectionRequest(BaseModel):
    model_id: str
    reference_table: str
    current_table: str
    lancedb_path: Optional[str] = None

class ABTestRequest(BaseModel):
    model_a_id: str
    model_b_id: str
    test_name: str
    traffic_split: Optional[float] = 0.5
    lancedb_path: Optional[str] = None

# API Routes
@app.post("/api/data/ingest")
async def ingest_data(request: DataIngestionRequest):
    """
    Ingest data into LanceDB using the specified connector.
    """
    try:
        # Set default LanceDB path if not provided
        if request.lancedb_path is None:
            request.lancedb_path = LANCEDB_PATH
            
        # Call n8n workflow for data ingestion
        response = requests.post(
            f"{N8N_BASE_URL}ingest-data",
            json=request.dict()
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error ingesting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting data: {str(e)}")

@app.post("/api/models/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a model using the specified data and parameters.
    """
    try:
        # Set defaults if not provided
        if request.lancedb_path is None:
            request.lancedb_path = LANCEDB_PATH
        if request.models_dir is None:
            request.models_dir = MODELS_DIR
        if request.model_type is None:
            request.model_type = os.getenv("DEFAULT_MODEL_TYPE", "classifier")
            
        # Call n8n workflow for model training
        response = requests.post(
            f"{N8N_BASE_URL}train-model",
            json=request.dict()
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/api/models/predict")
async def predict(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Make predictions using the specified model.
    """
    try:
        # Set default LanceDB path if not provided
        if request.lancedb_path is None:
            request.lancedb_path = LANCEDB_PATH
            
        # Call n8n workflow for prediction
        response = requests.post(
            f"{N8N_BASE_URL}predict",
            json=request.dict()
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Model Registry API
@app.get("/api/models")
async def list_models(lancedb_path: Optional[str] = None):
    """
    List all models in the registry.
    """
    try:
        # Set default LanceDB path if not provided
        if lancedb_path is None:
            lancedb_path = LANCEDB_PATH
            
        # Import LanceDB here to avoid global dependencies
        import lancedb
        import pandas as pd
        
        db = lancedb.connect(lancedb_path)
        if "model_registry" not in db.table_names():
            return {"models": []}
            
        registry = db.open_table("model_registry")
        models_df = registry.to_pandas()
        
        # Group by model_id to get latest version
        models = []
        for model_id in models_df["model_id"].unique():
            model_versions = models_df[models_df["model_id"] == model_id]
            latest_version = model_versions.sort_values("created_at", ascending=False).iloc[0]
            models.append({
                "model_id": latest_version["model_id"],
                "version": latest_version["version"],
                "created_at": latest_version["created_at"],
                "metrics": latest_version["metrics"]
            })
        
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/api/models/{model_id}/versions")
async def list_model_versions(model_id: str, lancedb_path: Optional[str] = None):
    """
    List all versions of a specific model.
    """
    try:
        # Set default LanceDB path if not provided
        if lancedb_path is None:
            lancedb_path = LANCEDB_PATH
            
        # Import LanceDB here to avoid global dependencies
        import lancedb
        import pandas as pd
        
        db = lancedb.connect(lancedb_path)
        if "model_registry" not in db.table_names():
            return {"versions": []}
            
        registry = db.open_table("model_registry")
        models_df = registry.to_pandas()
        
        model_versions = models_df[models_df["model_id"] == model_id]
        if model_versions.empty:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        versions = []
        for _, version in model_versions.iterrows():
            versions.append({
                "model_id": version["model_id"],
                "version": version["version"],
                "created_at": version["created_at"],
                "metrics": version["metrics"]
            })
        
        return {"versions": sorted(versions, key=lambda v: v["created_at"], reverse=True)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing model versions: {str(e)}")

# Optional API routes based on feature flags
if enable_drift_detection:
    @app.post("/api/monitoring/drift")
    async def detect_drift(request: DriftDetectionRequest):
        """
        Detect drift between reference and current data.
        """
        try:
            # Set default LanceDB path if not provided
            if request.lancedb_path is None:
                request.lancedb_path = LANCEDB_PATH
                
            # Call n8n workflow for drift detection
            response = requests.post(
                f"{N8N_BASE_URL}detect-drift",
                json=request.dict()
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error detecting drift: {str(e)}")

if enable_ab_testing:
    @app.post("/api/testing/ab-test")
    async def create_ab_test(request: ABTestRequest):
        """
        Create an A/B test between two models.
        """
        try:
            # Set default LanceDB path if not provided
            if request.lancedb_path is None:
                request.lancedb_path = LANCEDB_PATH
                
            # Call n8n workflow for A/B testing
            response = requests.post(
                f"{N8N_BASE_URL}create-ab-test",
                json=request.dict()
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error creating A/B test: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating A/B test: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "features": {
            "ab_testing": enable_ab_testing,
            "drift_detection": enable_drift_detection,
            "experiment_tracking": enable_experiment_tracking,
            "auto_retraining": enable_auto_retraining,
            "model_monitoring": enable_model_monitoring
        }
    }

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    workers = int(os.getenv("API_WORKERS", "4"))
    
    # Log the configuration
    logger.info(f"Starting LanceDB MLOps Framework API on {host}:{port}")
    logger.info(f"LanceDB path: {LANCEDB_PATH}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"n8n.io base URL: {N8N_BASE_URL}")
    logger.info(f"Feature flags: AB Testing={enable_ab_testing}, Drift Detection={enable_drift_detection}")
    
    # Start the server
    uvicorn.run("api.app:app", host=host, port=port, workers=workers, log_level=os.getenv("LOG_LEVEL", "info").lower())
