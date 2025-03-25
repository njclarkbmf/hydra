"""
Utility modules for the LanceDB MLOps Framework.

This package provides various utility functions and classes used throughout
the framework, including configuration, validation, and common operations.
"""

import os
import logging
from typing import Dict, Any, Optional
import json
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from various sources.
    
    The function loads configuration in the following order of precedence:
    1. Environment variables
    2. .env file
    3. YAML configuration file
    4. Default values
    
    Args:
        config_path: Optional path to a YAML configuration file
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    # Load default configuration
    config = {
        "core": {
            "lancedb_path": "~/.lancedb",
            "models_dir": "./models",
            "temp_dir": "./temp",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "log_level": "info",
        },
        "features": {
            "ab_testing": False,
            "drift_detection": False,
            "experiment_tracking": False,
            "automatic_retraining": False,
            "model_monitoring": True,
        },
        "n8n": {
            "base_url": "http://localhost:5678/webhook/",
            "workflows_dir": "./workflows",
        },
        "models": {
            "default_type": "classifier",
            "default_embedding_model": "all-MiniLM-L6-v2",
        },
        "security": {
            "api_key_enabled": False,
            "api_key": "",
        },
    }
    
    # Load configuration from YAML file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                _merge_configs(config, yaml_config)
                logger.debug(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    # Load configuration from .env file
    load_dotenv()
    
    # Override with environment variables
    _override_from_env(config)
    
    return config

def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
    """
    Recursively merge configuration dictionaries.
    
    Args:
        base_config: Base configuration (modified in-place)
        override_config: Overriding configuration
    """
    for key, value in override_config.items():
        if (
            key in base_config
            and isinstance(base_config[key], dict)
            and isinstance(value, dict)
        ):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value

def _override_from_env(config: Dict[str, Any]) -> None:
    """
    Override configuration with environment variables.
    
    Environment variables are mapped to nested configuration keys.
    For example, API_PORT overrides config["api"]["port"].
    
    Args:
        config: Configuration dictionary (modified in-place)
    """
    # Core settings
    if os.getenv("LANCEDB_PATH"):
        config["core"]["lancedb_path"] = os.getenv("LANCEDB_PATH")
    if os.getenv("MODELS_DIR"):
        config["core"]["models_dir"] = os.getenv("MODELS_DIR")
    if os.getenv("TEMP_DIR"):
        config["core"]["temp_dir"] = os.getenv("TEMP_DIR")
    
    # API settings
    if os.getenv("API_HOST"):
        config["api"]["host"] = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config["api"]["port"] = int(os.getenv("API_PORT"))
    if os.getenv("API_WORKERS"):
        config["api"]["workers"] = int(os.getenv("API_WORKERS"))
    if os.getenv("LOG_LEVEL"):
        config["api"]["log_level"] = os.getenv("LOG_LEVEL").lower()
    
    # Feature toggles
    if os.getenv("ENABLE_AB_TESTING"):
        config["features"]["ab_testing"] = os.getenv("ENABLE_AB_TESTING").lower() == "true"
    if os.getenv("ENABLE_DRIFT_DETECTION"):
        config["features"]["drift_detection"] = os.getenv("ENABLE_DRIFT_DETECTION").lower() == "true"
    if os.getenv("ENABLE_EXPERIMENT_TRACKING"):
        config["features"]["experiment_tracking"] = os.getenv("ENABLE_EXPERIMENT_TRACKING").lower() == "true"
    if os.getenv("ENABLE_AUTOMATIC_RETRAINING"):
        config["features"]["automatic_retraining"] = os.getenv("ENABLE_AUTOMATIC_RETRAINING").lower() == "true"
    if os.getenv("ENABLE_MODEL_MONITORING"):
        config["features"]["model_monitoring"] = os.getenv("ENABLE_MODEL_MONITORING").lower() == "true"
    
    # n8n settings
    if os.getenv("N8N_BASE_URL"):
        config["n8n"]["base_url"] = os.getenv("N8N_BASE_URL")
    if os.getenv("N8N_WORKFLOWS_DIR"):
        config["n8n"]["workflows_dir"] = os.getenv("N8N_WORKFLOWS_DIR")
    
    # Model settings
    if os.getenv("DEFAULT_MODEL_TYPE"):
        config["models"]["default_type"] = os.getenv("DEFAULT_MODEL_TYPE")
    if os.getenv("DEFAULT_EMBEDDING_MODEL"):
        config["models"]["default_embedding_model"] = os.getenv("DEFAULT_EMBEDDING_MODEL")
    
    # Security settings
    if os.getenv("API_KEY_ENABLED"):
        config["security"]["api_key_enabled"] = os.getenv("API_KEY_ENABLED").lower() == "true"
    if os.getenv("API_KEY"):
        config["security"]["api_key"] = os.getenv("API_KEY")

# Import specific utility modules
from .validate_config import validate_config
