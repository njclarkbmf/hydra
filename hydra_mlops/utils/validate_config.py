"""
Configuration validation for the LanceDB MLOps Framework.

This module provides functions to validate the configuration and
ensure that all required settings are properly set.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import json

logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the configuration.
    
    Checks for:
    - Required settings
    - Valid values
    - Consistency between related settings
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation error messages)
    """
    errors = []
    
    # Check core settings
    if not config.get("core", {}).get("lancedb_path"):
        errors.append("LanceDB path not configured")
    
    if not config.get("core", {}).get("models_dir"):
        errors.append("Models directory not configured")
    
    # Check that models directory exists or can be created
    models_dir = config.get("core", {}).get("models_dir")
    if models_dir:
        try:
            os.makedirs(models_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create models directory: {str(e)}")
    
    # Check API settings
    api_port = config.get("api", {}).get("port")
    if api_port is not None:
        try:
            port = int(api_port)
            if port < 1 or port > 65535:
                errors.append(f"Invalid API port: {port} (must be between 1 and 65535)")
        except ValueError:
            errors.append(f"Invalid API port: {api_port} (must be an integer)")
    
    api_workers = config.get("api", {}).get("workers")
    if api_workers is not None:
        try:
            workers = int(api_workers)
            if workers < 1:
                errors.append(f"Invalid API workers: {workers} (must be at least 1)")
        except ValueError:
            errors.append(f"Invalid API workers: {api_workers} (must be an integer)")
    
    # Check n8n settings
    n8n_base_url = config.get("n8n", {}).get("base_url")
    if n8n_base_url and not (n8n_base_url.startswith("http://") or n8n_base_url.startswith("https://")):
        errors.append(f"Invalid n8n base URL: {n8n_base_url} (must start with http:// or https://)")
    
    # Check feature toggle consistency
    features = config.get("features", {})
    if features.get("automatic_retraining") and not features.get("model_monitoring"):
        errors.append("Automatic retraining requires model monitoring to be enabled")
    
    # Check security settings
    if config.get("security", {}).get("api_key_enabled") and not config.get("security", {}).get("api_key"):
        errors.append("API key authentication enabled but no API key configured")
    
    # Check for potentially unsafe configurations
    if not config.get("security", {}).get("api_key_enabled") and n8n_base_url and "localhost" not in n8n_base_url:
        errors.append("WARNING: API key authentication disabled with non-localhost n8n URL")
    
    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_and_print_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration and print results.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    is_valid, errors = validate_config(config)
    
    if is_valid:
        logger.info("Configuration validation passed")
        
        # Print configuration summary
        print("Configuration Summary:")
        print(f"- LanceDB Path: {config.get('core', {}).get('lancedb_path')}")
        print(f"- Models Directory: {config.get('core', {}).get('models_dir')}")
        print(f"- API: {config.get('api', {}).get('host')}:{config.get('api', {}).get('port')}")
        
        # Print enabled features
        features = config.get("features", {})
        enabled_features = [k for k, v in features.items() if v]
        print(f"- Enabled Features: {', '.join(enabled_features) or 'None'}")
        
        # Print security settings
        security = config.get("security", {})
        print(f"- API Key Authentication: {'Enabled' if security.get('api_key_enabled') else 'Disabled'}")
        
        return True
    else:
        logger.error("Configuration validation failed")
        
        # Print validation errors
        print("Configuration Errors:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load and validate configuration
    from . import load_config
    config = load_config()
    
    # Validate and print results
    is_valid = validate_and_print_config(config)
    
    # Exit with appropriate code
    exit(0 if is_valid else 1)
