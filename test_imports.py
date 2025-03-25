# test_imports.py
import sys

def test_imports():
    print("Testing imports...")
    try:
        # Core components
        from hydra_mlops.api.app import app
        print("✓ API app imported")
        
        # Connectors
        from hydra_mlops.connectors import DataConnector, CSVConnector, DatabaseConnector, APIConnector
        print("✓ Connectors imported")
        
        # Processors
        from hydra_mlops.processors import FeatureProcessor, TextProcessor
        print("✓ Processors imported")
        
        # Trainers
        from hydra_mlops.trainers import ModelTrainer, ClassifierTrainer, RegressorTrainer
        print("✓ Trainers imported")
        
        # Registry
        from hydra_mlops.registry import ModelRegistry, LanceDBModelRegistry
        print("✓ Registry imported")
        
        # Serving
        from hydra_mlops.serving import ModelServer, LanceDBModelServer
        print("✓ Serving imported")
        
        # Monitoring
        from hydra_mlops.monitoring import ModelMonitor, LanceDBModelMonitor
        print("✓ Monitoring imported")
        
        # Utils
        from hydra_mlops.utils import load_config, validate_config
        print("✓ Utils imported")
        
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"\nImport error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
