"""
Script to update configuration from optimization results
"""
import json
import shutil
from pathlib import Path

def update_production_config():
    """Update production config with optimized parameters"""
    
    # Source: notebooks folder
    notebooks_config = Path("notebooks/optimized_hyperparameters.json")
    
    # Destination: config folder
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    production_config = config_dir / "optimized_hyperparameters.json"
    
    if notebooks_config.exists():
        shutil.copy2(notebooks_config, production_config)
        print(f"✅ Updated production config from {notebooks_config}")
    else:
        print(f"❌ Optimization config not found: {notebooks_config}")

if __name__ == "__main__":
    update_production_config()
