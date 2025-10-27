"""
Lightweight sanity tests for the MLOps retraining pipeline.
These run in GitHub Actions to ensure core code and data files are intact
before starting expensive SageMaker retraining.
"""

import os
import json
import importlib

def test_key_paths_exist():
    """Ensure critical project folders exist."""
    for path in ["training/train.py", "pipeline/run_training_job.py", "config/config.yaml"]:
        assert os.path.exists(path), f"Missing required file: {path}"

def test_model_script_imports():
    """Check that main scripts import without errors."""
    for module in ["training.train", "pipeline.run_training_job"]:
        importlib.import_module(module)

def test_config_structure():
    """Basic validation of YAML config structure."""
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "aws" in cfg and "training" in cfg, "Config missing expected keys"

def test_metrics_file_format():
    """If metrics.json exists, ensure it is valid JSON."""
    metrics_path = "artifacts/metrics/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "accuracy" in data, "metrics.json missing accuracy field"
