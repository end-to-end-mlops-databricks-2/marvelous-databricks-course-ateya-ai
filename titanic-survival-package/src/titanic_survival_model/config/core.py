import os
from pathlib import Path
from typing import Any, Dict, List

import yaml  # type: ignore
from pydantic import BaseModel  # type: ignore

import titanic_survival_model

PACKAGE_ROOT = Path(titanic_survival_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
DATASET_DIR = os.path.join(PACKAGE_ROOT, "data")
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_models")
CONFIG_FILE_PATH = os.path.join(PACKAGE_ROOT, "config.yml")


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    unused_features: List[str]
    cabin_feature: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters
    pipeline_id: str  # pipeline id for data live tables
    training_data_file: str  # path to training data
    test_data_file: str  # path to test data
    raw_data_file: str  # path to raw data
    pipeline_name: str
    pipeline_save_file: str
    model_name: str
    package_name: str
    test_size: float
    random_state: int
    features_to_remove: List[str]


def check_file_exists(config_path: Path = None):
    """Check if the training data file exists."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training data file not found at {config_path}")
    return config_path


def load_conf_from_yaml(config_path: Path = None):
    """Load configuration from a YAML file."""
    if check_file_exists(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return ProjectConfig(**config_dict)
    raise FileNotFoundError(f"Configuration file not found at {config_path}")


project_config = load_conf_from_yaml(CONFIG_FILE_PATH)
