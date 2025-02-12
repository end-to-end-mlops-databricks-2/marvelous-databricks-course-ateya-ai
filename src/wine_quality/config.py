from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from yaml"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
