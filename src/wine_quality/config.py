from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]
    experiment_name_basic: Optional[str]
    experiment_name_custom: Optional[str]
    experiment_name_fe: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from yaml"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class Tags(BaseModel):
    git_sha: str
    branch: str