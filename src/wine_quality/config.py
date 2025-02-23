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
    parameters_a: Optional[Dict[str, Any]]  # for AB testing
    parameters_b: Optional[Dict[str, Any]]  # for AB testing
    experiment_name_basic: Optional[str]
    experiment_name_custom: Optional[str]
    experiment_name_fe: Optional[str]
    pipeline_id: Optional[str]
    # ab_test: Dict[str, Any]  # type: ignore # Dictionary to hold A/B test parameters

    @classmethod
    def from_yaml(cls, config_path: str, env: str = None):
        """Load configuration from yaml"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if env is not None:
            config["catalog_name"] = config[env]["catalog_name"]
            config["schema_name"] = config[env]["schema_name"]
            config["pipeline_id"] = config[env]["pipeline_id"]
        else:
            config["catalog_name"] = config["catalog_name"]
            config["schema_name"] = config["schema_name"]
            config["pipeline_id"] = config["pipeline_id"]

        return cls(**config)


class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: str
