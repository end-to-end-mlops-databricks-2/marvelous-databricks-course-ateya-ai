import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.serving.fe_model_serving import FeatureLookupServing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")


# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
# config = ProjectConfig.from_yaml(config_path=config_path, env="dev")
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "wine-quality-model-serving-fe-{args.env}"
# endpoint_name = "wine-quality-model-serving-fe-dev"

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name="wine_quality_model_fe_dab",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.wine_quality_features_dab",
)


# Create the online table for house features
feature_model_server.create_online_table()
logger.info("Created online table")

# feature_model_server.update_online_table(config=config)
# logger.info("Updated online table")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")
