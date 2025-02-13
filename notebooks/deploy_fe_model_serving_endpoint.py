# Databricks notebook source
# Databricks notebook source
import os

from loguru import logger
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.serving.fe_model_serving import FeatureLookupServing

# Load project config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(config)

# Create the online table for house features
feature_model_server.create_online_table()

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "LotFrontage",
    "LotArea",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "TotalBsmtSF",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Neighborhood",
    "Condition1",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "SaleType",
    "SaleCondition",
    "Id",
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])

# COMMAND ----------
# Call the endpoint with one sample record
os.environ["TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

status_code, response_text = feature_model_server.call_endpoint(sampled_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

