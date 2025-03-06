# Databricks notebook source
# MAGIC %pip install  /Volumes/mlops_dev/ateyatec/packages/wine_quality-0.0.3-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import os
import time
from typing import Dict, List

import requests
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.serving.fe_model_serving import FeatureLookupServing

spark = SparkSession.builder.getOrCreate()

# Load project config
config = '/Volumes/ws_mlops/wine_schema/data/project_config.yml'
catalog_name = 'ws_mlops'
schema_name = 'wine_schema'
endpoint_name = "wine-qualities-model-serving-fe"


# COMMAND ----------

# Create a sample request body
required_columns = [
    "type",
    "alcohol",
    "density",
    "chlorides",
    "residual_sugar",
    "free_sulfur_dioxide",
    "pH",
    "total_sulfur_dioxide",
    "sulphates",
    "Id",
]

spark = SparkSession.builder.getOrCreate()

# Use Databricks notebook context to get the API token
dbutils = DBUtils(spark)

train_set = spark.table("ws_mlops.wine_schema.train_set_dab").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]


# COMMAND ----------


def call_endpoint(record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = "https://adb-3396447954555637.17.azuredatabricks.net/serving-endpoints/wine-quality-model-serving-fe-dev/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# COMMAND ----------

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
status_code, response_text = call_endpoint(dataframe_records[0])

print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# "load test"

for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    time.sleep(0.2)
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")

# COMMAND ----------
