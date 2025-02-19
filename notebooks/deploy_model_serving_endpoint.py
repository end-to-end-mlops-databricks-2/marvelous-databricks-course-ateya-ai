# Databricks notebook source
# MAGIC %pip install  /Volumes/mlops_dev/ateyatec/packages/wine_quality-0.0.3-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time
from typing import Dict, List


import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
endpoint_name = "wine-quality-model-serving"
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.wine_quality_model_basic", endpoint_name="wine-quality-model-serving"
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body
required_columns = [
    "type",
    "alcohol",
    "density",
    "volatile_acidity",
    "chlorides",
    "residual_sugar",
    "free_sulfur_dioxide",
    "pH",
    "total_sulfur_dioxide",
    "sulphates",
    "citric_acid",
    "fixed_acidity",
]

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each body should be list of json with columns

[
{'type': 'red',
  'alcohol': 9.1,
  'density': 0.997,
  'volatile_acidity': 0.65,
  'chlorides': 0.07,
  'residual_sugar': 1.8,
  'free_sulfur_dioxide': 13,
  'pH': 3.44,
  'total_sulfur_dioxide': 40,
  'sulphates': 0.6,
  'citric_acid': 0.18,
  'fixed_acidity': 7.1,
  'quality': 0}
  },

{'type': 'white',
  'alcohol': 11.9,
  'density': 0.99064,
  'volatile_acidity': 0.25,
  'chlorides': 0.045,
  'residual_sugar': 2.3,
  'free_sulfur_dioxide': 40,
  'pH': 3.16,
  'total_sulfur_dioxide': 118,
  'sulphates': 0.48,
  'citric_acid': 0.45,
  'fixed_acidity': 7,
  'quality': 1}
  }
  ]
"""


def call_endpoint(record: List[Dict]):  # type: ignore
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# "load test"

for i in range(len(dataframe_records)):
    status_code, response_text=call_endpoint(dataframe_records[i])
    time.sleep(0.2)
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")

# COMMAND ----------


