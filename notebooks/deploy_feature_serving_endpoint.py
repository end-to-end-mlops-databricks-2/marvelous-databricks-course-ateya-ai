# Databricks notebook source
# MAGIC %pip install  /Volumes/mlops_dev/ateyatec/packages/wine_quality-0.0.3-py3-none-any.whl 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time
import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from databricks import feature_engineering
import pandas as pd

from wine_quality.config import ProjectConfig
from wine_quality.serving.feature_serving import FeatureServing

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.wine_qualities_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = "wine-qualities-feature-serving"

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
df = pd.concat([train_set, test_set])

model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.wine_quality_model_basic@latest-model")


# COMMAND ----------

preds_df = df[["Id","fixed_acidity", "citric_acid", "volatile_acidity"]]
preds_df["Predicted_WineQuality"] = model.predict(df[config.cat_features + config.num_features])
preds_df = spark.createDataFrame(preds_df)

fe.create_table(
    name=feature_table_name, primary_keys=["Id"], df=preds_df, description="Wine Quality predictions feature table"
)

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)


# COMMAND ----------

# Create online table
feature_serving.create_online_table()

# COMMAND ----------

# Load data and model
feature_manager.load_data()
feature_manager.load_model()

# COMMAND ----------

# Create feature table and enable Change Data Feed
feature_manager.create_feature_table()

# COMMAND ----------

# Create online table
feature_manager.create_online_table()

# COMMAND ----------

# Create feature spec
feature_manager.create_feature_spec()

# COMMAND ----------

# Deploy feature serving endpoint
feature_manager.deploy_serving_endpoint()

# COMMAND ----------

# Test feature serving
os.environ["TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

entity_id = "182"
status_code, response_text = feature_manager.call_endpoint(entity_id)
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Run a load test with 10 requests
average_latency = feature_manager.load_test(num_requests=10)
print(f"Average Latency per Request: {average_latency} seconds")
