# Databricks notebook source
# Databricks notebook source
import os
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

from wine_quality.config import ProjectConfig
from wine_quality.serving.feature_serving import FeatureServing

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
# Initialize feature store manager
feature_manager = FeatureServing(config)

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
