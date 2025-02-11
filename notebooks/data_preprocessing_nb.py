# Databricks notebook source
# %pip install /Volumes/mlops_dev/ateyatec/packages/wine_quality-0.0.1-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import logging

import pandas as pd
import yaml
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# print("Configuration loaded:")
# print(yaml.dump(config, default_flow_style=False))

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Initialize DataProcessor
# file_path = "/Volumes/mlops_dev/ateyatec/wine_quality_data/processed_data.csv"

filepath = "../data/processed_data.csv"
# Load the data
pandas_df = pd.read_csv(filepath)

spark = SparkSession.builder.getOrCreate()

# spark_df = spark.read.csv(
#     f"/Volumes/{config.catalog_name}/{config.schema_name}/data/processed_data.csv", header=True, inferSchema=True
# ).toPandas()
# Initialize DataProcessor
data_processor = DataProcessor(spark=spark, config=config, df=pandas_df)

# Preprocess the data
data_processor.process()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

if "spark" not in locals():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

data_processor.save_to_catalog(X_train, X_test)

# COMMAND ----------
