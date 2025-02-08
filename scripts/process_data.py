import logging

import yaml
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")


logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

# file_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/wine_quality_data/red_white_wines_combined.csv"
file_path = "../data/red_white_wines_combined.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
