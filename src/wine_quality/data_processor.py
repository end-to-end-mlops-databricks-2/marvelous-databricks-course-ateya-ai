import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from wine_quality.config import ProcessingConfig
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, spark: SparkSession,df: pd.DataFrame, config: ProcessingConfig):
        self.spark = spark
        self.df = df 
        self.config = config

    def process(self):
        """Preprocess the dataframe stored in self.df"""
        # Handle spaces in the column names
        self.df.colums = [col.replace(" ", "_") for col in self.df.columns]

        # Handle numeric variables
        numeric_cols = self.config.numeric_cols
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Let's fill missing values with mean or default values
        self.df["alcohol"] = self.df["alcohol"].fillna(self.df["alcohol"].mean(), inplace=True)

        self.df.fillna({"citric_acid": self.df["citric_acid"].mean()}, {"sulphates":0}, inplace=True)

        # Let's extract the target variable and relevant features
        target = self.config.target
        self.df["Id"] = range(1, self.df.shape[0] + 1)
        relevant_features = numeric_cols + target + ["Id"]
        self.df = self.df[relevant_features]
        self.df["Id"] = self.df["Id"].astype(str)

        def split_data(self, test_size=0.2, random_state=42):
            """Split the data into train and test sets"""   
            train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
            return train_set, test_set
        
        def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame): 
            """Saves the train and test sets to databricks catalog table."""
            train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

            test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )

            train_set_with_timestamp.write.mode("append").saveAsTable(
                f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
            )

            test_set_with_timestamp.write.mode("append").saveAsTable(
                f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
            )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )



        



