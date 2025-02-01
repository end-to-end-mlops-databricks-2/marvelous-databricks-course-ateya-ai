import os
import re
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from titanic_survival_model.config.core import DATASET_DIR, ProjectConfig, project_config


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, file_name: str, config: ProjectConfig):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.file_name = file_name

    def get_first_cabin(self, record: Any):
        try:
            return record.split()[0]
        except ArithmeticError:
            return np.nan

    def get_title(self, passenger: str) -> str:
        """Extracts the title (Mr, Ms, etc) from the name variable."""
        line = passenger
        if re.search("Mrs", line):
            return "Mrs"
        elif re.search("Mr", line):
            return "Mr"
        elif re.search("Miss", line):
            return "Miss"
        elif re.search("Master", line):
            return "Master"
        else:
            return "Other"

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # replace question marks with NaN values
        self.df.columns = [col.strip() for col in self.df.columns]
        self.df = self.df.replace("?", np.nan)

        # retain only the first cabin if more than
        # 1 are available per passenger
        self.df["cabin"] = self.df["cabin"].apply(self.get_first_cabin)

        self.df["title"] = self.df["name"].apply(self.get_title)

        # cast numerical variables as floats
        self.df["fare"] = self.df["fare"].astype("float")
        self.df["age"] = self.df["age"].astype("float")

        # drop unnecessary variables
        self.df.drop(labels=self.config.features_to_remove, axis=1, inplace=True)

        return self.df

    def load_dataset(self):
        filepath = os.path.join(DATASET_DIR, self.file_name)
        if not os.path.isfile(self.file_name):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        dataframe = pd.read_csv(self.file_name)
        transformed_df = self.preprocess(dataframe)

        return transformed_df

    # Separate X and y
    def separate_data(self):
        X = self.df.drop(project_config.target, axis=1)
        y = self.df[project_config.target]
        return X, y

    def split_data(self):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(
            self.df, test_size=project_config.test_data_file, random_state=project_config.random_state
        )
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
