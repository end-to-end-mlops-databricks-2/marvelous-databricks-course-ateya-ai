import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split  # type: ignore

from wine_quality.config import ProjectConfig


class DataProcessor:
    def __init__(self, spark: SparkSession, df: pd.DataFrame, config: ProjectConfig):
        self.spark = spark
        self.df = df
        self.config = config

    def process(self):
        """Preprocess the dataframe stored in self.df"""

        # If self.df is not a DataFrame, stop execution
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("self.df is not a DataFrame! Check data loading.")

        # Handle spaces in the column names
        self.df.columns = [col.replace(" ", "_") for col in self.df.columns]

        # Handle numeric variables
        numeric_cols = self.config.num_features
        self.df[numeric_cols] = self.df[self.config.num_features].apply(pd.to_numeric, errors="coerce")

        # Let's fill missing values with mean or default values
        self.df["alcohol"] = self.df["alcohol"].fillna(self.df["alcohol"].mean())

        self.df = self.df.fillna({"citric_acid": self.df["citric_acid"].mean(), "sulphates": 0})

        # Let's extract the target variable and relevant features
        target = self.config.target
        self.df["Id"] = range(1, self.df.shape[0] + 1)
        relevant_features = numeric_cols + [target] + ["Id"]
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

        # train_set_with_timestamp.write.mode("append").saveAsTable(
        #     f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        # )

        # test_set_with_timestamp.write.mode("append").saveAsTable(
        #     f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        # )
        train_set_with_timestamp.write.mode("append").saveAsTable("train_set")

        test_set_with_timestamp.write.mode("append").saveAsTable("test_set")

    def enable_change_data_feed(self):
        # self.spark.sql(
        #     f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
        #     "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        # )

        # self.spark.sql(
        #     f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
        #     "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        # )

        self.spark.sql("ALTER TABLE train_set" "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql("ALTER TABLE test_set" "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
