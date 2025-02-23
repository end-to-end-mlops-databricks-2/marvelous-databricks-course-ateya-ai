import time

import numpy as np
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

    def preprocess(self):
        """Preprocess the dataframe stored in self.df"""

        # If self.df is not a DataFrame, stop execution
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("self.df is not a DataFrame! Check data loading.")

        # Handle missing values and convert data types as needed
        self.df["fixed_acidity"] = pd.to_numeric(self.df["fixed_acidity"], errors="coerce")

        # Let's fill missing values with mean or default values
        self.df["alcohol"] = self.df["alcohol"].fillna(self.df["alcohol"].mean())

        # Handle numeric features
        num_features = self.config.num_features
        missing_cols = []
        for col in num_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            else:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")

        # Fill missing values with mean or default values
        self.df = self.df.fillna(
            {
                "citric_acid": self.df["citric_acid"].mean() if "citric_acid" in self.df.columns else 0,
                "sulphates": 0,
            }
        )

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        missing_cat_cols = []
        for cat_col in cat_features:
            if cat_col in self.df.columns:
                self.df[cat_col] = self.df[cat_col].astype("category")
            else:
                missing_cat_cols.append(cat_col)

        if missing_cat_cols:
            raise ValueError(f"Columns {missing_cat_cols} not found in DataFrame")

        # Let's extract the target variable and relevant features
        target = self.config.target
        self.df["Id"] = range(1, self.df.shape[0] + 1)
        relevant_features = cat_features + num_features + [target] + ["Id"]
        self.df = self.df[relevant_features]
        self.df["Id"] = self.df["Id"].astype(str)

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into train and test sets"""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

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


def generate_synthetic_data(df, num_rows=10):
    """Generates synthetic data based on the distribution of the input DataFrame."""
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)
            except ValueError as err:
                raise ValueError(f"Cannot generate synthetic data for column {column}.") from err
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            try:
                synthetic_data[column] = np.random.choice(
                    df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
                )
            except ValueError as err:
                raise ValueError(f"Cannot generate synthetic data for column {column}.") from err

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    float_columns = {
        "alcohol",
        "density",
        "volatile_acidity",
        "chlorides",
        "residual_sugar",
        "free_sulfur_dioxide",
        "pH",
        "total_sulfur_dioxide",
        "citric_acid",
        "fixed_acidity",
        "sulphates",
    }
    for col in float_columns.intersection(df.columns):
        synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors="coerce")
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    synthetic_data["quality"] = synthetic_data["quality"].astype(np.int32)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    return synthetic_data
