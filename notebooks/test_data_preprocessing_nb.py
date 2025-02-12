# Databricks notebook source
# MAGIC %md
# MAGIC # Wine Quality Prediction Exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict wine quality. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# MAGIC
# MAGIC ## Importing Required Libraries
# MAGIC
# MAGIC First, let's import all the necessary libraries.

# COMMAND ----------

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Only works in a Databricks environment if the data is there
# file_path = "/Volumes/mlops_dev/ateyatec/data/data.csv"
# Load the data
# df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
file_path = "../data/data.csv"
# Load the data
df = pd.read_csv(file_path)
df.head(5)

# COMMAND ----------

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print(config.get("catalog_name"))
num_features = config.get("num_features")
print(num_features)


# COMMAND ----------

## Preprocessing
# Handle spaces in the column names
df.columns = [col.replace(" ", "_") for col in df.columns]
df.head(5)

# COMMAND ----------

# Handle missing values and convert data types as needed
df["fixed_acidity"] = pd.to_numeric(df["fixed_acidity"], errors="coerce")

# Let's fill missing values with mean or default values
df["alcohol"] = df["alcohol"].fillna(df["alcohol"].mean())

# Handle numeric features
num_features = config.get("num_features", [])
missing_cols = []
for col in num_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        missing_cols.append(col)

if missing_cols:
    raise ValueError(f"Columns {missing_cols} not found in DataFrame")

# Fill missing values with mean or default values
df = df.fillna(
    {
        "citric_acid": df["citric_acid"].mean() if "citric_acid" in df.columns else 0,
        "sulphates": 0,
    }
)

# Convert categorical features to the appropriate type
cat_features = config.get("cat_features", [])
missing_cat_cols = []
for cat_col in cat_features:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype("category")
    else:
        missing_cat_cols.append(cat_col)

if missing_cat_cols:
    raise ValueError(f"Columns {missing_cat_cols} not found in DataFrame")

# Extract target and relevant features
target = config.get("target")

df["Id"] = range(1, df.shape[0] + 1)
relevant_columns = [col for col in cat_features + num_features + [target] + ["Id"] if col in df.columns]
print(relevant_columns)

df = df[relevant_columns]
df["Id"] = df["Id"].astype("str")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# display(df.head(5))
df.head(5)

# COMMAND ----------
