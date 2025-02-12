import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifer
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wine_quality.config import ProjectConfig, Tags


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.wine_quality_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_total_acidity_index"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self):
        """
        Create or replace the wine_quality_features table and populate it.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL, (fixed_acidity DOUBLE, citric_acid DOUBLE, volatile_acidity DOUBLE);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT wine_quality_pk PRIMARY KEY(Id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, fixed_acidity, citric_acid, volatile_acidity FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, fixed_acidity, citric_acid, volatile_acidity FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """
        Define a function to calculate the wine's acidity balance (Total Acidiy Index).
        Total Acidity Index = fixed_acidity + citric_acid - volatile_acidity
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(fixed_acidity DOUBLE, citric_acid DOUBLE, volatile_acidity DOUBLE)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        if fixed_acidity < 0 or citric_acid < 0 or volatile_acidity < 0:
            raise ValueError("All input values must be non-negative.")
        return fixed_acidity + citric_acid - volatile_acidity
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "fixed_acidity", "citric_acid", "volatile_acidity"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = (
            self.train_set.withColumn("fixed_acidity", self.train_set["fixed_acidity"].cast("DOUBLE"))
            .withColumn("citric_acid", self.train_set["citric_acid"].cast("DOUBLE"))
            .withColumn("volatile_acidity", self.train_set["volatile_acidity"].cast("DOUBLE"))
        )
        self.train_set = self.train_set.withColumn("Id", self.train_set["Id"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["fixed_acidity", "citric_acid", "volatile_acidity"],
                    lookup_key="Id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="total_acidity_index",
                    input_bindings={
                        "fixed_acidity": "fixed_acidity",
                        "citric_acid": "citric_acid",
                        "volatile_acidity": "volatile_acidity",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["total_acidity_index"] = (
            self.test_set["fixed_acidity"] + self.test_set["citric_acid"] - self.test_set["volatile_acidity"]
        )

        self.X_train = self.training_df[self.num_features + self.cat_features + ["total_acidity_index"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["total_acidity_index"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """
        Train the model and log results to MLflow.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifer(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)
        mlflow.autolog(disable=True)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            logger.info(f"📊 Accuracy Score: {accuracy}")
            logger.info(f"📊 ROC AUC SCORE: {roc_auc}")
            logger.info(f"📊 Classification Report: {report}")
            logger.info(f"📊 Precision Score: {precision}")
            logger.info(f"📊 Recall Score: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM Clasisifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("Accuracy Score", accuracy)
            mlflow.log_metric("ROC AUC SCORE", roc_auc)
            mlflow.log_metric("Classification Report:", report)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("f1_score", f1)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.wine_quality_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.wine_quality_model_fe",
            alias="latest-model",
            version=latest_version,
        )

    def load_latest_model_and_predict(self, X):
        """
        Load the trained model from MLflow using Feature Engineering Client and make predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.wine_quality_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
