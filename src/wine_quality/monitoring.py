import logging

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

logger = logging.getLogger(__name__)


def create_or_refresh_monitoring(config, spark, workspace):
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`wine-quality-model-serving-fe-dev_payload`")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("alcohol", DoubleType(), True),
                            StructField("density", DoubleType(), True),
                            StructField("volatile_acidity", DoubleType(), True),
                            StructField("chlorides", DoubleType(), True),
                            StructField("residual_sugar", DoubleType(), True),
                            StructField("free_sulfur_dioxide", DoubleType(), True),
                            StructField("pH", IntegerType(), True),
                            StructField("total_sulfur_dioxide", DoubleType(), True),
                            StructField("citric_acid", DoubleType(), True),
                            StructField("fixed_acidity", DoubleType(), True),
                            StructField("sulphates", StringType(), True),
                            StructField("Id", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.Id").alias("Id"),
        F.col("record.alcohol").alias("alcohol"),
        F.col("record.density").alias("density"),
        F.col("record.volatile_acidity").alias("volatile_acidity"),
        F.col("record.chlorides").alias("chlorides"),
        F.col("record.residual_sugar").alias("residual_sugar"),
        F.col("record.free_sulfur_dioxide").alias("free_sulfur_dioxide"),
        F.col("record.pH").alias("pH"),
        F.col("record.total_sulfur_dioxide").alias("total_sulfur_dioxide"),
        F.col("record.citric_acid").alias("citric_acid"),
        F.col("record.fixed_acidity").alias("fixed_acidity"),
        F.col("record.sulphates").alias("sulphates"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("wine_quality_model_fe_dab").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(test_set.select("Id", "quality"), on="Id", how="left")
        .withColumnRenamed("quality", "wine_quality_test")
        .join(inference_set_skewed.select("Id", "quality"), on="Id", how="left")
        .withColumnRenamed("quality", "wine_quality_inference")
        .select("*", F.coalesce(F.col("wine_quality_test"), F.col("wine_quality_inference")).alias("wine_quality"))
        .drop("wine_quality_test", "wine_quality_inference")
        .withColumn("wine_quality", F.col("wine_quality").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["wine_quality", "prediction"])
    )

    house_features = spark.table(f"{config.catalog_name}.{config.schema_name}.wine_quality_features")

    df_final_with_features = df_final_with_status.join(house_features, on="Id", how="left")

    df_final_with_features = df_final_with_features.withColumn("fixed_acidity", F.col("fixed_acidity").cast("double"))

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="quality",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
