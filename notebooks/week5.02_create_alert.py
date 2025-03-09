# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of MAE being higher than 7000

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
SELECT
  (COUNT(CASE WHEN wasserstein_distance > 0.1 THEN 1 END) * 100.0 / COUNT(CASE WHEN wasserstein_distance IS NOT NULL AND NOT isnan(wasserstein_distance) THEN 1 END)) AS percentage_higher_than_70000
FROM ws_mlops.wine_schema.model_monitoring_drift_metrics"""


query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"wine-quality-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on wine quality model for Wasserstein Distance (data drift)",
        query_text=alert_query,
    )
)

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="features_shifted_over_pt1")),
            op=sql.AlertOperator.GREATER_THAN_OR_EQUAL,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=80)),
        ),
        display_name=f"wine-quality-wasserstein-alert-{time.time_ns()}",
        query_id=query.id,
    )
)


# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
