"""
PySpark Connector
=================
Create Spark sessions and bridge between PySpark and Pandas DataFrames.
Supports reading from Hadoop/HDFS, Parquet, CSV, and Delta tables.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class SparkConnector:
    """Manage PySpark sessions and data I/O."""

    def __init__(
        self,
        app_name: str = "DataSpark",
        master: str = "local[*]",
        config: dict[str, str] | None = None,
    ) -> None:
        self.app_name = app_name
        self.master = master
        self.extra_config = config or {}
        self._spark = None

    @property
    def spark(self):
        """Lazy-initialized SparkSession."""
        if self._spark is None:
            self._spark = self._create_session()
        return self._spark

    def _create_session(self):
        from pyspark.sql import SparkSession

        builder = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.master)
        )
        for k, v in self.extra_config.items():
            builder = builder.config(k, v)
        session = builder.getOrCreate()
        logger.info("Spark session created — {}", self.app_name)
        return session

    def read_csv(self, path: str, header: bool = True, infer_schema: bool = True):
        """Read CSV into a Spark DataFrame."""
        return self.spark.read.csv(path, header=header, inferSchema=infer_schema)

    def read_parquet(self, path: str):
        return self.spark.read.parquet(path)

    def read_jdbc(self, url: str, table: str, properties: dict[str, str]):
        """Read from a JDBC source (SQL databases via Spark)."""
        return self.spark.read.jdbc(url, table, properties=properties)

    def to_pandas(self, spark_df) -> pd.DataFrame:
        """Convert Spark DataFrame to Pandas."""
        return spark_df.toPandas()

    def from_pandas(self, df: pd.DataFrame):
        """Convert Pandas DataFrame to Spark."""
        return self.spark.createDataFrame(df)

    def sql(self, query: str):
        """Run a Spark SQL query."""
        return self.spark.sql(query)

    def write_parquet(self, spark_df, path: str, mode: str = "overwrite") -> None:
        spark_df.write.mode(mode).parquet(path)
        logger.info("Wrote Spark DF to parquet: {}", path)

    def stop(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            logger.info("Spark session stopped")
