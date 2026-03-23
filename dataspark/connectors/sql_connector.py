"""
SQL Connector
=============
Read/write DataFrames from/to SQL databases via SQLAlchemy.
Supports chunked reads for large-scale data.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger


class SQLConnector:
    """Manage SQL database connections and data transfer."""

    def __init__(self, connection_string: str) -> None:
        self.engine = create_engine(connection_string)
        logger.info("SQLConnector created for {}", self.engine.url.database)

    @contextmanager
    def connection(self) -> Generator:
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def read_query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame."""
        with self.connection() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
        logger.info("Query returned {} rows", len(df))
        return df

    def read_table(self, table_name: str, schema: str | None = None) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine, schema=schema)

    def read_chunked(
        self, query: str, chunksize: int = 10_000
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream large query results as DataFrame chunks."""
        with self.connection() as conn:
            for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):
                yield chunk

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        schema: str | None = None,
        chunksize: int = 5_000,
    ) -> None:
        """Write a DataFrame to a SQL table."""
        df.to_sql(
            table_name,
            self.engine,
            if_exists=if_exists,
            schema=schema,
            index=False,
            chunksize=chunksize,
        )
        logger.info("Wrote {} rows to {}", len(df), table_name)

    def execute(self, statement: str) -> None:
        """Execute a DDL/DML statement."""
        with self.connection() as conn:
            conn.execute(text(statement))
            conn.commit()
