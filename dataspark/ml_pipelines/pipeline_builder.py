"""
Scikit-Learn Pipeline Builder
=============================
Composable ML pipelines with preprocessing, feature selection,
and model training in a single reproducible object.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC, SVR
from loguru import logger


CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "svm": SVC(probability=True),
}

REGRESSORS = {
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elastic_net": ElasticNet(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "svr": SVR(),
}


class PipelineBuilder:
    """Build end-to-end sklearn Pipelines with automatic preprocessing."""

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        scaler: Literal["standard", "robust"] = "standard",
        impute_strategy: str = "median",
    ) -> None:
        self.task = task
        self.scaler_cls = StandardScaler if scaler == "standard" else RobustScaler
        self.impute_strategy = impute_strategy

    def build(
        self,
        X: pd.DataFrame,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> Pipeline:
        """Build a Pipeline with preprocessing + estimator."""
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=self.impute_strategy)),
            ("scaler", self.scaler_cls()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ])

        if model is not None:
            estimator = model
        elif model_name:
            catalog = CLASSIFIERS if self.task == "classification" else REGRESSORS
            estimator = catalog[model_name]
        else:
            estimator = (
                CLASSIFIERS["random_forest"]
                if self.task == "classification"
                else REGRESSORS["random_forest"]
            )

        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        logger.info("Built pipeline: task={}, model={}", self.task, type(estimator).__name__)
        return pipe

    def cross_validate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str | list[str] | None = None,
    ) -> dict:
        """Run cross-validation and return scores."""
        if scoring is None:
            scoring = (
                ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
                if self.task == "classification"
                else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
            )
        results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
        summary = {}
        for key, vals in results.items():
            summary[key + "_mean"] = float(np.mean(vals))
            summary[key + "_std"] = float(np.std(vals))
        logger.info("CV complete — {} folds", cv)
        return summary
