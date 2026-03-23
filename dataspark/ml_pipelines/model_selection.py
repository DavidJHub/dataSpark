"""
Automated Model Selection
=========================
Compare multiple models via cross-validation and hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from loguru import logger

from dataspark.ml_pipelines.pipeline_builder import PipelineBuilder, CLASSIFIERS, REGRESSORS


class ModelSelector:
    """Compare and select the best model from a catalog."""

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        cv: int = 5,
        scoring: str | None = None,
    ) -> None:
        self.task = task
        self.cv = cv
        self.scoring = scoring or ("accuracy" if task == "classification" else "r2")

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Cross-validate all candidate models and return ranked results."""
        builder = PipelineBuilder(task=self.task)
        catalog = models or (CLASSIFIERS if self.task == "classification" else REGRESSORS)
        rows = []
        for name, estimator in catalog.items():
            pipe = builder.build(X, model=estimator)
            cv_results = cross_validate(
                pipe, X, y, cv=self.cv, scoring=self.scoring, return_train_score=True
            )
            rows.append({
                "model": name,
                "test_score_mean": cv_results["test_score"].mean(),
                "test_score_std": cv_results["test_score"].std(),
                "train_score_mean": cv_results["train_score"].mean(),
                "fit_time_mean": cv_results["fit_time"].mean(),
                "overfit_gap": (
                    cv_results["train_score"].mean() - cv_results["test_score"].mean()
                ),
            })
        result = pd.DataFrame(rows).sort_values("test_score_mean", ascending=False)
        logger.info("Model comparison complete — best: {}", result.iloc[0]["model"])
        return result.reset_index(drop=True)

    def hyperparameter_search(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict,
        method: Literal["grid", "random"] = "random",
        n_iter: int = 50,
    ) -> dict:
        """Grid or randomized search over hyperparameters."""
        if method == "grid":
            search = GridSearchCV(
                pipeline, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=n_iter, cv=self.cv,
                scoring=self.scoring, n_jobs=-1, random_state=42
            )
        search.fit(X, y)
        logger.info("Best score: {:.4f} with params: {}", search.best_score_, search.best_params_)
        return {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "best_estimator": search.best_estimator_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }
