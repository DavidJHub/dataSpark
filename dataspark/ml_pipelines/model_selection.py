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
        cv: int | None = None,
    ) -> pd.DataFrame:
        """Cross-validate all candidate models and return ranked results."""
        cv_folds = cv if cv is not None else self.cv
        builder = PipelineBuilder(task=self.task)
        catalog = models or (CLASSIFIERS if self.task == "classification" else REGRESSORS)
        rows = []
        for name, estimator in catalog.items():
            pipe = builder.build(X, model=estimator)
            cv_results = cross_validate(
                pipe, X, y, cv=cv_folds, scoring=self.scoring, return_train_score=True
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
        X_or_pipeline: pd.DataFrame | Pipeline,
        y: pd.Series | None = None,
        param_grid: dict | None = None,
        *,
        model_name: str | None = None,
        pipeline: Pipeline | None = None,
        method: Literal["grid", "random"] = "random",
        search_type: str | None = None,
        n_iter: int = 50,
        cv: int | None = None,
    ) -> dict:
        """Grid or randomized search over hyperparameters.

        Can be called as:
        - hyperparameter_search(X, y, model_name="random_forest", param_grid=..., search_type="grid")
        - hyperparameter_search(pipeline, X, y, param_grid, method="grid")  (legacy)
        """
        cv_folds = cv if cv is not None else self.cv
        search_method = search_type if search_type is not None else method

        # Detect calling convention
        if isinstance(X_or_pipeline, Pipeline):
            # Legacy: hyperparameter_search(pipeline, X, y, param_grid)
            pipe = X_or_pipeline
            X_data = y
            y_data = param_grid
            grid = model_name  # positional shift
            if not isinstance(X_data, (pd.DataFrame, np.ndarray)):
                raise TypeError("When passing a Pipeline as first arg, second arg must be X (data)")
        else:
            # New: hyperparameter_search(X, y, model_name=..., param_grid=...)
            X_data = X_or_pipeline
            y_data = y
            grid = param_grid
            if model_name is not None:
                builder = PipelineBuilder(task=self.task)
                pipe = builder.build(X_data, model_name=model_name)
            elif pipeline is not None:
                pipe = pipeline
            else:
                raise ValueError("Must specify either model_name or pipeline")

        if search_method == "grid":
            search = GridSearchCV(
                pipe, grid, cv=cv_folds, scoring=self.scoring, n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                pipe, grid, n_iter=n_iter, cv=cv_folds,
                scoring=self.scoring, n_jobs=-1, random_state=42
            )
        search.fit(X_data, y_data)
        logger.info("Best score: {:.4f} with params: {}", search.best_score_, search.best_params_)
        return {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "best_estimator": search.best_estimator_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }
