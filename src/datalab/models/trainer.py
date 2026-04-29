from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from datalab.utils import get_logger
from .registry import ModelRegistry

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model: Any
    model_name: str
    task: str
    train_score: float
    test_score: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    best_params: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    training_time: float = 0.0
    X_test: Any = None
    y_test: Any = None
    y_pred: Any = None
    y_proba: Any = None


class ModelTrainer:
    """Entraîne, valide et optimise les modèles."""

    def __init__(self, random_state: int = 42, test_size: float = 0.2, cv_folds: int = 5):
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        task: str = "classification",
        tune_hyperparams: bool = False,
        search_strategy: str = "grid",
        n_iter: int = 20,
    ) -> TrainingResult:
        # Aligner X et y sur l'index commun (drop_duplicates peut réduire X)
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        stratify = y_arr if task == "classification" and len(np.unique(y_arr)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        model = ModelRegistry.get(task, model_name)
        info = ModelRegistry.get_info(task, model_name)

        if tune_hyperparams and info.get("tuning"):
            model = self._tune(model, info["tuning"], X_train, y_train, task, search_strategy, n_iter)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state) \
            if task == "classification" else self.cv_folds
        scoring = "accuracy" if task == "classification" else "r2"
        cv_scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1)

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        logger.info(
            "[%s] %s — train=%.4f test=%.4f cv=%.4f±%.4f (%.2fs)",
            task, model_name, train_score, test_score,
            cv_scores.mean(), cv_scores.std(), training_time,
        )

        return TrainingResult(
            model=model,
            model_name=model_name,
            task=task,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            best_params=getattr(model, "best_params_", {}),
            feature_names=list(X.columns) if isinstance(X, pd.DataFrame) else [],
            training_time=training_time,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification",
        model_names: list[str] | None = None,
    ) -> pd.DataFrame:
        names = model_names or ModelRegistry.list_models(task)
        results = []
        for name in names:
            try:
                r = self.train(X, y, name, task)
                results.append({
                    "Modèle": name,
                    "Score Train": round(r.train_score, 4),
                    "Score Test": round(r.test_score, 4),
                    "CV Moyen": round(r.cv_mean, 4),
                    "CV Std": round(r.cv_std, 4),
                    "Temps (s)": round(r.training_time, 3),
                    "Surapprentissage": round(r.train_score - r.test_score, 4),
                })
            except Exception as e:
                logger.warning("Échec pour %s : %s", name, e)
        df = pd.DataFrame(results)
        return df.sort_values("CV Moyen", ascending=False).reset_index(drop=True)

    def _tune(self, model, param_grid, X_train, y_train, task, strategy, n_iter):
        scoring = "accuracy" if task == "classification" else "r2"
        cv = min(self.cv_folds, 3)

        if strategy == "random":
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, random_state=self.random_state,
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv,
                scoring=scoring, n_jobs=-1,
            )

        search.fit(X_train, y_train)
        logger.info("Meilleurs paramètres : %s", search.best_params_)
        return search.best_estimator_
