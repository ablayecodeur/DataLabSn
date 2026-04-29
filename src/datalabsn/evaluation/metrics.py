from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from datalabsn.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Calcule les métriques d'évaluation selon la tâche."""

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        average: str = "weighted",
    ) -> dict[str, Any]:
        n_classes = len(np.unique(y_true))
        metrics: dict[str, Any] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                if n_classes == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba[:, 1]), 4)
                else:
                    metrics["roc_auc"] = round(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 4
                    )
            except Exception:
                pass

        logger.info(
            "Évaluation classification — Accuracy: %.4f | F1: %.4f",
            metrics["accuracy"], metrics["f1_score"],
        )
        return metrics

    def evaluate_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            "r2_score": round(r2_score(y_true, y_pred), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "mse": round(mse, 4),
            "rmse": round(np.sqrt(mse), 4),
            "mape": round(
                np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100, 2
            ),
        }
        logger.info(
            "Évaluation régression — R²: %.4f | RMSE: %.4f | MAE: %.4f",
            metrics["r2_score"], metrics["rmse"], metrics["mae"],
        )
        return metrics

    def evaluate_clustering(
        self, X: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics: dict[str, float] = {"n_clusters": n_clusters}
        if n_clusters > 1:
            try:
                metrics["silhouette_score"] = round(silhouette_score(X, labels), 4)
            except Exception:
                pass
        logger.info("Évaluation clustering — Clusters: %d", n_clusters)
        return metrics

    def learning_curve_data(
        self, model, X: np.ndarray, y: np.ndarray, task: str = "classification"
    ) -> pd.DataFrame:
        from sklearn.model_selection import learning_curve

        scoring = "accuracy" if task == "classification" else "r2"
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=5, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        return pd.DataFrame({
            "train_size": train_sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        })
