"""Tests du module d'évaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from datalabsn.evaluation import Evaluator


@pytest.fixture
def binary_preds():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], 100)
    y_pred = np.where(rng.random(100) > 0.2, y_true, 1 - y_true)
    y_proba = rng.dirichlet([1, 1], 100)
    return y_true, y_pred, y_proba


@pytest.fixture
def multiclass_preds():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1, 2], 150)
    y_pred = np.where(rng.random(150) > 0.2, y_true, rng.choice([0, 1, 2], 150))
    y_proba = rng.dirichlet([1, 1, 1], 150)
    return y_true, y_pred, y_proba


@pytest.fixture
def regression_preds():
    rng = np.random.default_rng(42)
    y_true = rng.normal(50, 10, 200)
    y_pred = y_true + rng.normal(0, 3, 200)
    return y_true, y_pred


class TestEvaluatorClassification:
    def test_binary_metrics(self, binary_preds):
        y_true, y_pred, y_proba = binary_preds
        ev = Evaluator()
        metrics = ev.evaluate_classification(y_true, y_pred, y_proba)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_multiclass_metrics(self, multiclass_preds):
        y_true, y_pred, y_proba = multiclass_preds
        ev = Evaluator()
        metrics = ev.evaluate_classification(y_true, y_pred, y_proba)
        assert "confusion_matrix" in metrics
        assert metrics["confusion_matrix"].shape == (3, 3)

    def test_without_probas(self, binary_preds):
        y_true, y_pred, _ = binary_preds
        ev = Evaluator()
        metrics = ev.evaluate_classification(y_true, y_pred)
        assert "accuracy" in metrics
        assert "roc_auc" not in metrics

    def test_perfect_classifier(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        ev = Evaluator()
        metrics = ev.evaluate_classification(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0


class TestEvaluatorRegression:
    def test_regression_metrics(self, regression_preds):
        y_true, y_pred = regression_preds
        ev = Evaluator()
        metrics = ev.evaluate_regression(y_true, y_pred)
        assert "r2_score" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert metrics["r2_score"] > 0.5
        assert metrics["rmse"] > 0
        assert metrics["mae"] <= metrics["rmse"]

    def test_perfect_regression(self):
        y = np.arange(100, dtype=float)
        ev = Evaluator()
        metrics = ev.evaluate_regression(y, y)
        assert metrics["r2_score"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)

    def test_rmse_equals_sqrt_mse(self, regression_preds):
        y_true, y_pred = regression_preds
        ev = Evaluator()
        metrics = ev.evaluate_regression(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), rel=1e-4)


class TestEvaluatorClustering:
    def test_clustering_metrics(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        labels = rng.choice([0, 1, 2], 100)
        ev = Evaluator()
        metrics = ev.evaluate_clustering(X, labels)
        assert "n_clusters" in metrics
        assert metrics["n_clusters"] == 3
        assert "silhouette_score" in metrics
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_single_cluster(self):
        X = np.random.default_rng(42).normal(0, 1, (50, 3))
        labels = np.zeros(50, dtype=int)
        ev = Evaluator()
        metrics = ev.evaluate_clustering(X, labels)
        assert metrics["n_clusters"] == 1
        assert "silhouette_score" not in metrics
