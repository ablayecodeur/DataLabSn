"""Tests des modèles ML."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets as sk_datasets

from datalab.models import ModelRegistry, ModelTrainer, CLASSIFIERS, REGRESSORS
from datalab.models.predictor import Predictor
from datalab.pipeline import DataPreprocessor


@pytest.fixture
def classification_data():
    X, y = sk_datasets.load_iris(return_X_y=True, as_frame=True)
    prep = DataPreprocessor()
    X_proc = prep.fit_transform(X, scaler="standard")
    return X_proc, pd.Series(y, name="target")


@pytest.fixture
def regression_data():
    X, y = sk_datasets.load_diabetes(return_X_y=True, as_frame=True)
    prep = DataPreprocessor()
    X_proc = prep.fit_transform(X, scaler="standard")
    return X_proc, pd.Series(y, name="target")


class TestModelRegistry:
    def test_get_classifier(self):
        model = ModelRegistry.get("classification", "Random Forest")
        assert hasattr(model, "fit")

    def test_get_regressor(self):
        model = ModelRegistry.get("regression", "Linear Regression")
        assert hasattr(model, "fit")

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            ModelRegistry.get("classification", "nonexistent")

    def test_list_classifiers(self):
        models = ModelRegistry.list_models("classification")
        assert "Random Forest" in models
        assert "Logistic Regression" in models

    def test_list_regressors(self):
        models = ModelRegistry.list_models("regression")
        assert "Linear Regression" in models
        assert "Ridge" in models

    def test_all_classifiers_instantiate(self):
        for name in CLASSIFIERS:
            model = ModelRegistry.get("classification", name)
            assert model is not None

    def test_all_regressors_instantiate(self):
        for name in REGRESSORS:
            model = ModelRegistry.get("regression", name)
            assert model is not None


class TestModelTrainer:
    def test_basic_classification(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Logistic Regression", "classification")
        assert result.train_score > 0.5
        assert result.test_score > 0.5
        assert len(result.cv_scores) == 3

    def test_basic_regression(self, regression_data):
        X, y = regression_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Linear Regression", "regression")
        assert result.train_score > 0
        assert result.test_score != 0

    def test_result_has_predictions(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Random Forest", "classification")
        assert result.y_pred is not None
        assert len(result.y_pred) == len(result.y_test)

    def test_result_has_probas(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Random Forest", "classification")
        assert result.y_proba is not None
        assert result.y_proba.shape[1] == 3

    def test_compare_returns_dataframe(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        df = trainer.compare_models(X, y, "classification",
                                   ["Logistic Regression", "Naive Bayes"])
        assert isinstance(df, pd.DataFrame)
        assert "Modèle" in df.columns
        assert len(df) == 2

    def test_training_time_recorded(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Logistic Regression", "classification")
        assert result.training_time > 0

    def test_feature_names_stored(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Random Forest", "classification")
        assert result.feature_names == list(X.columns)


class TestPredictor:
    def test_predict(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Random Forest", "classification")
        predictor = Predictor(result.model, result.feature_names)
        preds = predictor.predict(result.X_test)
        assert len(preds) == len(result.y_test)

    def test_feature_importance(self, classification_data):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Random Forest", "classification")
        predictor = Predictor(result.model, result.feature_names)
        fi = predictor.feature_importance()
        assert fi is not None
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_save_load(self, classification_data, tmp_path):
        X, y = classification_data
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train(X, y, "Logistic Regression", "classification")
        predictor = Predictor(result.model, result.feature_names)

        save_path = tmp_path / "model.joblib"
        predictor.save(save_path)
        loaded = Predictor.load(save_path)

        preds_orig = predictor.predict(result.X_test)
        preds_loaded = loaded.predict(result.X_test)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
