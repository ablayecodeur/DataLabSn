"""Tests du pipeline de traitement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from datalab.pipeline import DataIngestion, DataPreprocessor, DataValidator, FeatureEngineer


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, 100),
        "num2": rng.exponential(2, 100),
        "cat1": rng.choice(["a", "b", "c"], 100),
        "cat2": rng.choice(["x", "y"], 100),
    })
    df.loc[rng.integers(0, 100, 10), "num1"] = np.nan
    df.loc[rng.integers(0, 100, 5), "cat1"] = np.nan
    return df


@pytest.fixture
def sample_target():
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice([0, 1], 100), name="target")


class TestDataIngestion:
    def test_load_iris(self):
        ing = DataIngestion()
        X, y, task = ing.load_sample("iris")
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert task == "classification"

    def test_load_diabetes(self):
        ing = DataIngestion()
        X, y, task = ing.load_sample("diabetes")
        assert X.shape[0] == 442
        assert task == "regression"

    def test_load_titanic(self):
        ing = DataIngestion()
        X, y, task = ing.load_sample("titanic")
        assert X.shape[0] == 891
        assert task == "classification"

    def test_invalid_dataset(self):
        ing = DataIngestion()
        with pytest.raises(ValueError, match="Dataset inconnu"):
            ing.load_sample("nonexistent")


class TestDataPreprocessor:
    def test_fit_transform_shape(self, sample_df):
        prep = DataPreprocessor()
        result = prep.fit_transform(sample_df, scaler="standard", encode="onehot")
        assert result.shape[0] == 100
        assert result.shape[1] > sample_df.shape[1]

    def test_no_missing_after_imputation(self, sample_df):
        prep = DataPreprocessor()
        result = prep.fit_transform(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_all_numeric_after_encoding(self, sample_df):
        prep = DataPreprocessor()
        result = prep.fit_transform(sample_df, encode="onehot")
        assert all(result.dtypes.apply(lambda t: np.issubdtype(t, np.number)))

    def test_label_encoding(self, sample_df):
        prep = DataPreprocessor()
        result = prep.fit_transform(sample_df, encode="label", scaler="none")
        assert "cat1" in result.columns
        assert result["cat1"].dtype in (np.int32, np.int64, np.float64)

    def test_transform_uses_fitted_params(self, sample_df):
        prep = DataPreprocessor()
        prep.fit_transform(sample_df, scaler="standard")
        new_data = sample_df.iloc[:10].copy()
        result = prep.transform(new_data)
        assert result.shape[0] == 10

    def test_transform_without_fit_raises(self, sample_df):
        prep = DataPreprocessor()
        with pytest.raises(RuntimeError):
            prep.transform(sample_df)

    def test_duplicate_removal(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})
        prep = DataPreprocessor()
        result = prep.fit_transform(df, drop_duplicates=True, scaler="none", encode="label")
        assert len(result) == 3

    def test_scalers(self, sample_df):
        for scaler in ("standard", "minmax", "robust", "none"):
            prep = DataPreprocessor()
            result = prep.fit_transform(sample_df.select_dtypes(include=np.number), scaler=scaler)
            assert result.isnull().sum().sum() == 0


class TestDataValidator:
    def test_valid_data(self, sample_df, sample_target):
        validator = DataValidator()
        report = validator.validate(sample_df, sample_target)
        assert isinstance(report.passed, bool)
        assert "n_rows" in report.stats

    def test_too_few_rows(self):
        tiny = pd.DataFrame({"a": range(10)})
        validator = DataValidator()
        report = validator.validate(tiny, min_rows=50)
        assert not report.passed
        assert any("Trop peu" in e for e in report.errors)

    def test_detects_missing_target(self, sample_df):
        y = pd.Series([0, None] * 50, dtype=float)
        validator = DataValidator()
        report = validator.validate(sample_df, y)
        assert not report.passed

    def test_detects_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 5]})
        validator = DataValidator()
        report = validator.validate(df, min_rows=1)
        assert any("dupliqu" in w for w in report.warnings)


class TestFeatureEngineer:
    def test_select_k_best(self, sample_df, sample_target):
        prep = DataPreprocessor()
        X = prep.fit_transform(sample_df, scaler="standard")
        eng = FeatureEngineer()
        X_sel, scores = eng.select_k_best(X, sample_target, k=3)
        assert X_sel.shape[1] == 3
        assert len(scores) == X.shape[1]

    def test_pca_reduces_dims(self, sample_df):
        prep = DataPreprocessor()
        X = prep.fit_transform(sample_df.select_dtypes(include=np.number), scaler="standard")
        eng = FeatureEngineer()
        X_pca, pca = eng.apply_pca(X, n_components=2)
        assert X_pca.shape[1] == 2

    def test_correlation_analysis(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": [x * 1.001 for x in range(100)],
            "c": np.random.default_rng(42).normal(0, 1, 100),
        })
        eng = FeatureEngineer()
        to_drop = eng.correlation_analysis(df, threshold=0.99)
        assert len(to_drop) >= 1
