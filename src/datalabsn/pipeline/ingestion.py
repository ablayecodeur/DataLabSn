from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets

from datalabsn.utils import get_logger

logger = get_logger(__name__)

SAMPLE_DATASETS = {
    "iris": "Classification — fleurs d'iris (150 lignes, 4 features)",
    "wine": "Classification — qualité de vin (178 lignes, 13 features)",
    "breast_cancer": "Classification — cancer du sein (569 lignes, 30 features)",
    "diabetes": "Régression — progression du diabète (442 lignes, 10 features)",
    "california_housing": "Régression — prix immobiliers Californie (20 640 lignes, 8 features)",
    "titanic": "Classification — survie Titanic (891 lignes, 11 features)",
    "ecommerce": "Classification — satisfaction client e-commerce (1 000 lignes, 8 features)",
}


class DataIngestion:
    """Charge des données depuis fichiers ou datasets intégrés."""

    def load_file(
        self,
        path: str | Path,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        path = Path(path)
        suffix = path.suffix.lower()

        loaders = {
            ".csv": pd.read_csv,
            ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".parquet": pd.read_parquet,
            ".json": pd.read_json,
        }

        if suffix not in loaders:
            raise ValueError(f"Format non supporté : {suffix}")

        df = loaders[suffix](path, **kwargs)
        logger.info("Fichier chargé : %s (%d lignes, %d colonnes)", path.name, *df.shape)

        if target_col and target_col in df.columns:
            return df.drop(columns=[target_col]), df[target_col]
        return df, None

    def load_sample(self, name: str) -> tuple[pd.DataFrame, pd.Series, str]:
        """Charge un dataset de démonstration."""
        loaders = {
            "iris": self._load_sklearn("iris", "classification"),
            "wine": self._load_sklearn("wine", "classification"),
            "breast_cancer": self._load_sklearn("breast_cancer", "classification"),
            "diabetes": self._load_sklearn("diabetes", "regression"),
            "california_housing": self._load_sklearn("fetch_california_housing", "regression"),
            "titanic": self._load_titanic,
            "ecommerce": self._load_ecommerce,
        }

        if name not in loaders:
            raise ValueError(f"Dataset inconnu. Choisissez parmi : {list(loaders)}")

        loader = loaders[name]
        if callable(loader):
            return loader()
        return loader

    def _load_sklearn(self, name: str, task: str):
        def _inner():
            fn_name = "fetch_california_housing" if name == "fetch_california_housing" else f"load_{name}"
            fn = getattr(sk_datasets, fn_name)
            ds = fn()
            X = pd.DataFrame(ds.data, columns=ds.feature_names)
            y = pd.Series(ds.target, name="target")
            logger.info("Dataset sklearn '%s' chargé : %d lignes", name, len(X))
            return X, y, task
        return _inner

    def _load_titanic(self) -> tuple[pd.DataFrame, pd.Series, str]:
        rng = np.random.default_rng(42)
        n = 891
        pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
        sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
        age = np.where(
            rng.random(n) < 0.05, np.nan,
            rng.normal(29, 14, n).clip(1, 80)
        )
        fare = np.where(pclass == 1, rng.exponential(80, n),
               np.where(pclass == 2, rng.exponential(20, n),
                        rng.exponential(12, n)))
        sibsp = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01])
        parch = rng.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.09, 0.02])
        embarked = rng.choice(["S", "C", "Q", np.nan], n, p=[0.72, 0.19, 0.086, 0.004])

        # Survival probability based on features
        p_survive = (
            0.74 * (sex == "female")
            + 0.40 * (sex == "male")
            + 0.05 * (pclass == 1)
            - 0.10 * (pclass == 3)
        ).clip(0, 1)
        survived = rng.binomial(1, p_survive)

        df = pd.DataFrame({
            "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
            "Parch": parch, "Fare": fare, "Embarked": embarked,
        })
        return df, pd.Series(survived, name="Survived"), "classification"

    def _load_ecommerce(self) -> tuple[pd.DataFrame, pd.Series, str]:
        rng = np.random.default_rng(42)
        n = 1000
        age = rng.integers(18, 70, n)
        purchases = rng.integers(1, 50, n)
        avg_order = rng.exponential(60, n).clip(5, 500)
        days_since_last = rng.integers(1, 365, n)
        returns = rng.integers(0, 10, n)
        pages_visited = rng.integers(1, 100, n)
        time_on_site = rng.exponential(15, n).clip(1, 120)
        membership = rng.choice(["bronze", "silver", "gold", "platinum"], n,
                                p=[0.4, 0.3, 0.2, 0.1])

        score = (
            0.4 * (purchases / 50)
            + 0.3 * (avg_order / 500)
            - 0.1 * (days_since_last / 365)
            - 0.1 * (returns / 10)
            + 0.1 * (time_on_site / 120)
        )
        satisfied = (score + rng.normal(0, 0.1, n) > 0.4).astype(int)

        df = pd.DataFrame({
            "age": age, "purchases": purchases, "avg_order_value": avg_order.round(2),
            "days_since_last_purchase": days_since_last, "returns": returns,
            "pages_visited": pages_visited, "time_on_site_min": time_on_site.round(1),
            "membership": membership,
        })
        return df, pd.Series(satisfied, name="satisfied"), "classification"
