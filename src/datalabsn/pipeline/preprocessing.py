from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from datalabsn.utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Nettoyage, encodage et normalisation des données."""

    SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "none": None,
    }

    IMPUTERS = {
        "mean": lambda: SimpleImputer(strategy="mean"),
        "median": lambda: SimpleImputer(strategy="median"),
        "most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
        "knn": lambda: KNNImputer(n_neighbors=5),
    }

    def __init__(self):
        self._scaler = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._ohe: Optional[OneHotEncoder] = None
        self._ohe_cols: list[str] = []
        self._num_imputer = None
        self._cat_imputer = None
        self._fitted = False

    def analyze(self, df: pd.DataFrame) -> dict:
        """Retourne un rapport d'analyse du dataframe."""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        return {
            "shape": df.shape,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "missing": missing.to_dict(),
            "missing_pct": (missing / len(df) * 100).round(2).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        }

    def fit_transform(
        self,
        df: pd.DataFrame,
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        scaler: str = "standard",
        encode: str = "onehot",
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        df = df.copy()

        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df)
            if removed:
                logger.info("Doublons supprimés : %d", removed)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Imputation numérique
        if num_cols and df[num_cols].isnull().any().any():
            self._num_imputer = self.IMPUTERS[num_impute]()
            df[num_cols] = self._num_imputer.fit_transform(df[num_cols])
            logger.info("Imputation numérique (%s) appliquée", num_impute)

        # Imputation catégorielle
        if cat_cols and df[cat_cols].isnull().any().any():
            self._cat_imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = self._cat_imputer.fit_transform(df[cat_cols])
            logger.info("Imputation catégorielle appliquée")

        # Encodage
        if cat_cols:
            if encode == "onehot":
                self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                self._ohe_cols = cat_cols
                encoded = self._ohe.fit_transform(df[cat_cols])
                ohe_df = pd.DataFrame(
                    encoded,
                    columns=self._ohe.get_feature_names_out(cat_cols),
                    index=df.index,
                )
                df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)
            elif encode == "label":
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._label_encoders[col] = le
            elif encode == "ordinal":
                enc = OrdinalEncoder()
                df[cat_cols] = enc.fit_transform(df[cat_cols])

        # Normalisation
        scaler_cls = self.SCALERS.get(scaler)
        if scaler_cls:
            num_cols_final = df.select_dtypes(include=np.number).columns.tolist()
            self._scaler = scaler_cls()
            df[num_cols_final] = self._scaler.fit_transform(df[num_cols_final])
            logger.info("Normalisation (%s) appliquée", scaler)

        self._fitted = True
        logger.info("Preprocessing terminé : %d lignes × %d colonnes", *df.shape)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations ajustées sur de nouvelles données."""
        if not self._fitted:
            raise RuntimeError("Appelez fit_transform() avant transform().")
        df = df.copy()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if self._num_imputer and num_cols:
            df[num_cols] = self._num_imputer.transform(df[num_cols])
        if self._cat_imputer and cat_cols:
            df[cat_cols] = self._cat_imputer.transform(df[cat_cols])

        if self._ohe and self._ohe_cols:
            encoded = self._ohe.transform(df[self._ohe_cols])
            ohe_df = pd.DataFrame(
                encoded,
                columns=self._ohe.get_feature_names_out(self._ohe_cols),
                index=df.index,
            )
            df = pd.concat([df.drop(columns=self._ohe_cols), ohe_df], axis=1)
        elif self._label_encoders:
            for col, le in self._label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col].astype(str))

        if self._scaler:
            num_cols_final = df.select_dtypes(include=np.number).columns.tolist()
            df[num_cols_final] = self._scaler.transform(df[num_cols_final])

        return df
