from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from datalabsn.utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Sélection et transformation des features."""

    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        task: str = "classification",
        score_func: str = "f_test",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        score_funcs = {
            "f_test": f_classif if task == "classification" else f_regression,
            "mutual_info": mutual_info_classif if task == "classification" else mutual_info_regression,
        }
        fn = score_funcs.get(score_func, f_classif)
        k = min(k, X.shape[1])

        selector = SelectKBest(fn, k=k)
        X_new = selector.fit_transform(X, y)

        selected = X.columns[selector.get_support()].tolist()
        scores = pd.DataFrame({
            "feature": X.columns,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }).sort_values("score", ascending=False)

        logger.info("Features sélectionnées (%d/%d) : %s", k, X.shape[1], selected[:5])
        return pd.DataFrame(X_new, columns=selected, index=X.index), scores

    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: int | float = 0.95,
    ) -> tuple[pd.DataFrame, PCA]:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        logger.info(
            "PCA : %d composantes (variance expliquée : %.1f%%)",
            X_pca.shape[1],
            pca.explained_variance_ratio_.sum() * 100,
        )
        return pd.DataFrame(X_pca, columns=cols, index=X.index), pca

    def correlation_analysis(self, df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
        """Identifie les features fortement corrélées à supprimer."""
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        if to_drop:
            logger.info("Features très corrélées à supprimer : %s", to_drop)
        return to_drop

    def add_polynomial_features(
        self, df: pd.DataFrame, degree: int = 2, cols: list[str] | None = None
    ) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures

        target_cols = cols or df.select_dtypes(include=np.number).columns.tolist()
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(df[target_cols])
        poly_cols = poly.get_feature_names_out(target_cols)
        new_cols = [c for c in poly_cols if c not in target_cols]
        new_idx = [list(poly_cols).index(c) for c in new_cols]
        poly_df = pd.DataFrame(X_poly[:, new_idx], columns=new_cols, index=df.index)
        return pd.concat([df, poly_df], axis=1)
