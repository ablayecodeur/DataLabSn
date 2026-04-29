from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False


class DataValidator:
    """Valide la qualité des données avant modélisation."""

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        min_rows: int = 50,
        max_missing_pct: float = 50.0,
    ) -> ValidationReport:
        report = ValidationReport()
        report.stats["n_rows"] = len(X)
        report.stats["n_cols"] = X.shape[1]

        if len(X) < min_rows:
            report.add_error(f"Trop peu de lignes : {len(X)} (minimum : {min_rows})")

        missing_pct = X.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > max_missing_pct]
        if not high_missing.empty:
            report.add_warning(
                f"Colonnes avec beaucoup de valeurs manquantes (>{max_missing_pct}%) : "
                f"{high_missing.index.tolist()}"
            )

        report.stats["total_missing_pct"] = round(X.isnull().mean().mean() * 100, 2)

        constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
        if constant_cols:
            report.add_warning(f"Colonnes constantes détectées : {constant_cols}")
        report.stats["constant_cols"] = constant_cols

        dup_count = X.duplicated().sum()
        if dup_count > 0:
            report.add_warning(f"{dup_count} lignes dupliquées détectées")
        report.stats["duplicates"] = int(dup_count)

        num_cols = X.select_dtypes(include=np.number)
        if not num_cols.empty:
            z_scores = np.abs((num_cols - num_cols.mean()) / num_cols.std())
            outlier_pct = (z_scores > 3).mean().mean() * 100
            report.stats["outlier_pct"] = round(outlier_pct, 2)
            if outlier_pct > 10:
                report.add_warning(f"Fort taux d'outliers : {outlier_pct:.1f}%")

        if y is not None:
            report.stats["target_unique"] = y.nunique()
            report.stats["target_missing"] = int(y.isnull().sum())
            if y.isnull().any():
                report.add_error("La cible contient des valeurs manquantes")

            if y.dtype in (np.int64, np.int32, object, "category"):
                counts = y.value_counts(normalize=True)
                min_class_pct = counts.min() * 100
                report.stats["min_class_pct"] = round(min_class_pct, 2)
                if min_class_pct < 5:
                    report.add_warning(
                        f"Déséquilibre de classes détecté : classe minoritaire = {min_class_pct:.1f}%"
                    )

        return report
