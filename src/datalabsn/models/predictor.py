from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datalabsn.utils import get_logger

logger = get_logger(__name__)


class Predictor:
    """Sauvegarde, chargement et prédiction avec un modèle entraîné."""

    def __init__(self, model: Any = None, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or []

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self.predict(X)
        result = X.copy()
        result["prediction"] = preds

        probas = self.predict_proba(X)
        if probas is not None:
            result["confidence"] = probas.max(axis=1).round(4)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "feature_names": self.feature_names}
        joblib.dump(payload, path)
        logger.info("Modèle sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        payload = joblib.load(path)
        return cls(model=payload["model"], feature_names=payload.get("feature_names", []))

    def feature_importance(self) -> pd.DataFrame | None:
        if hasattr(self.model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
            return fi
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            fi = pd.DataFrame({
                "feature": self.feature_names,
                "importance": np.abs(coef),
            }).sort_values("importance", ascending=False)
            return fi
        return None
