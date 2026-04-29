from __future__ import annotations

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFIERS: dict[str, dict] = {
    "Logistic Regression": {
        "model": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42},
        "tuning": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "description": "Modèle linéaire, rapide et interprétable.",
    },
    "Random Forest": {
        "model": RandomForestClassifier,
        "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "tuning": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "description": "Ensemble d'arbres, robuste et précis.",
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "tuning": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "description": "Boosting séquentiel, excellent sur données tabulaires.",
    },
    "SVM": {
        "model": SVC,
        "params": {"probability": True, "random_state": 42},
        "tuning": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear", "poly"],
        },
        "description": "Marge maximale, efficace sur données de haute dimension.",
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier,
        "params": {"random_state": 42},
        "tuning": {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
        },
        "description": "Arbre de décision, très interprétable.",
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier,
        "params": {"n_jobs": -1},
        "tuning": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        },
        "description": "Basé sur la proximité, simple et non-paramétrique.",
    },
    "Naive Bayes": {
        "model": GaussianNB,
        "params": {},
        "tuning": {},
        "description": "Probabiliste, très rapide sur texte et données éparses.",
    },
}

REGRESSORS: dict[str, dict] = {
    "Linear Regression": {
        "model": LinearRegression,
        "params": {},
        "tuning": {},
        "description": "Régression linéaire simple, très interprétable.",
    },
    "Ridge": {
        "model": Ridge,
        "params": {"random_state": 42},
        "tuning": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "description": "Régression L2 — réduit le surapprentissage.",
    },
    "Lasso": {
        "model": Lasso,
        "params": {"random_state": 42, "max_iter": 10000},
        "tuning": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        "description": "Régression L1 — sélection automatique de features.",
    },
    "ElasticNet": {
        "model": ElasticNet,
        "params": {"random_state": 42, "max_iter": 10000},
        "tuning": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
        "description": "Combinaison L1+L2, robuste avec beaucoup de features.",
    },
    "Random Forest": {
        "model": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "tuning": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
        },
        "description": "Ensemble non-linéaire, gère bien les interactions.",
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "random_state": 42},
        "tuning": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "description": "Boosting, souvent le plus précis sur données tabulaires.",
    },
    "SVR": {
        "model": SVR,
        "params": {},
        "tuning": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        },
        "description": "Support Vector Regression, robuste aux outliers.",
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsRegressor,
        "params": {"n_jobs": -1},
        "tuning": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
        "description": "Régression par voisinage, non-paramétrique.",
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor,
        "params": {"random_state": 42},
        "tuning": {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
        },
        "description": "Arbre de régression, facilement interprétable.",
    },
}

CLUSTERERS: dict[str, dict] = {
    "K-Means": {
        "model": KMeans,
        "params": {"random_state": 42, "n_init": "auto"},
        "tuning": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
        "description": "Clustering centroïde, rapide et scalable.",
    },
    "DBSCAN": {
        "model": DBSCAN,
        "params": {},
        "tuning": {
            "eps": [0.3, 0.5, 0.8, 1.0],
            "min_samples": [3, 5, 10],
        },
        "description": "Clustering par densité, détecte les formes arbitraires.",
    },
    "Agglomerative": {
        "model": AgglomerativeClustering,
        "params": {},
        "tuning": {
            "n_clusters": [2, 3, 4, 5, 6],
            "linkage": ["ward", "complete", "average"],
        },
        "description": "Clustering hiérarchique, donne un dendrogramme.",
    },
}


class ModelRegistry:
    """Registre centralisé des modèles disponibles."""

    TASK_MAP = {
        "classification": CLASSIFIERS,
        "regression": REGRESSORS,
        "clustering": CLUSTERERS,
    }

    @classmethod
    def get(cls, task: str, name: str):
        catalog = cls.TASK_MAP.get(task, {})
        if name not in catalog:
            raise ValueError(f"Modèle '{name}' non trouvé pour la tâche '{task}'.")
        info = catalog[name]
        return info["model"](**info["params"])

    @classmethod
    def list_models(cls, task: str) -> list[str]:
        return list(cls.TASK_MAP.get(task, {}).keys())

    @classmethod
    def get_info(cls, task: str, name: str) -> dict:
        return cls.TASK_MAP[task][name]
