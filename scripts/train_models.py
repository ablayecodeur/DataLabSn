#!/usr/bin/env python3
"""Benchmark : entraîne tous les modèles sur chaque dataset et affiche les résultats."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from datalab.pipeline import DataIngestion, DataPreprocessor
from datalab.models import ModelTrainer


BENCHMARKS = [
    ("iris", "classification"),
    ("wine", "classification"),
    ("breast_cancer", "classification"),
    ("diabetes", "regression"),
]


def run_benchmark():
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(random_state=42, test_size=0.2, cv_folds=5)

    for dataset_name, task in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Dataset : {dataset_name} | Tâche : {task}")
        print("=" * 60)

        X, y, _ = ingestion.load_sample(dataset_name)
        X_proc = preprocessor.fit_transform(X, scaler="standard", encode="onehot")

        df = trainer.compare_models(X_proc, y, task)
        print(df.to_string(index=False))

        best = df.iloc[0]
        print(f"\n→ Meilleur : {best['Modèle']} (CV: {best['CV Moyen']:.4f})")


if __name__ == "__main__":
    run_benchmark()
