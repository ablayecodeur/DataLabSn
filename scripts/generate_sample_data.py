#!/usr/bin/env python3
"""Génère des fichiers de données d'exemple dans data/samples/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from datalabsn.pipeline import DataIngestion


def main():
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    ingestion = DataIngestion()
    datasets = ["iris", "wine", "titanic", "ecommerce", "diabetes"]

    for name in datasets:
        try:
            X, y, task = ingestion.load_sample(name)
            df = X.copy()
            df["target"] = y.values
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"✓ {name:20s} → {path}  ({df.shape[0]} lignes × {df.shape[1]} colonnes)")
        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"\nFichiers générés dans : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
