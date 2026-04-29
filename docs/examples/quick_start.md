# Exemple — Démarrage rapide

Workflow complet en moins de 30 lignes : chargement → preprocessing → entraînement → évaluation → prédiction.

```python
import sys
sys.path.insert(0, "src")

from datalab.pipeline import DataIngestion, DataPreprocessor
from datalab.models import ModelTrainer
from datalab.models.predictor import Predictor
from datalab.evaluation import Evaluator

# ── Chargement ────────────────────────────────────────────────────────────────
ingestion = DataIngestion()
X, y, task = ingestion.load_sample("iris")
print(f"Dataset : {X.shape[0]} lignes × {X.shape[1]} colonnes | tâche : {task}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X, scaler="standard")

# ── Entraînement ──────────────────────────────────────────────────────────────
trainer = ModelTrainer(random_state=42)
result = trainer.train(X_processed, y, "Random Forest", task)

print(f"Score test : {result.test_score:.4f}")
print(f"CV         : {result.cv_mean:.4f} ± {result.cv_std:.4f}")

# ── Évaluation ────────────────────────────────────────────────────────────────
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(result.y_test, result.y_pred, result.y_proba)
print(f"Accuracy   : {metrics['accuracy']:.4f}")
print(f"F1-Score   : {metrics['f1_score']:.4f}")
print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")

# ── Prédiction sur nouvelles données ─────────────────────────────────────────
predictor = Predictor(model=result.model, feature_names=result.feature_names)
sample = X_processed.iloc[:3]
preds = predictor.predict(sample)
probas = predictor.predict_proba(sample)
for i, (pred, proba) in enumerate(zip(preds, probas)):
    print(f"  Obs {i+1} → classe {pred}  (confiance : {proba.max():.1%})")
```

**Sortie attendue :**

```
Dataset : 150 lignes × 4 colonnes | tâche : classification
Score test : 0.9667
CV         : 0.9600 ± 0.0327
Accuracy   : 0.9667
F1-Score   : 0.9665
ROC-AUC    : 0.9978
  Obs 1 → classe 0  (confiance : 99.0%)
  Obs 2 → classe 0  (confiance : 98.0%)
  Obs 3 → classe 0  (confiance : 97.0%)
```

---

## Comparer tous les modèles en une ligne

```python
df = trainer.compare_models(X_processed, y, task)
print(df.to_string(index=False))
```

```
          Modèle  Score Train  Score Test  CV Moyen  CV Std  Temps (s)
Gradient Boosting       1.0000      0.9333    0.9600  0.0327      0.150
    Random Forest       1.0000      0.9667    0.9533  0.0422      0.103
              SVM       0.9833      0.9667    0.9600  0.0327      0.002
              ...
```

---

## Charger son propre fichier CSV

```python
ingestion = DataIngestion()
X, y = ingestion.load_file(
    "mon_dataset.csv",
    target_col="label",  # colonne à prédire
)
```

---

## Sauvegarder et recharger un modèle

```python
# Sauvegarder
predictor.save("models/saved/iris_rf.joblib")

# Recharger dans un autre script
from datalab.models.predictor import Predictor
import pandas as pd

predictor = Predictor.load("models/saved/iris_rf.joblib")
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                        columns=predictor.feature_names)
print(predictor.predict(new_data))  # [0]
```
