# Exemple — Pipeline personnalisé avancé

Démonstration d'un pipeline complet avec feature engineering, optimisation d'hyperparamètres et analyse des résultats sur le dataset Titanic.

```python
import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datalabsn.pipeline import DataIngestion, DataPreprocessor, FeatureEngineer, DataValidator
from datalabsn.models import ModelTrainer
from datalabsn.models.predictor import Predictor
from datalabsn.evaluation import Evaluator

# ═══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET VALIDATION
# ═══════════════════════════════════════════════════════════════════════
ingestion = DataIngestion()
X, y, task = ingestion.load_sample("titanic")

validator = DataValidator()
report = validator.validate(X, y)
print("=== Rapport de validation ===")
print(f"  Lignes         : {report.stats['n_rows']}")
print(f"  Colonnes       : {report.stats['n_cols']}")
print(f"  Manquantes     : {report.stats['total_missing_pct']:.1f}%")
print(f"  Déséquilibre   : classe minoritaire = {report.stats.get('min_class_pct', '?')}%")
for w in report.warnings:
    print(f"  ⚠ {w}")

# ═══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING ADAPTÉ AU TITANIC
# ═══════════════════════════════════════════════════════════════════════
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(
    X,
    num_impute="median",        # Age a des valeurs manquantes → médiane robuste
    cat_impute="most_frequent", # Embarked a 2 valeurs manquantes
    scaler="standard",          # Normalisation pour le SVM et la régression logistique
    encode="onehot",            # Sex, Embarked sont nominales
    drop_duplicates=True,
)
print(f"\n=== Preprocessing ===")
print(f"  {X.shape[1]} colonnes brutes → {X_processed.shape[1]} colonnes après encodage")

# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
engineer = FeatureEngineer()

# Aligner y avec X_processed (drop_duplicates peut avoir réduit X)
y_aligned = y.loc[X_processed.index] if hasattr(X_processed, 'index') else y

# Supprimer les features très corrélées
corr_to_drop = engineer.correlation_analysis(X_processed, threshold=0.95)
if corr_to_drop:
    X_processed = X_processed.drop(columns=corr_to_drop)
    print(f"  Features corrélées supprimées : {corr_to_drop}")

# Sélectionner les 10 meilleures features
X_selected, scores = engineer.select_k_best(
    X_processed, y_aligned, k=10, task=task, score_func="mutual_info"
)
print(f"  {X_processed.shape[1]} features → {X_selected.shape[1]} sélectionnées")
print("\n  Top 5 features :")
for _, row in scores.head(5).iterrows():
    marker = "✓" if row["selected"] else " "
    print(f"    {marker} {row['feature']:<30} score={row['score']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. COMPARAISON RAPIDE DE MODÈLES
# ═══════════════════════════════════════════════════════════════════════
trainer = ModelTrainer(random_state=42, test_size=0.2, cv_folds=5)
print("\n=== Comparaison des modèles ===")
df_cmp = trainer.compare_models(
    X_selected, y_aligned, task,
    model_names=["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
)
print(df_cmp.to_string(index=False))

best_model_name = df_cmp.iloc[0]["Modèle"]
print(f"\n→ Meilleur modèle : {best_model_name}")

# ═══════════════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT DU MEILLEUR MODÈLE AVEC TUNING
# ═══════════════════════════════════════════════════════════════════════
print(f"\n=== Tuning de {best_model_name} ===")
result = trainer.train(
    X_selected, y_aligned,
    model_name=best_model_name,
    task=task,
    tune_hyperparams=True,
    search_strategy="random",
    n_iter=30,
)
if result.best_params:
    print(f"  Meilleurs params : {result.best_params}")
print(f"  Score test : {result.test_score:.4f}  |  CV : {result.cv_mean:.4f} ± {result.cv_std:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 6. ÉVALUATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(
    result.y_test, result.y_pred, result.y_proba
)
print("\n=== Métriques finales ===")
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  F1-Score  : {metrics['f1_score']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
if "roc_auc" in metrics:
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
print("\n" + metrics["classification_report"])

# ═══════════════════════════════════════════════════════════════════════
# 7. IMPORTANCE DES FEATURES
# ═══════════════════════════════════════════════════════════════════════
predictor = Predictor(model=result.model, feature_names=result.feature_names)
fi = predictor.feature_importance()
if fi is not None:
    print("=== Importance des features ===")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance"] / fi["importance"].max() * 20)
        print(f"  {row['feature']:<30} {bar} {row['importance']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 8. PRÉDICTIONS SUR CAS CONCRETS
# ═══════════════════════════════════════════════════════════════════════
# Créer quelques passagers fictifs pour tester
test_cases = pd.DataFrame({
    "Pclass": [1, 3],
    "Sex":    ["female", "male"],
    "Age":    [28.0, 35.0],
    "SibSp":  [1, 0],
    "Parch":  [0, 0],
    "Fare":   [100.0, 8.0],
    "Embarked": ["C", "S"],
})

# Appliquer le même preprocessing
test_processed = preprocessor.transform(test_cases)
# Aligner les colonnes avec les features sélectionnées
test_selected = test_processed[result.feature_names]

preds = predictor.predict(test_selected)
probas = predictor.predict_proba(test_selected)

print("\n=== Prédictions sur cas concrets ===")
labels = {0: "N'a pas survécu", 1: "A survécu"}
for i, (pred, proba) in enumerate(zip(preds, probas)):
    case = test_cases.iloc[i]
    confidence = proba.max()
    print(f"  Passager {i+1} : {case['Sex']}, {case['Age']} ans, classe {case['Pclass']}")
    print(f"    → {labels[pred]}  (confiance : {confidence:.1%})")

# ═══════════════════════════════════════════════════════════════════════
# 9. SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════
predictor.save("models/saved/titanic_best_model.joblib")
print(f"\n✓ Modèle sauvegardé dans models/saved/titanic_best_model.joblib")
```

---

## Lancer cet exemple

```bash
python docs/examples/custom_pipeline.py
```

Ou depuis la racine du projet dans un notebook :

```python
%run docs/examples/custom_pipeline.py
```
