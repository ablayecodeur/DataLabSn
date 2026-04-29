# Référence API — Évaluation

## Evaluator

```python
from datalab.evaluation import Evaluator
```

### `evaluate_classification(y_true, y_pred, y_proba, average)`

Calcule les métriques de classification.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `y_true` | `np.ndarray` | — | Labels réels |
| `y_pred` | `np.ndarray` | — | Labels prédits |
| `y_proba` | `np.ndarray \| None` | `None` | Probabilités de shape (n, n_classes) |
| `average` | `str` | `"weighted"` | `"weighted"`, `"macro"`, `"micro"` |

**Retourne :** `dict` avec :

| Clé | Type | Description |
|---|---|---|
| `accuracy` | `float` | Accuracy globale |
| `f1_score` | `float` | F1-Score (average) |
| `precision` | `float` | Precision (average) |
| `recall` | `float` | Recall (average) |
| `confusion_matrix` | `np.ndarray` | Matrice de confusion (n_classes × n_classes) |
| `classification_report` | `str` | Rapport complet sklearn |
| `roc_auc` | `float` | ROC-AUC *(si y_proba fourni)* |

```python
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(
    result.y_test,
    result.y_pred,
    result.y_proba,
)
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"ROC-AUC  : {metrics.get('roc_auc', 'N/A')}")
print(metrics["classification_report"])
```

---

### `evaluate_regression(y_true, y_pred)`

Calcule les métriques de régression.

**Retourne :** `dict` avec :

| Clé | Type | Description |
|---|---|---|
| `r2_score` | `float` | Coefficient de détermination R² |
| `mae` | `float` | Mean Absolute Error |
| `mse` | `float` | Mean Squared Error |
| `rmse` | `float` | Root Mean Squared Error |
| `mape` | `float` | Mean Absolute Percentage Error (%) |

```python
metrics = evaluator.evaluate_regression(result.y_test, result.y_pred)
print(f"R²   : {metrics['r2_score']:.4f}")
print(f"RMSE : {metrics['rmse']:.4f}")
print(f"MAE  : {metrics['mae']:.4f}")
```

---

### `evaluate_clustering(X, labels)`

Calcule les métriques de clustering.

**Paramètres :**

| Paramètre | Type | Description |
|---|---|---|
| `X` | `np.ndarray` | Données originales |
| `labels` | `np.ndarray` | Labels de cluster assignés |

**Retourne :** `dict` avec :

| Clé | Type | Description |
|---|---|---|
| `n_clusters` | `int` | Nombre de clusters (hors bruit DBSCAN) |
| `silhouette_score` | `float` | Score silhouette [-1, 1] *(si n_clusters > 1)* |

---

### `learning_curve_data(model, X, y, task)`

Calcule les données pour tracer les courbes d'apprentissage.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `model` | sklearn estimator | — | Modèle entraîné |
| `X` | `np.ndarray` | — | Features |
| `y` | `np.ndarray` | — | Cible |
| `task` | `str` | `"classification"` | Détermine le scoring |

**Retourne :** `pd.DataFrame` avec les colonnes :
- `train_size` : taille du set d'entraînement
- `train_mean`, `train_std` : score moyen ± std sur l'entraînement
- `val_mean`, `val_std` : score moyen ± std sur la validation

```python
lc_data = evaluator.learning_curve_data(result.model, X.values, y.values, task)

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=lc_data["train_size"], y=lc_data["train_mean"],
                         name="Train"))
fig.add_trace(go.Scatter(x=lc_data["train_size"], y=lc_data["val_mean"],
                         name="Validation"))
fig.show()
```
