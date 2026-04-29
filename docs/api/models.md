# Référence API — Modèles

## ModelRegistry

```python
from datalab.models import ModelRegistry
```

Registre centralisé de tous les algorithmes disponibles.

### `ModelRegistry.get(task, name)`

Instancie un modèle par tâche et nom.

**Paramètres :**

| Paramètre | Type | Valeurs |
|---|---|---|
| `task` | `str` | `"classification"`, `"regression"`, `"clustering"` |
| `name` | `str` | Nom de l'algorithme (voir listes ci-dessous) |

**Retourne :** instance sklearn non entraînée

**Lève :** `ValueError` si le modèle n'existe pas

```python
model = ModelRegistry.get("classification", "Random Forest")
model.fit(X_train, y_train)
```

---

### `ModelRegistry.list_models(task)`

**Retourne :** `list[str]` — noms des modèles disponibles pour la tâche

```python
classifiers = ModelRegistry.list_models("classification")
# ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM',
#  'Decision Tree', 'K-Nearest Neighbors', 'Naive Bayes']
```

---

### `ModelRegistry.get_info(task, name)`

**Retourne :** `dict` avec les clés :
- `model` : classe sklearn
- `params` : paramètres par défaut
- `tuning` : grille d'hyperparamètres pour la recherche
- `description` : description courte

---

### Modèles disponibles

**Classification**

| Nom | Classe sklearn |
|---|---|
| `"Logistic Regression"` | `LogisticRegression` |
| `"Random Forest"` | `RandomForestClassifier` |
| `"Gradient Boosting"` | `GradientBoostingClassifier` |
| `"SVM"` | `SVC` |
| `"Decision Tree"` | `DecisionTreeClassifier` |
| `"K-Nearest Neighbors"` | `KNeighborsClassifier` |
| `"Naive Bayes"` | `GaussianNB` |

**Régression**

| Nom | Classe sklearn |
|---|---|
| `"Linear Regression"` | `LinearRegression` |
| `"Ridge"` | `Ridge` |
| `"Lasso"` | `Lasso` |
| `"ElasticNet"` | `ElasticNet` |
| `"Random Forest"` | `RandomForestRegressor` |
| `"Gradient Boosting"` | `GradientBoostingRegressor` |
| `"SVR"` | `SVR` |
| `"K-Nearest Neighbors"` | `KNeighborsRegressor` |
| `"Decision Tree"` | `DecisionTreeRegressor` |

**Clustering**

| Nom | Classe sklearn |
|---|---|
| `"K-Means"` | `KMeans` |
| `"DBSCAN"` | `DBSCAN` |
| `"Agglomerative"` | `AgglomerativeClustering` |

---

## ModelTrainer

```python
from datalab.models import ModelTrainer
```

### Constructeur

```python
trainer = ModelTrainer(
    random_state=42,   # Reproductibilité
    test_size=0.2,     # Proportion du set de test
    cv_folds=5,        # Nombre de folds CV
)
```

---

### `train(X, y, model_name, task, tune_hyperparams, search_strategy, n_iter)`

Entraîne un modèle avec validation croisée.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Features |
| `y` | `pd.Series` | — | Cible |
| `model_name` | `str` | — | Nom du modèle dans le registre |
| `task` | `str` | `"classification"` | `"classification"` ou `"regression"` |
| `tune_hyperparams` | `bool` | `False` | Active la recherche d'hyperparamètres |
| `search_strategy` | `str` | `"grid"` | `"grid"` (GridSearch) ou `"random"` (RandomSearch) |
| `n_iter` | `int` | `20` | Nombre d'itérations pour RandomSearch |

**Retourne :** `TrainingResult`

---

### `TrainingResult` — attributs

| Attribut | Type | Description |
|---|---|---|
| `model` | sklearn estimator | Modèle entraîné |
| `model_name` | `str` | Nom du modèle |
| `task` | `str` | Tâche ML |
| `train_score` | `float` | Score sur l'ensemble d'entraînement |
| `test_score` | `float` | Score sur l'ensemble de test |
| `cv_scores` | `np.ndarray` | Scores de chaque fold CV |
| `cv_mean` | `float` | Moyenne des scores CV |
| `cv_std` | `float` | Écart-type des scores CV |
| `best_params` | `dict` | Meilleurs hyperparamètres (si tuning activé) |
| `feature_names` | `list[str]` | Noms des features |
| `training_time` | `float` | Durée d'entraînement en secondes |
| `X_test` | `np.ndarray` | Set de test (features) |
| `y_test` | `np.ndarray` | Set de test (cible réelle) |
| `y_pred` | `np.ndarray` | Prédictions sur le set de test |
| `y_proba` | `np.ndarray \| None` | Probabilités (classification uniquement) |

```python
result = trainer.train(X, y, "Random Forest", "classification")

print(f"Test accuracy : {result.test_score:.4f}")
print(f"CV            : {result.cv_mean:.4f} ± {result.cv_std:.4f}")
print(f"Durée         : {result.training_time:.2f}s")
```

---

### `compare_models(X, y, task, model_names)`

Entraîne plusieurs modèles et les compare.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Features |
| `y` | `pd.Series` | — | Cible |
| `task` | `str` | `"classification"` | Tâche ML |
| `model_names` | `list[str] \| None` | `None` | Liste de modèles, ou tous si `None` |

**Retourne :** `pd.DataFrame` trié par CV Moyen décroissant, avec les colonnes :
`Modèle`, `Score Train`, `Score Test`, `CV Moyen`, `CV Std`, `Temps (s)`, `Surapprentissage`

```python
df = trainer.compare_models(X, y, "regression",
                            model_names=["Ridge", "Lasso", "Random Forest"])
print(df.to_string(index=False))
```

---

## Predictor

```python
from datalab.models.predictor import Predictor
```

### Constructeur

```python
predictor = Predictor(
    model=result.model,
    feature_names=result.feature_names,
)
```

---

### `predict(X)`

**Paramètres :** `X : pd.DataFrame | np.ndarray`

**Retourne :** `np.ndarray` de prédictions

---

### `predict_proba(X)`

**Retourne :** `np.ndarray` de shape `(n_samples, n_classes)` ou `None` si le modèle ne supporte pas les probabilités

---

### `predict_with_confidence(X)`

**Retourne :** `pd.DataFrame` = X + colonnes `prediction` et `confidence`

---

### `feature_importance()`

**Retourne :** `pd.DataFrame` avec colonnes `feature`, `importance` trié par importance décroissante, ou `None` si non disponible pour ce modèle.

Fonctionne pour :
- Modèles avec `feature_importances_` : Random Forest, Gradient Boosting, Decision Tree
- Modèles avec `coef_` : Logistic Regression, Ridge, Lasso, ElasticNet, SVM linéaire

---

### `save(path)`

Sauvegarde le modèle et les noms de features dans un fichier `.joblib`.

```python
predictor.save("models/saved/mon_modele.joblib")
```

---

### `Predictor.load(path)` *(classmethod)*

Charge un modèle sauvegardé.

```python
predictor = Predictor.load("models/saved/mon_modele.joblib")
preds = predictor.predict(X_new)
```
