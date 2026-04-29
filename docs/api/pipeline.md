# Référence API — Pipeline

## DataIngestion

```python
from datalabsn.pipeline import DataIngestion
```

### `load_file(path, target_col=None, **kwargs)`

Charge un fichier de données depuis le disque.

**Paramètres :**

| Paramètre | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Chemin vers le fichier (.csv, .tsv, .xlsx, .xls, .parquet, .json) |
| `target_col` | `str \| None` | Nom de la colonne cible. Si fourni, elle est séparée de X. |
| `**kwargs` | | Arguments supplémentaires passés à `pd.read_csv` / `pd.read_excel` etc. |

**Retourne :** `tuple[pd.DataFrame, pd.Series | None]` — (X, y)

**Exemple :**
```python
ingestion = DataIngestion()
X, y = ingestion.load_file("data/titanic.csv", target_col="Survived", sep=",")
```

---

### `load_sample(name)`

Charge un dataset intégré.

**Paramètres :**

| Paramètre | Type | Valeurs acceptées |
|---|---|---|
| `name` | `str` | `iris`, `wine`, `breast_cancer`, `diabetes`, `california_housing`, `titanic`, `ecommerce` |

**Retourne :** `tuple[pd.DataFrame, pd.Series, str]` — (X, y, task) où task est `"classification"` ou `"regression"`

**Exemple :**
```python
X, y, task = ingestion.load_sample("breast_cancer")
# task == "classification"
```

---

## DataPreprocessor

```python
from datalabsn.pipeline import DataPreprocessor
```

### `analyze(df)`

Analyse le dataframe sans le modifier.

**Retourne :** `dict` avec les clés :
- `shape` : tuple (n_rows, n_cols)
- `numeric_cols` : liste des colonnes numériques
- `categorical_cols` : liste des colonnes catégorielles
- `missing` : dict {colonne: nombre de manquants}
- `missing_pct` : dict {colonne: pourcentage de manquants}
- `dtypes` : dict {colonne: type}
- `duplicates` : nombre de lignes dupliquées
- `memory_mb` : consommation mémoire en Mo

---

### `fit_transform(df, num_impute, cat_impute, scaler, encode, drop_duplicates)`

Ajuste et applique toutes les transformations.

**Paramètres :**

| Paramètre | Type | Défaut | Valeurs |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | — |
| `num_impute` | `str` | `"median"` | `"mean"`, `"median"`, `"knn"`, `"most_frequent"` |
| `cat_impute` | `str` | `"most_frequent"` | `"most_frequent"`, `"median"`, `"mean"` |
| `scaler` | `str` | `"standard"` | `"standard"`, `"minmax"`, `"robust"`, `"none"` |
| `encode` | `str` | `"onehot"` | `"onehot"`, `"label"`, `"ordinal"` |
| `drop_duplicates` | `bool` | `True` | — |

**Retourne :** `pd.DataFrame` transformé

**Exemple :**
```python
prep = DataPreprocessor()
X_processed = prep.fit_transform(
    X,
    num_impute="knn",
    scaler="robust",
    encode="onehot",
)
```

---

### `transform(df)`

Applique les transformations ajustées sur de nouvelles données. Doit être appelée après `fit_transform`.

**Retourne :** `pd.DataFrame`

**Exemple :**
```python
X_new_processed = prep.transform(X_new)
```

---

## FeatureEngineer

```python
from datalabsn.pipeline import FeatureEngineer
```

### `select_k_best(X, y, k, task, score_func)`

Sélectionne les k features les plus pertinentes.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Features |
| `y` | `pd.Series` | — | Variable cible |
| `k` | `int` | `10` | Nombre de features à conserver |
| `task` | `str` | `"classification"` | `"classification"` ou `"regression"` |
| `score_func` | `str` | `"f_test"` | `"f_test"` ou `"mutual_info"` |

**Retourne :** `tuple[pd.DataFrame, pd.DataFrame]` — (X_sélectionné, scores_dataframe)

---

### `apply_pca(X, n_components)`

Applique une PCA.

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Features numériques |
| `n_components` | `int \| float` | `0.95` | Nombre de composantes (int) ou variance cible (float entre 0 et 1) |

**Retourne :** `tuple[pd.DataFrame, PCA]` — (X_réduit, objet_PCA)

---

### `correlation_analysis(df, threshold)`

Identifie les features fortement corrélées.

**Retourne :** `list[str]` — colonnes à supprimer

---

## DataValidator

```python
from datalabsn.pipeline import DataValidator
```

### `validate(X, y, min_rows, max_missing_pct)`

Valide la qualité des données.

**Paramètres :**

| Paramètre | Type | Défaut |
|---|---|---|
| `X` | `pd.DataFrame` | — |
| `y` | `pd.Series \| None` | `None` |
| `min_rows` | `int` | `50` |
| `max_missing_pct` | `float` | `50.0` |

**Retourne :** `ValidationReport` avec :
- `passed` : bool
- `warnings` : list[str]
- `errors` : list[str]
- `stats` : dict

**Exemple :**
```python
validator = DataValidator()
report = validator.validate(X, y)
if not report.passed:
    for err in report.errors:
        print(f"ERREUR: {err}")
```
