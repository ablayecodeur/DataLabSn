<div align="center">

# 🔬 DataLabSn

**Plateforme d'analyse de données et de machine learning**

Pipeline de traitement · Modèles prédictifs · Dashboard interactif

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Tests](https://img.shields.io/badge/Tests-45%2F45%20✓-4CAF50?style=flat-square)](#tests)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-4CAF50?style=flat-square)](#tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

[📖 Documentation](#documentation) · [🚀 Démarrage rapide](#démarrage-rapide) · [✨ Fonctionnalités](#fonctionnalités) · [🤖 Modèles](#algorithmes-disponibles) · [🧪 Tests](#tests)

</div>

---

## Présentation

DataLabSn est une plateforme complète pour l'analyse de données et le machine learning, conçue pour permettre à des data scientists, chercheurs et développeurs d'explorer des datasets, construire des pipelines de traitement, entraîner et évaluer des modèles prédictifs, et prédire de nouvelles données — entièrement depuis une interface visuelle Streamlit ou en ligne de commande Python.

```
Données brutes → Pipeline → Modèle entraîné → Évaluation → Prédictions
```

### Ce que DataLabSn fait pour vous

- **Aucun code requis** pour les tâches courantes via le dashboard Streamlit
- **API Python complète** pour les workflows automatisés et les pipelines CI/CD
- **17 algorithmes ML** prêts à l'emploi avec optimisation d'hyperparamètres
- **7 datasets intégrés** pour démarrer immédiatement sans données
- **Export complet** : prédictions CSV, modèles sérialisés `.joblib`

---

## Fonctionnalités

<table>
<tr>
<td width="50%">

### 📊 Data Explorer
- Chargement de fichiers CSV, Excel, Parquet, JSON, TSV
- 7 datasets intégrés (Iris, Wine, Titanic, etc.)
- Statistiques descriptives complètes
- Analyse des valeurs manquantes avec visualisation
- Matrices de corrélation (Pearson, Spearman, Kendall)
- Distributions univariées et bivariées
- Rapport de validation automatique de la qualité

</td>
<td width="50%">

### 🔧 Pipeline de traitement
- Imputation : mean, median, most_frequent, KNN
- Encodage : OneHot, Label, Ordinal
- Normalisation : StandardScaler, MinMax, Robust
- Sélection de features : SelectKBest (F-test, mutual info)
- Réduction dimensionnelle : PCA avec variance cible
- Suppression des features corrélées (seuil configurable)
- Suppression des doublons

</td>
</tr>
<tr>
<td width="50%">

### 🤖 Entraînement de modèles
- 7 classifieurs + 9 régresseurs + 3 algorithmes de clustering
- Validation croisée stratifiée (k-fold configurable)
- Optimisation d'hyperparamètres : Grid Search / Random Search
- Comparaison côte-à-côte de tous les modèles
- Détection du surapprentissage (train vs test vs CV)
- Mesure du temps d'entraînement

</td>
<td width="50%">

### 📈 Évaluation complète
- **Classification** : Accuracy, F1, Precision, Recall, ROC-AUC
- **Régression** : R², RMSE, MAE, MAPE
- **Clustering** : Silhouette Score
- Matrice de confusion annotée
- Courbes ROC et Precision-Recall (binaire + multiclasse)
- Courbes d'apprentissage (détection overfitting/underfitting)
- Analyse des résidus pour la régression

</td>
</tr>
<tr>
<td width="50%">

### 🔮 Prédictions
- Formulaire de saisie manuelle adaptatif
- Prédictions en lot depuis un fichier
- Scores de confiance (probabilités par classe)
- Indication du percentile pour la régression
- Export des résultats en CSV

</td>
<td width="50%">

### 💾 Persistance & Export
- Sauvegarde des modèles en `.joblib`
- Chargement de modèles existants
- Export des prédictions en CSV
- Importance des features (tree-based et linéaires)
- Configuration via YAML ou variables d'environnement

</td>
</tr>
</table>

---

## Démarrage rapide

### Prérequis

- Python 3.10 ou supérieur
- pip

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/ablayecodeur/DataLabSn.git
cd DataLabSn

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le dashboard
streamlit run dashboard/app.py
```

Ouvrez **http://localhost:8501** dans votre navigateur.

### Workflow en 5 étapes

```
1. 📊 Data Explorer   →  Chargez un dataset (ou utilisez un dataset intégré)
2. 🔧 Pipeline        →  Configurez le preprocessing, cliquez ▶ Appliquer
3. 🤖 Entraînement    →  Choisissez un algorithme, cliquez 🚀 Entraîner
4. 📈 Évaluation      →  Consultez les métriques et visualisations
5. 🔮 Prédictions     →  Saisissez de nouvelles données ou uploadez un fichier
```

---

## Structure du projet

```
DataLabSn/
│
├── 📁 src/datalabsn/                   # Code source principal
│   ├── pipeline/
│   │   ├── ingestion.py              # Chargement de données multi-format
│   │   ├── preprocessing.py          # Imputation, encodage, normalisation
│   │   ├── features.py               # Sélection de features, PCA
│   │   └── validator.py              # Validation de la qualité des données
│   ├── models/
│   │   ├── registry.py               # Catalogue des 19 algorithmes
│   │   ├── trainer.py                # Entraînement, CV, comparaison, tuning
│   │   └── predictor.py              # Prédiction, sauvegarde/chargement
│   └── evaluation/
│       └── metrics.py                # Métriques classification/régression/clustering
│
├── 📁 dashboard/                     # Interface Streamlit
│   ├── app.py                        # Page d'accueil
│   └── pages/
│       ├── 1_📊_Data_Explorer.py
│       ├── 2_🔧_Pipeline.py
│       ├── 3_🤖_Model_Training.py
│       ├── 4_📈_Evaluation.py
│       └── 5_🔮_Predictions.py
│
├── 📁 docs/                          # Documentation complète
│   ├── user_guide/                   # Guide utilisateur par fonctionnalité
│   ├── api/                          # Référence API Python
│   └── examples/                     # Exemples de code
│
├── 📁 data/
│   ├── raw/                          # Données brutes (gitignorées)
│   ├── processed/                    # Données traitées (gitignorées)
│   └── samples/                      # Datasets d'exemple générés
│
├── 📁 scripts/
│   ├── generate_sample_data.py       # Génère les CSV d'exemple
│   └── train_models.py               # Benchmark CLI de tous les modèles
│
├── 📁 tests/                         # Suite de tests pytest (45 tests)
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── 📁 configs/
│   └── default.yaml                  # Configuration par défaut
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Algorithmes disponibles

### Classification — 7 modèles

| Algorithme | Classe sklearn | Points forts |
|---|---|---|
| **Logistic Regression** | `LogisticRegression` | Rapide, interprétable, baseline solide |
| **Random Forest** | `RandomForestClassifier` | Robuste, peu de tuning nécessaire |
| **Gradient Boosting** | `GradientBoostingClassifier` | Très précis sur données tabulaires |
| **SVM** | `SVC` | Efficace en haute dimension |
| **Decision Tree** | `DecisionTreeClassifier` | Visualisable, entièrement interprétable |
| **K-Nearest Neighbors** | `KNeighborsClassifier` | Non-paramétrique, simple |
| **Naive Bayes** | `GaussianNB` | Ultra-rapide, bien sur texte |

### Régression — 9 modèles

| Algorithme | Classe sklearn | Points forts |
|---|---|---|
| **Linear Regression** | `LinearRegression` | Baseline, coefficient directement lisible |
| **Ridge** | `Ridge` | Régularisation L2, réduit l'overfitting |
| **Lasso** | `Lasso` | Régularisation L1, sélection de features automatique |
| **ElasticNet** | `ElasticNet` | L1 + L2, nombreuses features corrélées |
| **Random Forest** | `RandomForestRegressor` | Non-linéaire, interactions automatiques |
| **Gradient Boosting** | `GradientBoostingRegressor` | Souvent le plus précis |
| **SVR** | `SVR` | Robuste aux outliers |
| **K-Nearest Neighbors** | `KNeighborsRegressor` | Local, non-paramétrique |
| **Decision Tree** | `DecisionTreeRegressor` | Interprétable, règles lisibles |

### Clustering — 3 algorithmes

| Algorithme | Classe sklearn | Points forts |
|---|---|---|
| **K-Means** | `KMeans` | Rapide, scalable, centroïdes interprétables |
| **DBSCAN** | `DBSCAN` | Formes arbitraires, détecte les outliers |
| **Agglomerative** | `AgglomerativeClustering` | Hiérarchique, dendrogramme |

---

## Datasets intégrés

| Nom | Tâche | Lignes | Features | Description |
|---|---|---|---|---|
| `iris` | Classification | 150 | 4 | 3 espèces de fleurs — dataset classique |
| `wine` | Classification | 178 | 13 | Cépages de vins italiens par analyse chimique |
| `breast_cancer` | Classification | 569 | 30 | Diagnostic tumeur maligne/bénigne |
| `diabetes` | Régression | 442 | 10 | Progression de la maladie un an après baseline |
| `california_housing` | Régression | 20 640 | 8 | Prix médians des maisons en Californie |
| `titanic` | Classification | 891 | 7 | Survie des passagers (données synthétiques réalistes) |
| `ecommerce` | Classification | 1 000 | 8 | Satisfaction client (données synthétiques) |

---

## Utilisation programmatique

DataLabSn peut être utilisé entièrement en Python, sans le dashboard.

### Exemple complet — Classification

```python
from datalabsn.pipeline import DataIngestion, DataPreprocessor, FeatureEngineer
from datalabsn.models import ModelTrainer
from datalabsn.models.predictor import Predictor
from datalabsn.evaluation import Evaluator

# ── 1. Chargement ────────────────────────────────────────────────────────────
ingestion = DataIngestion()
X, y, task = ingestion.load_sample("breast_cancer")   # ou load_file("data.csv")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(
    X,
    num_impute="median",    # mean | median | knn | most_frequent
    scaler="standard",      # standard | minmax | robust | none
    encode="onehot",        # onehot | label | ordinal
    drop_duplicates=True,
)

# ── 3. Sélection de features ─────────────────────────────────────────────────
engineer = FeatureEngineer()
X_selected, scores = engineer.select_k_best(X_processed, y, k=15, task=task)

# ── 4. Entraînement ──────────────────────────────────────────────────────────
trainer = ModelTrainer(random_state=42, test_size=0.2, cv_folds=5)
result = trainer.train(
    X_selected, y,
    model_name="Random Forest",
    task=task,
    tune_hyperparams=True,
    search_strategy="random",
    n_iter=30,
)

print(f"Score test : {result.test_score:.4f}")
print(f"CV moyen   : {result.cv_mean:.4f} ± {result.cv_std:.4f}")
print(f"Durée      : {result.training_time:.2f}s")

# ── 5. Évaluation ────────────────────────────────────────────────────────────
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(
    result.y_test, result.y_pred, result.y_proba
)
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"F1-Score : {metrics['f1_score']:.4f}")
print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

# ── 6. Sauvegarde et prédiction ───────────────────────────────────────────────
predictor = Predictor(model=result.model, feature_names=result.feature_names)
predictor.save("models/saved/breast_cancer_rf.joblib")

# Prédire de nouvelles données
import pandas as pd
new_data = pd.DataFrame([X_selected.iloc[0].to_dict()])
prediction = predictor.predict(new_data)
confidence = predictor.predict_proba(new_data).max()
print(f"Prédiction : {prediction[0]}  (confiance : {confidence:.1%})")
```

### Comparer tous les modèles

```python
from datalabsn.pipeline import DataIngestion, DataPreprocessor
from datalabsn.models import ModelTrainer

ingestion = DataIngestion()
X, y, task = ingestion.load_sample("wine")

preprocessor = DataPreprocessor()
X_proc = preprocessor.fit_transform(X, scaler="standard")

trainer = ModelTrainer(random_state=42)
df = trainer.compare_models(X_proc, y, task)
print(df.to_string(index=False))
```

```
          Modèle  Score Train  Score Test  CV Moyen  CV Std  Temps (s)  Surapprentissage
Gradient Boosting       1.0000      0.9722    0.9651  0.0284      0.125            0.0349
    Random Forest       1.0000      0.9722    0.9566  0.0377      0.098            0.0434
              SVM       0.9944      0.9722    0.9777  0.0215      0.006            0.0222
              ...
```

### Charger votre propre fichier

```python
from datalabsn.pipeline import DataIngestion

ingestion = DataIngestion()

# CSV
X, y = ingestion.load_file("mon_dataset.csv", target_col="label")

# Excel
X, y = ingestion.load_file("donnees.xlsx", target_col="cible")

# Parquet
X, y = ingestion.load_file("data.parquet", target_col="target")
```

---

## Tests

### Lancer la suite complète

```bash
pytest tests/ -v
```

### Avec couverture de code

```bash
pytest tests/ --cov=src/datalabsn --cov-report=html
# Ouvrir htmlcov/index.html pour le rapport détaillé
```

### Par module

```bash
pytest tests/test_pipeline.py   -v   # Pipeline : 20 tests
pytest tests/test_models.py     -v   # Modèles  : 16 tests
pytest tests/test_evaluation.py -v   # Éval     : 9 tests
```

### Résultats actuels

```
45 passed in 27.84s

Module                              Stmts  Cover
────────────────────────────────────────────────
datalabsn.pipeline.preprocessing        89    87%
datalabsn.pipeline.validator            55    93%
datalabsn.models.registry               26   100%
datalabsn.models.trainer                79    86%
datalabsn.evaluation.metrics            40    80%
────────────────────────────────────────────────
TOTAL                                514    80%
```

---

## Scripts CLI

### Générer les datasets d'exemple

```bash
python scripts/generate_sample_data.py
```

Crée des fichiers CSV dans `data/samples/` pour chaque dataset intégré.

### Benchmark des modèles

```bash
python scripts/train_models.py
```

Lance un benchmark complet sur Iris, Wine, Breast Cancer et Diabetes, affiche le classement des modèles pour chaque dataset.

---

## Configuration

### Via YAML (`configs/default.yaml`)

```yaml
data_dir: data
models_dir: models/saved
random_state: 42
test_size: 0.2
cv_folds: 5
n_jobs: -1
```

### Via variables d'environnement

```bash
export DATALAB_RANDOM_STATE=42
export DATALAB_TEST_SIZE=0.2
export DATALAB_CV_FOLDS=5
```

### Via code Python

```python
from datalabsn.utils import Config

cfg = Config.from_yaml("configs/default.yaml")
# ou
cfg = Config.from_env()
```

---

## Documentation

La documentation complète est disponible dans le dossier [`docs/`](docs/) :

| Document | Description |
|---|---|
| [Guide d'installation](docs/installation.md) | Installation détaillée, environnements virtuels, Docker |
| [Guide utilisateur — Data Explorer](docs/user_guide/data_explorer.md) | Exploration des données pas à pas |
| [Guide utilisateur — Pipeline](docs/user_guide/pipeline.md) | Preprocessing : stratégies et recommandations |
| [Guide utilisateur — Entraînement](docs/user_guide/model_training.md) | Choix du modèle, tuning, comparaison |
| [Guide utilisateur — Évaluation](docs/user_guide/evaluation.md) | Interprétation des métriques et graphiques |
| [Guide utilisateur — Prédictions](docs/user_guide/predictions.md) | Prédictions manuelles et en lot |
| [Référence API — Pipeline](docs/api/pipeline.md) | DataIngestion, DataPreprocessor, FeatureEngineer, DataValidator |
| [Référence API — Modèles](docs/api/models.md) | ModelRegistry, ModelTrainer, Predictor |
| [Référence API — Évaluation](docs/api/evaluation.md) | Evaluator |
| [Exemple — Démarrage rapide](docs/examples/quick_start.md) | Workflow complet en 30 lignes |
| [Exemple — Pipeline personnalisé](docs/examples/custom_pipeline.md) | Pipeline avancé avec feature engineering |
| [Guide de contribution](CONTRIBUTING.md) | Comment contribuer au projet |

---

## Formats de données supportés

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Séparateur configurable |
| TSV | `.tsv` | Tab-separated automatique |
| Excel | `.xlsx`, `.xls` | Feuille active par défaut |
| Parquet | `.parquet` | Recommandé pour les gros volumes |
| JSON | `.json` | Format orienté colonnes |

---

## Stack technique

| Composant | Bibliothèque | Version |
|---|---|---|
| Langage | Python | 3.10+ |
| Traitement de données | Pandas | 2.0+ |
| Calcul numérique | NumPy + SciPy | 1.24+ / 1.10+ |
| Machine Learning | scikit-learn | 1.3+ |
| Dashboard | Streamlit | 1.28+ |
| Visualisation | Plotly + Seaborn + Matplotlib | 5.15+ / 0.12+ / 3.7+ |
| Sérialisation | Joblib | 1.3+ |
| Validation de données | Pydantic | 2.0+ |
| Configuration | PyYAML | 6.0+ |
| Formats fichiers | openpyxl + pyarrow | 3.1+ / 12.0+ |
| Tests | pytest + pytest-cov | 7.4+ / 4.1+ |

---

## Contribuer

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines détaillées.

```bash
# Fork → clone → branche
git checkout -b feature/ma-fonctionnalite

# Développer + tester
pytest tests/ -v

# Committer et ouvrir une PR
git commit -m "feat: description de la fonctionnalité"
git push origin feature/ma-fonctionnalite
```

Types de contributions appréciées :
- Nouveaux algorithmes ML (XGBoost, LightGBM, etc.)
- Nouvelles visualisations dans le dashboard
- Nouveaux datasets intégrés
- Amélioration de la couverture de tests
- Traductions de la documentation

---

## Feuille de route

- [ ] Support XGBoost / LightGBM / CatBoost
- [ ] Détection automatique du type de tâche (classification vs régression)
- [ ] Export de rapport PDF complet
- [ ] Déploiement Docker + docker-compose
- [ ] Support des séries temporelles (ARIMA, Prophet)
- [ ] Mode multi-utilisateur avec historique des expériences
- [ ] Intégration MLflow pour le tracking des expériences

---

## Licence

Distribué sous licence **MIT**. Voir [LICENSE](LICENSE) pour plus d'informations.

---

<div align="center">

Construit avec ❤️ · Python 3.10+ · Streamlit · scikit-learn · Pandas

</div>
