# 🔬 DataLab

> Plateforme d'analyse de données et de machine learning avec dashboard interactif

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green?logo=pandas)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

DataLab est une plateforme complète permettant de charger, explorer, prétraiter des données, entraîner des modèles de machine learning et visualiser les résultats — le tout depuis un dashboard Streamlit intuitif.

---

## ✨ Fonctionnalités

| Module | Description |
|--------|-------------|
| **📊 Data Explorer** | Chargement CSV/Excel/Parquet/JSON, statistiques descriptives, distributions, corrélations, analyse des valeurs manquantes |
| **🔧 Pipeline** | Imputation (mean/median/KNN), encodage (OneHot/Label/Ordinal), normalisation (Standard/MinMax/Robust), sélection de features |
| **🤖 Entraînement** | 7 classifieurs + 9 régresseurs + 3 algorithmes de clustering, comparaison côte-à-côte, optimisation d'hyperparamètres |
| **📈 Évaluation** | Matrice de confusion, courbes ROC, Precision-Recall, courbes d'apprentissage, résidus |
| **🔮 Prédictions** | Saisie manuelle ou prédictions en lot (CSV), scores de confiance, export des résultats |

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/DataLab.git
cd DataLab
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou
.venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

Ouvrez [http://localhost:8501](http://localhost:8501) dans votre navigateur.

---

## 📁 Structure du projet

```
DataLab/
├── src/datalab/
│   ├── pipeline/
│   │   ├── ingestion.py       # Chargement de données (fichiers + datasets intégrés)
│   │   ├── preprocessing.py   # Imputation, encodage, normalisation
│   │   ├── features.py        # Sélection de features, PCA
│   │   └── validator.py       # Validation de la qualité des données
│   ├── models/
│   │   ├── registry.py        # Catalogue de tous les algorithmes disponibles
│   │   ├── trainer.py         # Entraînement, validation croisée, comparaison
│   │   └── predictor.py       # Prédiction, sauvegarde/chargement de modèles
│   └── evaluation/
│       └── metrics.py         # Métriques de classification, régression, clustering
├── dashboard/
│   ├── app.py                 # Page d'accueil Streamlit
│   └── pages/
│       ├── 1_📊_Data_Explorer.py
│       ├── 2_🔧_Pipeline.py
│       ├── 3_🤖_Model_Training.py
│       ├── 4_📈_Evaluation.py
│       └── 5_🔮_Predictions.py
├── data/
│   ├── samples/               # Datasets d'exemple générés
│   └── raw/                   # Vos données brutes (gitignorées)
├── scripts/
│   ├── generate_sample_data.py
│   └── train_models.py        # Benchmark CLI
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_evaluation.py
└── configs/
    └── default.yaml
```

---

## 🤖 Algorithmes disponibles

### Classification (7 modèles)
| Algorithme | Description |
|------------|-------------|
| Logistic Regression | Modèle linéaire, rapide et interprétable |
| Random Forest | Ensemble d'arbres, robuste et précis |
| Gradient Boosting | Boosting séquentiel, excellent sur données tabulaires |
| SVM | Marge maximale, haute dimension |
| Decision Tree | Très interprétable |
| K-Nearest Neighbors | Non-paramétrique |
| Naive Bayes | Probabiliste, très rapide |

### Régression (9 modèles)
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR, KNN, Decision Tree

### Clustering (3 algorithmes)
K-Means, DBSCAN, Agglomerative Clustering

---

## 📊 Datasets intégrés

| Dataset | Type | Lignes | Features | Description |
|---------|------|--------|----------|-------------|
| `iris` | Classification | 150 | 4 | Espèces de fleurs d'iris |
| `wine` | Classification | 178 | 13 | Qualité de vin |
| `breast_cancer` | Classification | 569 | 30 | Diagnostic cancer du sein |
| `diabetes` | Régression | 442 | 10 | Progression du diabète |
| `california_housing` | Régression | 20 640 | 8 | Prix immobiliers Californie |
| `titanic` | Classification | 891 | 7 | Survie au Titanic |
| `ecommerce` | Classification | 1 000 | 8 | Satisfaction client e-commerce |

---

## 🛠️ Utilisation en ligne de commande

### Générer les datasets d'exemple
```bash
python scripts/generate_sample_data.py
```

### Benchmark des modèles
```bash
python scripts/train_models.py
```

### Lancer les tests
```bash
pytest tests/ -v --cov=src/datalab
```

---

## 📖 Utilisation programmatique

```python
from datalab.pipeline import DataIngestion, DataPreprocessor
from datalab.models import ModelTrainer
from datalab.evaluation import Evaluator

# Charger un dataset
ingestion = DataIngestion()
X, y, task = ingestion.load_sample("iris")

# Prétraitement
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X, scaler="standard", encode="onehot")

# Entraînement
trainer = ModelTrainer(random_state=42, test_size=0.2, cv_folds=5)
result = trainer.train(X_processed, y, "Random Forest", task)

print(f"Score test : {result.test_score:.4f}")
print(f"CV moyen   : {result.cv_mean:.4f} ± {result.cv_std:.4f}")

# Comparer tous les modèles
df = trainer.compare_models(X_processed, y, task)
print(df.to_string(index=False))

# Évaluation
evaluator = Evaluator()
metrics = evaluator.evaluate_classification(result.y_test, result.y_pred, result.y_proba)
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
```

---

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=src/datalab --cov-report=html

# Un module spécifique
pytest tests/test_models.py -v
```

---

## ⚙️ Configuration

Modifiez `configs/default.yaml` ou définissez des variables d'environnement :

```bash
export DATALAB_RANDOM_STATE=42
export DATALAB_TEST_SIZE=0.2
export DATALAB_CV_FOLDS=5
```

---

## 🖥️ Captures d'écran

| Page | Description |
|------|-------------|
| 🏠 Accueil | Vue d'ensemble de la plateforme |
| 📊 Data Explorer | Exploration interactive avec 6 onglets |
| 🔧 Pipeline | Configuration visuelle du preprocessing |
| 🤖 Entraînement | Sélection, entraînement et comparaison |
| 📈 Évaluation | Métriques, courbes ROC, matrices de confusion |
| 🔮 Prédictions | Saisie manuelle ou upload en lot |

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -m 'Ajout nouvelle fonctionnalite'`)
4. Push sur la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

---

## 📄 Licence

Distribué sous licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.

---

## 🏗️ Stack technique

- **[Pandas 2.x](https://pandas.pydata.org/)** — Manipulation et analyse de données
- **[NumPy](https://numpy.org/)** — Calcul numérique
- **[Scikit-learn 1.3+](https://scikit-learn.org/)** — Algorithmes de machine learning
- **[Streamlit 1.28+](https://streamlit.io/)** — Dashboard interactif
- **[Plotly](https://plotly.com/python/)** — Visualisations interactives
- **[Joblib](https://joblib.readthedocs.io/)** — Sérialisation des modèles
