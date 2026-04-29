# Documentation DataLab

Bienvenue dans la documentation complète de **DataLab**, une plateforme d'analyse de données et de machine learning basée sur Pandas, scikit-learn et Streamlit.

---

## Navigation

### Démarrage

| Document | Description |
|---|---|
| [Installation](installation.md) | Installation, environnements virtuels, résolution de problèmes |
| [Démarrage rapide](examples/quick_start.md) | Workflow complet en 30 lignes de code |

### Guide utilisateur (Dashboard)

| Page | Document |
|---|---|
| 📊 Data Explorer | [user_guide/data_explorer.md](user_guide/data_explorer.md) |
| 🔧 Pipeline | [user_guide/pipeline.md](user_guide/pipeline.md) |
| 🤖 Model Training | [user_guide/model_training.md](user_guide/model_training.md) |
| 📈 Evaluation | [user_guide/evaluation.md](user_guide/evaluation.md) |
| 🔮 Predictions | [user_guide/predictions.md](user_guide/predictions.md) |

### Référence API (Python)

| Module | Document |
|---|---|
| `datalab.pipeline` | [api/pipeline.md](api/pipeline.md) |
| `datalab.models` | [api/models.md](api/models.md) |
| `datalab.evaluation` | [api/evaluation.md](api/evaluation.md) |

### Exemples

| Exemple | Document |
|---|---|
| Workflow complet | [examples/quick_start.md](examples/quick_start.md) |
| Pipeline avancé (Titanic) | [examples/custom_pipeline.md](examples/custom_pipeline.md) |

---

## Architecture en 30 secondes

```
                   ┌─────────────────────────────────────┐
                   │         Dashboard Streamlit          │
                   │  (5 pages multimodales interactives) │
                   └──────────────┬──────────────────────┘
                                  │ utilise
                   ┌──────────────▼──────────────────────┐
                   │          src/datalab/               │
                   │                                     │
                   │  pipeline/     models/    evaluation/│
                   │  ─────────     ───────    ──────────│
                   │  ingestion     registry   metrics   │
                   │  preprocess    trainer              │
                   │  features      predictor            │
                   │  validator                          │
                   └─────────────────────────────────────┘
```

DataLab suit une architecture **pipeline → modèle → évaluation** où chaque composant est indépendant et réutilisable directement en Python, sans passer par le dashboard.

---

## Flux de données typique

```
Fichier CSV / Dataset intégré
         │
         ▼
   DataIngestion.load_file()
   DataIngestion.load_sample()
         │
         ▼  (X: DataFrame, y: Series)
   DataValidator.validate()        ← rapport de qualité
         │
         ▼
   DataPreprocessor.fit_transform() ← imputation, encodage, scaling
         │
         ▼
   FeatureEngineer.select_k_best()  ← sélection optionnelle
         │
         ▼  (X_processed)
   ModelTrainer.train()             ← split, CV, optimisation
         │
         ▼  (TrainingResult)
   Evaluator.evaluate_*()           ← métriques, courbes
         │
         ▼
   Predictor.predict()              ← nouvelles données
   Predictor.save()                 ← persistance .joblib
```

---

## Conventions

- Toutes les classes sont importables depuis `datalab.pipeline`, `datalab.models`, `datalab.evaluation`
- `fit_transform()` ajuste sur les données courantes ; `transform()` réapplique les paramètres appris
- `TrainingResult` contient tout ce qui est nécessaire pour l'évaluation et les prédictions
- Les logs sont écrits dans `logs/datalab.log` et sur `stdout`
