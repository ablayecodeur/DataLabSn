# Guide de contribution

Merci de l'intérêt que vous portez à DataLabSn ! Ce guide explique comment contribuer efficacement au projet.

---

## Avant de commencer

- Consultez les [issues ouvertes](https://github.com/ablayecodeur/DataLabSn/issues) pour éviter les doublons
- Pour une nouvelle fonctionnalité, ouvrez d'abord une issue pour en discuter
- Pour un bug fix, vous pouvez directement ouvrir une Pull Request

---

## Mise en place de l'environnement de développement

```bash
# 1. Forker et cloner
git clone https://github.com/votre-fork/DataLabSn.git
cd DataLabSn

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# 3. Installer en mode développement
pip install -r requirements.txt
pip install pytest pytest-cov

# 4. Vérifier que les tests passent
pytest tests/ -v
```

---

## Workflow de contribution

```bash
# 1. Créer une branche descriptive
git checkout -b feat/xgboost-support
# ou
git checkout -b fix/knn-imputer-memory

# 2. Développer et tester
pytest tests/ -v

# 3. Committer avec un message clair (convention Conventional Commits)
git commit -m "feat: ajout du support XGBoost dans ModelRegistry"

# 4. Pousser et ouvrir une PR
git push origin feat/xgboost-support
```

---

## Convention des messages de commit

Nous utilisons [Conventional Commits](https://www.conventionalcommits.org/) :

| Type | Quand l'utiliser |
|---|---|
| `feat:` | Nouvelle fonctionnalité |
| `fix:` | Correction de bug |
| `docs:` | Modification de documentation uniquement |
| `test:` | Ajout ou modification de tests |
| `refactor:` | Refactoring sans changement de comportement |
| `perf:` | Amélioration de performance |
| `chore:` | Maintenance (dépendances, config CI, etc.) |

**Exemples :**
```
feat: ajouter LightGBM dans ModelRegistry
fix: corriger l'alignement d'index après drop_duplicates dans DataPreprocessor
docs: ajouter exemple de régression dans quick_start.md
test: ajouter tests pour evaluate_clustering avec labels négatifs (DBSCAN)
```

---

## Standards de code

### Style
- **PEP 8** — utilisez `ruff` ou `flake8` pour la vérification
- **Type hints** sur toutes les signatures publiques
- **Docstrings** pour les classes et méthodes publiques (format Google)

### Tests
- Tout nouveau code doit avoir des tests dans `tests/`
- Les tests doivent passer sans erreur : `pytest tests/ -v`
- La couverture ne doit pas descendre sous 75% : `pytest --cov=src/datalabsn`
- Utilisez des fixtures pytest pour les données partagées

### Exemple de test bien structuré

```python
class TestMaNouvelleFeature:
    def test_cas_nominal(self, sample_df):
        """Décrit ce qui est testé."""
        result = ma_fonction(sample_df)
        assert result.shape[0] == len(sample_df)

    def test_cas_limite_vide(self):
        with pytest.raises(ValueError, match="message attendu"):
            ma_fonction(pd.DataFrame())

    def test_coherence_fit_transform(self, sample_df):
        """fit_transform et transform doivent donner le même résultat."""
        obj = MaClasse()
        r1 = obj.fit_transform(sample_df)
        r2 = obj.transform(sample_df)
        pd.testing.assert_frame_equal(r1, r2)
```

---

## Ajouter un nouvel algorithme ML

1. Ouvrez `src/datalabsn/models/registry.py`
2. Ajoutez l'entrée dans `CLASSIFIERS` ou `REGRESSORS` :

```python
"XGBoost": {
    "model": XGBClassifier,
    "params": {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss"},
    "tuning": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "description": "Gradient boosting optimisé, souvent le plus performant.",
},
```

3. Ajoutez la dépendance dans `requirements.txt` : `xgboost>=2.0.0`
4. Ajoutez un test dans `tests/test_models.py` :

```python
def test_xgboost_instantiate(self):
    model = ModelRegistry.get("classification", "XGBoost")
    assert hasattr(model, "fit")
```

---

## Ajouter un nouveau dataset intégré

1. Ouvrez `src/datalabsn/pipeline/ingestion.py`
2. Ajoutez une entrée dans `SAMPLE_DATASETS` et une méthode `_load_*`
3. Référencez la méthode dans le `dict` de `load_sample()`
4. Ajoutez un test dans `tests/test_pipeline.py`

---

## Checklist avant d'ouvrir une PR

- [ ] Les tests passent : `pytest tests/ -v`
- [ ] La couverture est stable : `pytest --cov=src/datalabsn`
- [ ] Le code suit PEP 8
- [ ] Les nouvelles fonctions/classes ont des type hints
- [ ] La documentation est mise à jour si nécessaire
- [ ] Le message de commit respecte la convention
- [ ] La PR décrit clairement le changement et pourquoi

---

## Questions ?

Ouvrez une [Discussion GitHub](https://github.com/ablayecodeur/DataLabSn/discussions) ou une issue avec le label `question`.
