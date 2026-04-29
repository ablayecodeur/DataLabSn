# Guide utilisateur — Entraînement des modèles

La page **🤖 Model Training** permet d'entraîner un ou plusieurs algorithmes de machine learning sur vos données prétraitées.

---

## Modes disponibles

### Modèle unique

Entraîne un seul algorithme avec les paramètres de votre choix. Idéal pour :
- Explorer un algorithme en profondeur
- Optimiser les hyperparamètres
- Analyser l'importance des features

### Comparaison de modèles

Entraîne plusieurs algorithmes avec les mêmes données et paramètres, et les classe par performance CV. Idéal pour :
- Identifier le meilleur algorithme pour vos données
- Obtenir une vue d'ensemble rapide
- Justifier le choix d'un modèle

---

## Paramètres communs

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| **Taille du set de test** | 20% | Proportion des données réservée à l'évaluation finale |
| **Folds de validation croisée** | 5 | Nombre de partitions pour la CV — plus élevé = estimation plus stable mais plus lent |
| **Random state** | 42 | Graine aléatoire pour la reproductibilité |

### Recommandations

- **Test size 20%** convient pour la plupart des datasets. Utilisez 30% si vous avez beaucoup de données, 15% si vous en avez peu.
- **CV = 5** est le standard. Utilisez CV = 10 pour une estimation plus fiable sur les petits datasets.
- **Toujours fixer le random_state** pour reproduire vos expériences.

---

## Optimisation des hyperparamètres

Activez la case **Optimiser les hyperparamètres** pour lancer une recherche automatique.

### Grid Search

Teste toutes les combinaisons de la grille définie. Garantit le meilleur résultat sur la grille, mais peut être lent si la grille est grande.

```
Grille Random Forest :
  n_estimators : [50, 100, 200]
  max_depth    : [None, 5, 10, 20]
  → 12 combinaisons × 5 folds = 60 entraînements
```

### Random Search

Tire aléatoirement `n_iter` combinaisons de la grille. Plus rapide que Grid Search pour les grandes grilles, souvent presque aussi bon.

**Recommandation :** utilisez Random Search avec `n_iter = 20-50` pour une première exploration, puis Grid Search focalisé sur la zone prometteuse.

---

## Choisir le bon algorithme

### Pour la classification

| Situation | Algorithme recommandé |
|---|---|
| Première baseline rapide | Logistic Regression |
| Beaucoup de features, peu de tuning | Random Forest |
| Maximiser la performance | Gradient Boosting |
| Données très non-linéaires | SVM (kernel RBF) |
| Interprétabilité maximale | Decision Tree |
| Très peu de données | Naive Bayes |

### Pour la régression

| Situation | Algorithme recommandé |
|---|---|
| Baseline interprétable | Linear Regression |
| Régularisation, éviter l'overfitting | Ridge ou Lasso |
| Sélection automatique de features | Lasso |
| Features corrélées | ElasticNet |
| Performance maximale | Gradient Boosting |
| Robustesse aux outliers | SVR |

### Signaux d'alerte

| Signal | Interprétation | Action |
|---|---|---|
| `Score Train ≈ 1.0` mais `Score Test << Score Train` | Surapprentissage (overfitting) | Régulariser (Ridge/Lasso), réduire `max_depth`, augmenter `min_samples_split` |
| `Score Train ≈ Score Test` mais les deux sont bas | Sous-apprentissage (underfitting) | Modèle plus complexe, plus de features, plus de données |
| `CV Std > 0.05` | Grande variance — modèle instable | Augmenter les folds CV, régulariser |

---

## Interpréter les résultats

### Métriques affichées

- **Score Train** : performance sur les données d'entraînement
- **Score Test** : performance sur les données non vues — métrique principale
- **CV Moyen ± Std** : estimation par validation croisée — plus fiable que le score test seul
- **Surapprentissage** = Score Train − Score Test. Une valeur > 0.05 mérite attention.

### Graphique des folds CV

Chaque barre représente le score d'un fold. Une grande variabilité entre les folds (barres de hauteurs très différentes) indique que le modèle est sensible à la partition des données.

### Importance des features

Affichée pour les modèles qui la calculent :
- **Arbres** (Random Forest, Gradient Boosting, Decision Tree) : impurity-based feature importance
- **Modèles linéaires** (Logistic Regression, Ridge, etc.) : valeur absolue des coefficients

> L'importance des features peut être utilisée pour affiner la sélection dans le Pipeline.

---

## Après l'entraînement

Le modèle entraîné est stocké en session et devient automatiquement disponible dans les pages **📈 Évaluation** et **🔮 Prédictions**. Vous n'avez pas besoin de le sauvegarder manuellement pour l'utiliser dans ces pages.

Pour persister le modèle entre les sessions, sauvegardez-le depuis la page Prédictions.
