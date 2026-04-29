# Guide utilisateur — Pipeline de traitement

La page **🔧 Pipeline** applique une séquence de transformations sur vos données avant la modélisation. Toutes les transformations sont ajustées sur les données d'entraînement et peuvent être réappliquées sur de nouvelles données via `preprocessor.transform()`.

---

## Étape 1 — Preprocessing

### Imputation des valeurs manquantes

Les valeurs manquantes doivent être traitées avant l'entraînement, car la plupart des algorithmes sklearn ne les acceptent pas.

| Stratégie | Quand l'utiliser |
|---|---|
| `median` *(recommandé)* | Variables numériques avec outliers ou distribution asymétrique |
| `mean` | Variables numériques avec distribution proche de la normale |
| `most_frequent` | Variables catégorielles, ou numériques à faible cardinalité |
| `knn` | Quand les valeurs manquantes ont une structure spatiale — plus précis mais lent |

> **Règle pratique** : utilisez `median` par défaut pour le numérique et `most_frequent` pour le catégoriel.

### Encodage des variables catégorielles

Les algorithmes ML travaillent avec des nombres. Les variables textuelles doivent être converties.

| Encodage | Quand l'utiliser | Avantage | Inconvénient |
|---|---|---|---|
| `onehot` *(recommandé)* | Variables nominales (pas d'ordre) | Pas de faux ordre introduit | Augmente la dimensionnalité |
| `label` | Variables ordinales, ou arbres de décision | Compact (1 colonne par variable) | Introduit un ordre arbitraire pour les modèles linéaires |
| `ordinal` | Variables avec un ordre explicite (ex. low/medium/high) | Préserve l'ordre sémantique | Requiert que l'ordre soit correct |

> Exemple : `Sex = [male, female]` → OneHot donne `Sex_male` et `Sex_female`. LabelEncoder donne `Sex = [0, 1]`.

### Normalisation

Les algorithmes basés sur la distance (KNN, SVM) et les modèles linéaires sont sensibles à l'échelle des features.

| Scaler | Formule | Quand l'utiliser |
|---|---|---|
| `standard` *(recommandé)* | `(x - μ) / σ` | Distribution approximativement normale |
| `minmax` | `(x - min) / (max - min)` | Bornes connues, réseaux de neurones |
| `robust` | `(x - médiane) / IQR` | Données avec outliers importants |
| `none` | Aucune transformation | Arbres de décision (insensibles à l'échelle) |

> Les arbres (Random Forest, Gradient Boosting, Decision Tree) n'ont pas besoin de normalisation. Pour tous les autres algorithmes, utilisez `standard`.

---

## Étape 2 — Feature Engineering

### Sélection de features (SelectKBest)

Sélectionne les `k` features les plus informatives par rapport à la variable cible.

| Fonction de score | Tâche | Description |
|---|---|---|
| `f_test` | Classification / Régression | Test ANOVA F — relations linéaires |
| `mutual_info` | Classification / Régression | Information mutuelle — relations non-linéaires |

**Quand l'utiliser :**
- Données à haute dimensionnalité (> 50 features)
- Améliorer la vitesse d'entraînement
- Réduire le risque de surapprentissage

### Réduction dimensionnelle (PCA)

L'Analyse en Composantes Principales projette les données dans un espace de dimension réduite en maximisant la variance expliquée.

**Paramètre `variance cible` :** pourcentage de variance à conserver. `95%` est un bon compromis.

**Quand l'utiliser :**
- Visualisation 2D/3D de données multi-dimensionnelles
- Features très corrélées (multicolinéarité)
- Réduction de bruit

> ⚠️ La PCA rend les features moins interprétables (PC1, PC2… au lieu des noms originaux). À éviter si l'interprétabilité est critique.

### Suppression des features corrélées

Supprime automatiquement une des deux features dont la corrélation de Pearson dépasse le seuil (0.95 par défaut).

**Pourquoi :** des features quasi-identiques n'apportent pas d'information supplémentaire mais augmentent le bruit et la multicolinéarité pour les modèles linéaires.

---

## Ordre d'application

Le pipeline applique les transformations dans cet ordre :

```
1. Suppression des doublons
2. Imputation numérique
3. Imputation catégorielle
4. Encodage des catégorielles
5. Suppression des features corrélées  (si activé)
6. Sélection SelectKBest              (si activé)
7. PCA                                (si activé)
8. Normalisation
```

---

## Exemple de configuration recommandée par scénario

### Dataset généraliste (point de départ)
```
Imputation numérique  : median
Imputation catégorielle : most_frequent
Encodage              : onehot
Normalisation         : standard
```

### Données avec outliers (ex. salaires, prix)
```
Imputation numérique  : median
Normalisation         : robust
```

### Arbres de décision / Random Forest
```
Encodage              : label  (ou onehot, les deux fonctionnent)
Normalisation         : none   (les arbres sont invariants à l'échelle)
```

### Haute dimensionnalité (> 50 features)
```
Sélection features    : activé, k = 20
Suppression corrélées : activé
```

---

## Réinitialiser le pipeline

Cliquez **Réinitialiser le pipeline** pour supprimer les données traitées et recommencer avec les données brutes. Les paramètres du formulaire sont conservés.
