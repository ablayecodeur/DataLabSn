# Guide utilisateur — Data Explorer

La page **📊 Data Explorer** est le point d'entrée de tout workflow DataLab. Elle permet de charger et d'explorer en profondeur n'importe quel dataset avant de passer au preprocessing.

---

## Charger des données

### Depuis un dataset intégré

Sélectionnez un dataset dans la liste déroulante, puis cliquez **Charger le dataset**. Le dataset est immédiatement disponible pour toutes les autres pages.

| Dataset | Cas d'usage recommandé |
|---|---|
| `iris` | Découverte de la classification, 3 classes équilibrées |
| `wine` | Classification avec features numériques continues |
| `breast_cancer` | Classification binaire, beaucoup de features |
| `diabetes` | Régression, cible continue |
| `california_housing` | Régression sur gros volume |
| `titanic` | Classification avec valeurs manquantes et variables mixtes |
| `ecommerce` | Classification métier, données synthétiques réalistes |

### Depuis un fichier local

Formats acceptés : **CSV, TSV, Excel (.xlsx/.xls), Parquet, JSON**.

Options disponibles :
- **Colonne cible** : indiquez le nom de la colonne `y` — elle sera séparée de `X`
- **Séparateur** : pour les CSV non-standards (`;`, `|`, `\t`)

> Si vous ne spécifiez pas de colonne cible, le dataset est chargé entièrement dans `X` et vous pourrez la désigner plus tard.

---

## Onglets d'exploration

### Vue d'ensemble

Affiche :
- **Métriques de base** : nombre de lignes, colonnes, valeurs manquantes, mémoire utilisée
- **Aperçu** : les N premières lignes (configurable 5–50), avec la colonne cible surlignée
- **Types de données** : type pandas, cardinalité, taux de valeurs manquantes par colonne
- **Statistiques descriptives** : count, mean, std, min, quartiles, max pour toutes les colonnes numériques

### Distributions

Visualisez la distribution de chaque variable :

- **Variables numériques** : histogramme + boxplot en marginal, coloré par classe cible si disponible
- **Variables catégorielles** : bar chart des effectifs par modalité

Sélectionnez les colonnes à afficher via le multiselect pour ne visualiser que ce qui vous intéresse.

### Corrélations

Matrice de corrélation interactive avec trois méthodes :

| Méthode | Quand l'utiliser |
|---|---|
| **Pearson** | Relations linéaires entre variables continues |
| **Spearman** | Relations monotones, robuste aux outliers |
| **Kendall** | Petit échantillon, beaucoup d'ex-aequo |

Le tableau **Paires les plus corrélées** liste les 10 couples de features avec la corrélation absolue la plus forte — utile pour détecter la multicolinéarité.

### Valeurs manquantes

Affiche un bar chart du taux de valeurs manquantes par colonne (en %) et un tableau récapitulatif. Les colonnes sans valeurs manquantes ne sont pas affichées.

> Un taux > 50% est signalé comme avertissement dans le rapport de validation. Envisagez de supprimer ces colonnes ou d'utiliser l'imputation KNN.

### Cible

Analyse de la variable cible :

- **Classification** : pie chart + bar chart de la distribution des classes, tableau des effectifs et proportions. Un déséquilibre de classes < 5% pour la classe minoritaire est signalé.
- **Régression** : histogramme + violin plot de la distribution de la cible.

### Validation

Rapport automatique sur la qualité des données, avec trois niveaux :

- 🚫 **Erreur** : problème bloquant (trop peu de lignes, valeurs manquantes dans la cible)
- ⚠️ **Avertissement** : problème non bloquant (colonnes constantes, fort taux de manquants, déséquilibre de classes)
- ✅ **Valide** : aucun problème détecté

Métriques calculées : nombre de lignes/colonnes, taux global de manquants, doublons, taux d'outliers (z-score > 3), cardinalité de la cible.

---

## Bonnes pratiques

1. **Commencez toujours par l'onglet Validation** pour identifier rapidement les problèmes.
2. **Inspectez les corrélations** avant d'entraîner un modèle linéaire — des features très corrélées (> 0.9) peuvent déstabiliser l'estimation des coefficients.
3. **Vérifiez l'équilibre des classes** avant la classification — un fort déséquilibre peut biaiser les métriques (l'accuracy peut être trompeuse).
4. **Regardez les distributions** des features numériques — une distribution très asymétrique (longue queue) peut justifier un `RobustScaler` dans le pipeline.
