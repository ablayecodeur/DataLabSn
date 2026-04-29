# Guide utilisateur — Évaluation des modèles

La page **📈 Évaluation** affiche l'analyse complète des performances du modèle entraîné.

---

## Classification

### Métriques principales

| Métrique | Formule | Interprétation |
|---|---|---|
| **Accuracy** | `(TP + TN) / total` | Proportion de prédictions correctes. Trompeuse si les classes sont déséquilibrées. |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Moyenne harmonique precision/recall. Bonne métrique principale en cas de déséquilibre. |
| **Precision** | `TP / (TP + FP)` | Parmi les prédictions positives, quelle proportion est réellement positive. |
| **Recall** | `TP / (TP + FN)` | Parmi les vrais positifs, quelle proportion a été détectée. |
| **ROC-AUC** | Aire sous la courbe ROC | Capacité du modèle à distinguer les classes, indépendamment du seuil. 0.5 = aléatoire, 1.0 = parfait. |

### Quand utiliser quelle métrique ?

| Contexte | Métrique à privilégier |
|---|---|
| Classes équilibrées | Accuracy ou F1-Score |
| Classes déséquilibrées | F1-Score (weighted) ou ROC-AUC |
| Coût élevé des faux positifs (spam, alarmes) | Precision |
| Coût élevé des faux négatifs (maladie, fraude) | Recall |
| Comparaison de modèles | ROC-AUC |

---

### Matrice de confusion

La matrice de confusion montre pour chaque classe réelle (lignes) la répartition des prédictions (colonnes).

```
                Prédit 0   Prédit 1
Réel 0   │   Vrais Négatifs  │  Faux Positifs  │
Réel 1   │   Faux Négatifs   │  Vrais Positifs  │
```

**Comment la lire :**
- La diagonale principale = prédictions correctes
- Hors diagonale = erreurs de classification
- Une colonne ou ligne hors diagonale très fournie indique une classe difficile

---

### Courbe ROC

La courbe ROC trace le Taux de Vrais Positifs (Recall) en fonction du Taux de Faux Positifs pour tous les seuils de décision.

- **AUC = 1.0** : modèle parfait
- **AUC = 0.5** : prédiction aléatoire (diagonale)
- **AUC < 0.5** : le modèle prédit à l'envers (rare)

Pour la classification **multiclasse**, une courbe ROC est tracée par classe (One-vs-Rest).

---

### Courbe Precision-Recall

Alternative à la ROC, particulièrement utile pour les **classes très déséquilibrées**.

La baseline (modèle aléatoire) correspond à `Precision = proportion de la classe positive` dans le dataset. Un bon modèle doit être nettement au-dessus.

---

### Rapport de classification complet

Affiché dans l'onglet **Rapport complet**, il donne la precision, recall et F1-score par classe, ainsi que les moyennes macro et weighted.

---

## Régression

### Métriques principales

| Métrique | Formule | Interprétation |
|---|---|---|
| **R²** (coefficient de détermination) | `1 - SS_res / SS_tot` | Proportion de variance expliquée. 1.0 = parfait, 0 = modèle constant, < 0 = pire qu'une constante. |
| **RMSE** (Root Mean Squared Error) | `√(Σ(y - ŷ)² / n)` | Erreur quadratique moyenne, dans l'unité de la cible. Pénalise les grandes erreurs. |
| **MAE** (Mean Absolute Error) | `Σ|y - ŷ| / n` | Erreur absolue moyenne, robuste aux outliers. |
| **MAPE** (Mean Absolute Percentage Error) | `Σ|y - ŷ| / |y| × 100` | Erreur relative en %. Difficile à interpréter si y ≈ 0. |

### R² : valeurs de référence

| R² | Qualité |
|---|---|
| > 0.9 | Excellent |
| 0.7 – 0.9 | Bon |
| 0.5 – 0.7 | Acceptable |
| < 0.5 | Faible |

> R² peut être négatif si le modèle est pire que de prédire la moyenne — cela signifie généralement un problème (pipeline mal configuré, données mal préparées).

---

### Réel vs Prédit

Nuage de points où chaque point est une observation du set de test. La droite diagonale représente la prédiction parfaite (ŷ = y).

**Interprétation :**
- Points proches de la diagonale = bonnes prédictions
- Nuage systématiquement au-dessus/dessous = biais
- Dispersion croissante = hétéroscédasticité (l'erreur augmente avec la valeur prédite)

---

### Analyse des résidus

Les résidus = valeurs réelles − valeurs prédites.

**Un bon modèle doit avoir des résidus :**
- Centrés autour de zéro (pas de biais systématique)
- Distribués uniformément (homoscédasticité)
- Sans structure visible (pas de pattern non capturé)

**Signaux d'alerte :**
- Résidus en entonnoir → hétéroscédasticité → envisager une transformation log(y)
- Résidus en arc → relation non linéaire non capturée → modèle plus complexe ou features polynomiales
- Valeurs aberrantes de résidus → outliers dans le set de test

---

## Courbes d'apprentissage

Les courbes d'apprentissage montrent l'évolution du score d'entraînement et de validation en fonction de la taille du dataset utilisé.

### Diagnostic visuel

**Surapprentissage (Overfitting) :**
```
Score Train : ████████████ 0.98  (très haut)
Score Valid : ████         0.75  (bas)
→ Grand écart entre les deux courbes
```
→ Régularisation, plus de données, réduction de complexité

**Sous-apprentissage (Underfitting) :**
```
Score Train : ████         0.72  (bas)
Score Valid : ████         0.70  (bas, proche du train)
→ Les deux courbes convergent vers une valeur basse
```
→ Modèle plus complexe, nouvelles features, moins de régularisation

**Bon ajustement :**
```
Score Train : ████████     0.90
Score Valid : ███████      0.87  (proche du train)
→ Les deux courbes convergent vers une valeur haute
```

---

## Export

Le bouton **Télécharger les prédictions (CSV)** exporte un fichier avec :
- `y_réel` : valeurs réelles du set de test
- `y_prédit` : prédictions du modèle
- `confiance` : score de confiance (classification uniquement)
