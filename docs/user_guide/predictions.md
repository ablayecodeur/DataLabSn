# Guide utilisateur — Prédictions

La page **🔮 Prédictions** permet d'utiliser votre modèle entraîné pour prédire de nouvelles données, soit manuellement, soit en lot depuis un fichier.

---

## Prérequis

Un modèle doit avoir été entraîné dans la page **🤖 Model Training** de la session courante. Les informations suivantes sont affichées en haut de page :

- Nom du modèle actif
- Score sur le set de test
- Nombre de features attendues

---

## Mode 1 — Saisie manuelle

Un formulaire adaptatif est généré automatiquement en fonction des features du modèle :

- **Feature numérique** → champ numérique pré-rempli avec la moyenne du dataset
- **Feature catégorielle** → liste déroulante des valeurs uniques observées à l'entraînement

### Résultats

**Classification :**
- Classe prédite
- Niveau de confiance global (probabilité de la classe prédite)
- Tableau des probabilités par classe
- Bar chart interactif des probabilités

**Régression :**
- Valeur prédite
- Percentile de la prédiction dans la distribution du dataset d'entraînement (ex. "cette prédiction est dans le top 15% des valeurs")

---

## Mode 2 — Prédictions en lot

Uploadez un fichier CSV, Excel ou JSON contenant plusieurs observations à prédire simultanément.

### Format attendu

Le fichier doit contenir les mêmes colonnes que les données d'entraînement (avant preprocessing). La colonne cible ne doit pas être présente.

**Exemple de fichier `a_predire.csv` :**
```csv
age,purchases,avg_order_value,days_since_last_purchase
35,12,89.50,45
28,3,34.20,180
52,28,156.00,7
```

### Résultats

- Tableau complet avec les prédictions ajoutées dans une colonne `🎯 prediction`
- Colonne `📊 confiance` (classification) : probabilité de la classe prédite
- Distribution des prédictions (histogramme)
- Bouton d'export CSV

---

## Sauvegarde du modèle

### Télécharger le modèle

Cliquez **⬇ Télécharger le modèle (.joblib)** pour télécharger le modèle sérialisé. Le fichier contient le modèle entraîné et la liste des features.

### Sauvegarder sur le disque

```
Nom du fichier : random_forest.joblib
→ Bouton 💾 Sauvegarder le modèle
→ Sauvegardé dans models/saved/random_forest.joblib
```

### Recharger un modèle sauvegardé

```python
from datalabsn.models.predictor import Predictor

predictor = Predictor.load("models/saved/random_forest.joblib")
predictions = predictor.predict(new_data)
```

---

## Bonnes pratiques

- Assurez-vous que les données à prédire ont été **prétraitées de la même façon** que les données d'entraînement. Si vous avez utilisé le pipeline DataLabSn, rechargez le `preprocessor` sauvegardé en session.
- Les valeurs catégorielles inconnues (non vues à l'entraînement) sont gérées par `handle_unknown='ignore'` dans OneHotEncoder — elles donnent une colonne de zéros.
- La **confiance** n'est pas un indicateur absolu de correction — un modèle peut être confiant et faux. Utilisez-la comme indicateur relatif.
