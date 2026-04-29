# Guide d'installation

## Prérequis

| Outil | Version minimale | Vérification |
|---|---|---|
| Python | 3.10 | `python --version` |
| pip | 23.0 | `pip --version` |
| Git | 2.x | `git --version` |

---

## Installation standard

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/DataLabSn.git
cd DataLabSn
```

### 2. Créer un environnement virtuel

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python -c "import streamlit, sklearn, pandas, plotly; print('OK')"
```

### 5. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

---

## Installation en mode développement

Pour contribuer au projet ou modifier le code source :

```bash
# Installer le package en mode éditable
pip install -e ".[dev]"

# Installer les dépendances de test
pip install pytest pytest-cov

# Vérifier que les tests passent
pytest tests/ -v
```

---

## Installation avec conda

```bash
conda create -n datalabsn python=3.11
conda activate datalabsn
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Variables d'environnement

Créez un fichier `.env` à la racine (optionnel) :

```bash
DATALAB_RANDOM_STATE=42
DATALAB_TEST_SIZE=0.2
DATALAB_CV_FOLDS=5
```

---

## Résolution de problèmes courants

### `ModuleNotFoundError: No module named 'datalabsn'`

Le package source n'est pas dans le PYTHONPATH. Lancez depuis la racine du projet ou installez en mode éditable :

```bash
pip install -e .
```

### `streamlit: command not found`

Streamlit n'est pas dans le PATH. Utilisez :

```bash
python -m streamlit run dashboard/app.py
```

### Erreur d'encodage sur Windows (emojis dans les noms de fichiers)

Sous Windows, configurez la console en UTF-8 :

```powershell
chcp 65001
```

### Port 8501 déjà utilisé

```bash
streamlit run dashboard/app.py --server.port 8502
```
