"""DataLabSn — Page d'accueil du dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="DataLabSn",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2D3748;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1A1F2E;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #6C63FF;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Hero section
    st.markdown('<div class="hero-title">🔬 DataLabSn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Plateforme d\'analyse de données et de machine learning'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithmes ML", "17", "Classification · Régression · Clustering")
    with col2:
        st.metric("Datasets intégrés", "7", "Iris, Wine, Titanic, ...")
    with col3:
        st.metric("Formats supportés", "6", "CSV, Excel, Parquet, JSON, ...")
    with col4:
        st.metric("Métriques d'éval.", "12+", "Accuracy, F1, ROC-AUC, R², ...")

    st.divider()

    # Feature cards
    st.subheader("Fonctionnalités")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Exploration des données</h3>
            <p>Visualisez vos données : statistiques descriptives, distributions, corrélations, valeurs manquantes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Pipeline de traitement</h3>
            <p>Nettoyage automatique, encodage des variables catégorielles, normalisation et sélection de features.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Entraînement de modèles</h3>
            <p>7 classifieurs, 9 régresseurs, 3 algorithmes de clustering avec optimisation d'hyperparamètres.</p>
        </div>
        """, unsafe_allow_html=True)

    col_d, col_e, col_f = st.columns(3)

    with col_d:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Évaluation complète</h3>
            <p>Matrice de confusion, courbes ROC, learning curves, comparaison de modèles côte-à-côte.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_e:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Prédictions en temps réel</h3>
            <p>Saisissez vos données et obtenez des prédictions instantanées avec scores de confiance.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_f:
        st.markdown("""
        <div class="feature-card">
            <h3>💾 Export & Sauvegarde</h3>
            <p>Exportez vos modèles entraînés, résultats CSV et visualisations PNG.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick start
    st.subheader("Démarrage rapide")
    st.markdown("""
    1. **📊 Data Explorer** — Chargez un dataset (CSV, Excel, ou dataset intégré) et explorez-le
    2. **🔧 Pipeline** — Configurez le preprocessing : imputation, encodage, normalisation
    3. **🤖 Entraînement** — Choisissez un algorithme, ajustez les paramètres, entraînez
    4. **📈 Évaluation** — Analysez les performances : métriques, graphiques, comparaisons
    5. **🔮 Prédictions** — Utilisez votre modèle pour prédire de nouvelles données
    """)

    st.divider()

    # Tech stack
    st.subheader("Stack technique")
    st.markdown("""
    | Composant | Technologie |
    |-----------|-------------|
    | Traitement de données | **Pandas 2.x**, **NumPy** |
    | Machine Learning | **Scikit-learn 1.3+** |
    | Dashboard interactif | **Streamlit 1.28+** |
    | Visualisation | **Plotly**, **Matplotlib**, **Seaborn** |
    | Sérialisation | **Joblib** |
    | Configuration | **YAML**, **Pydantic** |
    """)

    st.caption("DataLabSn v1.0.0 · Construit avec Streamlit · Python 3.10+")


if __name__ == "__main__":
    main()
