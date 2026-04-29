"""Page de configuration du pipeline de traitement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import streamlit as st

from datalab.pipeline import DataPreprocessor, FeatureEngineer

st.set_page_config(page_title="Pipeline · DataLab", page_icon="🔧", layout="wide")
st.title("🔧 Pipeline de traitement")
st.caption("Configurez et appliquez votre pipeline de preprocessing.")


def main():
    X: pd.DataFrame = st.session_state.get("X")
    y: pd.Series = st.session_state.get("y")

    if X is None:
        st.warning("Commencez par charger un dataset dans **📊 Data Explorer**.")
        return

    st.info(f"Dataset actuel : **{st.session_state.get('dataset_name', 'inconnu')}** — {X.shape[0]} lignes × {X.shape[1]} colonnes")

    # ── Preprocessing ────────────────────────────────────────────────────────
    st.subheader("1. Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        num_impute = st.selectbox(
            "Imputation numérique",
            ["median", "mean", "knn", "most_frequent"],
            help="Stratégie pour remplacer les valeurs manquantes numériques",
        )
        scaler = st.selectbox(
            "Normalisation",
            ["standard", "minmax", "robust", "none"],
            help="StandardScaler, MinMaxScaler, RobustScaler ou aucun",
        )
    with col2:
        cat_impute = st.selectbox(
            "Imputation catégorielle",
            ["most_frequent", "median", "mean"],
        )
        encode = st.selectbox(
            "Encodage des variables catégorielles",
            ["onehot", "label", "ordinal"],
            help="OneHot (recommandé), LabelEncoder ou OrdinalEncoder",
        )

    drop_duplicates = st.checkbox("Supprimer les doublons", value=True)

    # ── Feature Engineering ──────────────────────────────────────────────────
    st.subheader("2. Feature Engineering")

    col3, col4 = st.columns(2)
    with col3:
        use_feature_selection = st.checkbox("Sélection de features (SelectKBest)")
        if use_feature_selection:
            k_features = st.slider("Nombre de features à conserver", 1, X.shape[1], min(10, X.shape[1]))
            score_func = st.radio("Fonction de score", ["f_test", "mutual_info"], horizontal=True)

    with col4:
        use_pca = st.checkbox("Réduction dimensionnelle (PCA)")
        if use_pca:
            pca_variance = st.slider("Variance expliquée cible (%)", 50, 99, 95) / 100

    drop_correlated = st.checkbox("Supprimer features très corrélées (r > 0.95)")

    # ── Appliquer ────────────────────────────────────────────────────────────
    st.divider()
    if st.button("▶ Appliquer le pipeline", type="primary", use_container_width=True):
        with st.spinner("Application du pipeline..."):
            try:
                preprocessor = DataPreprocessor()
                X_processed = preprocessor.fit_transform(
                    X,
                    num_impute=num_impute,
                    cat_impute=cat_impute,
                    scaler=scaler,
                    encode=encode,
                    drop_duplicates=drop_duplicates,
                )

                engineer = FeatureEngineer()
                feature_scores = None

                if drop_correlated:
                    to_drop = engineer.correlation_analysis(X_processed)
                    if to_drop:
                        X_processed = X_processed.drop(columns=to_drop)
                        st.info(f"Features corrélées supprimées : {to_drop}")

                if use_feature_selection and y is not None:
                    X_processed, feature_scores = engineer.select_k_best(
                        X_processed, y,
                        k=k_features,
                        task=st.session_state.get("task", "classification"),
                        score_func=score_func,
                    )

                if use_pca:
                    X_processed, pca_model = engineer.apply_pca(X_processed, n_components=pca_variance)
                    st.session_state["pca_model"] = pca_model

                st.session_state["X_processed"] = X_processed
                st.session_state["preprocessor"] = preprocessor
                st.session_state["pipeline_config"] = {
                    "num_impute": num_impute, "cat_impute": cat_impute,
                    "scaler": scaler, "encode": encode,
                }

                st.success(f"Pipeline appliqué ! {X.shape[1]} → **{X_processed.shape[1]} colonnes**")

                # Résultats
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Colonnes avant", X.shape[1])
                col_b.metric("Colonnes après", X_processed.shape[1])
                col_c.metric("Lignes après", X_processed.shape[0])

                st.markdown("**Aperçu des données traitées**")
                st.dataframe(X_processed.head(10), use_container_width=True)

                if feature_scores is not None:
                    st.markdown("**Scores d'importance des features**")
                    import plotly.express as px
                    top = feature_scores.head(20)
                    fig = px.bar(
                        top, x="score", y="feature", orientation="h",
                        color="selected",
                        color_discrete_map={True: "#6C63FF", False: "#4A5568"},
                        title="Scores SelectKBest",
                        template="plotly_dark",
                    )
                    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur dans le pipeline : {e}")
                st.exception(e)

    # Statut actuel
    if "X_processed" in st.session_state:
        st.divider()
        st.success(
            f"Pipeline déjà appliqué — données prêtes : "
            f"{st.session_state['X_processed'].shape[0]} lignes × "
            f"{st.session_state['X_processed'].shape[1]} colonnes"
        )
        if st.button("Réinitialiser le pipeline"):
            del st.session_state["X_processed"]
            st.rerun()


main()
