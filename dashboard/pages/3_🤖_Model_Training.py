"""Page d'entraînement des modèles ML."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datalabsn.models import CLASSIFIERS, REGRESSORS, ModelRegistry, ModelTrainer
from datalabsn.models.predictor import Predictor

st.set_page_config(page_title="Entraînement · DataLabSn", page_icon="🤖", layout="wide")
st.title("🤖 Entraînement des modèles")
st.caption("Entraînez, optimisez et comparez vos modèles de machine learning.")


def get_data():
    X = st.session_state.get("X_processed", st.session_state.get("X"))
    y = st.session_state.get("y")
    task = st.session_state.get("task", "classification")
    return X, y, task


def single_model_training(X, y, task):
    st.subheader("Entraînement d'un modèle")

    catalog = CLASSIFIERS if task == "classification" else REGRESSORS
    model_name = st.selectbox(
        "Algorithme",
        list(catalog.keys()),
        format_func=lambda n: f"{n} — {catalog[n]['description']}",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Taille du set de test (%)", 10, 40, 20) / 100
    with col2:
        cv_folds = st.slider("Folds de validation croisée", 2, 10, 5)
    with col3:
        random_state = st.number_input("Random state", value=42, min_value=0)

    tune = st.checkbox("Optimiser les hyperparamètres")
    if tune:
        strategy = st.radio("Stratégie", ["grid", "random"], horizontal=True,
                           format_func=lambda s: "Grid Search" if s == "grid" else "Random Search")
        n_iter = st.slider("Nombre d'itérations (Random Search)", 5, 50, 20) if strategy == "random" else 20
    else:
        strategy, n_iter = "grid", 20

    if st.button("🚀 Entraîner le modèle", type="primary", use_container_width=True):
        with st.spinner(f"Entraînement de **{model_name}**..."):
            try:
                trainer = ModelTrainer(
                    random_state=random_state,
                    test_size=test_size,
                    cv_folds=cv_folds,
                )
                result = trainer.train(
                    X, y, model_name, task,
                    tune_hyperparams=tune,
                    search_strategy=strategy,
                    n_iter=n_iter,
                )
                st.session_state["training_result"] = result
                st.session_state["predictor"] = Predictor(
                    model=result.model,
                    feature_names=result.feature_names,
                )
                st.success(f"Modèle **{model_name}** entraîné en {result.training_time:.2f}s")
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.exception(e)

    result = st.session_state.get("training_result")
    if result:
        display_training_result(result, task)


def display_training_result(result, task):
    st.subheader(f"Résultats — {result.model_name}")

    metric_label = "Accuracy" if task == "classification" else "R²"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{metric_label} Train", f"{result.train_score:.4f}")
    c2.metric(f"{metric_label} Test", f"{result.test_score:.4f}")
    c3.metric("CV Moyen", f"{result.cv_mean:.4f} ± {result.cv_std:.4f}")
    c4.metric("Surapprentissage", f"{result.train_score - result.test_score:+.4f}")

    # CV scores plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(result.cv_scores))],
        y=result.cv_scores,
        marker_color="#6C63FF",
        name="CV Score",
    ))
    fig.add_hline(y=result.cv_mean, line_dash="dash", line_color="#3ECFCF",
                  annotation_text=f"Moyenne: {result.cv_mean:.4f}")
    fig.update_layout(
        title="Scores de validation croisée",
        template="plotly_dark",
        height=300,
        yaxis_title=metric_label,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    predictor = Predictor(model=result.model, feature_names=result.feature_names)
    fi = predictor.feature_importance()
    if fi is not None:
        fig_fi = px.bar(
            fi.head(20), x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Viridis",
            title="Importance des features (Top 20)",
            template="plotly_dark",
        )
        fig_fi.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)

    if result.best_params:
        st.markdown("**Meilleurs hyperparamètres**")
        st.json(result.best_params)


def compare_models(X, y, task):
    st.subheader("Comparaison de modèles")

    catalog = CLASSIFIERS if task == "classification" else REGRESSORS
    selected_models = st.multiselect(
        "Modèles à comparer",
        list(catalog.keys()),
        default=list(catalog.keys())[:4],
    )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Taille du set de test (%)", 10, 40, 20, key="cmp_test") / 100
    with col2:
        cv_folds = st.slider("Folds CV", 2, 10, 5, key="cmp_cv")

    if st.button("⚡ Comparer les modèles", type="primary", use_container_width=True):
        if not selected_models:
            st.warning("Sélectionnez au moins un modèle.")
            return

        with st.spinner("Comparaison en cours..."):
            trainer = ModelTrainer(test_size=test_size, cv_folds=cv_folds)
            df_cmp = trainer.compare_models(X, y, task, selected_models)
            st.session_state["comparison_df"] = df_cmp

    df_cmp = st.session_state.get("comparison_df")
    if df_cmp is not None:
        st.dataframe(
            df_cmp.style.background_gradient(subset=["CV Moyen", "Score Test"], cmap="Purples"),
            use_container_width=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Score Train", x=df_cmp["Modèle"], y=df_cmp["Score Train"],
                            marker_color="#6C63FF"))
        fig.add_trace(go.Bar(name="Score Test", x=df_cmp["Modèle"], y=df_cmp["Score Test"],
                            marker_color="#3ECFCF"))
        fig.add_trace(go.Bar(name="CV Moyen", x=df_cmp["Modèle"], y=df_cmp["CV Moyen"],
                            marker_color="#F6AD55"))
        fig.update_layout(
            barmode="group",
            title="Comparaison des modèles",
            template="plotly_dark",
            height=400,
            yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommandation
        best = df_cmp.iloc[0]
        st.info(
            f"**Meilleur modèle :** {best['Modèle']} "
            f"(CV moyen : {best['CV Moyen']:.4f}, "
            f"score test : {best['Score Test']:.4f})"
        )


def main():
    X, y, task = get_data()

    if X is None or y is None:
        st.warning("Chargez un dataset avec une variable cible dans **📊 Data Explorer**.")
        return

    st.info(
        f"**Tâche :** {'Classification' if task == 'classification' else 'Régression'} | "
        f"**Données :** {X.shape[0]} lignes × {X.shape[1]} colonnes | "
        f"**Cible :** `{y.name}`"
    )

    if "X_processed" not in st.session_state:
        st.warning("Le pipeline n'a pas été appliqué. Vous utilisez les données brutes — certains modèles peuvent échouer.")

    mode = st.radio("Mode", ["Modèle unique", "Comparaison"], horizontal=True)

    if mode == "Modèle unique":
        single_model_training(X, y, task)
    else:
        compare_models(X, y, task)


main()
