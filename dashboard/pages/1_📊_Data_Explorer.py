"""Page d'exploration des données."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from datalab.pipeline import DataIngestion, DataValidator
from datalab.pipeline.ingestion import SAMPLE_DATASETS

st.set_page_config(page_title="Data Explorer · DataLab", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption("Chargez et explorez vos données en profondeur.")


def load_data() -> tuple[pd.DataFrame, pd.Series | None, str]:
    source = st.radio(
        "Source de données",
        ["Dataset intégré", "Fichier local"],
        horizontal=True,
    )

    ingestion = DataIngestion()

    if source == "Dataset intégré":
        dataset_name = st.selectbox(
            "Choisissez un dataset",
            list(SAMPLE_DATASETS.keys()),
            format_func=lambda k: f"{k} — {SAMPLE_DATASETS[k]}",
        )
        if st.button("Charger le dataset", type="primary"):
            with st.spinner("Chargement..."):
                X, y, task = ingestion.load_sample(dataset_name)
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["task"] = task
                st.session_state["dataset_name"] = dataset_name
                st.success(f"Dataset **{dataset_name}** chargé — {X.shape[0]} lignes × {X.shape[1]} colonnes")

    else:
        uploaded = st.file_uploader(
            "Glissez votre fichier",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            help="CSV, Excel, Parquet ou JSON",
        )
        if uploaded:
            target_col = st.text_input("Colonne cible (optionnel)")
            sep = st.text_input("Séparateur CSV", value=",") if uploaded.name.endswith(".csv") else ","
            if st.button("Charger", type="primary"):
                with st.spinner("Lecture du fichier..."):
                    suffix = Path(uploaded.name).suffix.lower()
                    if suffix == ".csv":
                        df = pd.read_csv(uploaded, sep=sep)
                    elif suffix in (".xlsx", ".xls"):
                        df = pd.read_excel(uploaded)
                    elif suffix == ".parquet":
                        df = pd.read_parquet(uploaded)
                    else:
                        df = pd.read_json(uploaded)

                    y = None
                    if target_col and target_col in df.columns:
                        y = df[target_col]
                        df = df.drop(columns=[target_col])

                    st.session_state["X"] = df
                    st.session_state["y"] = y
                    st.session_state["task"] = "classification" if y is not None and y.nunique() < 20 else "regression"
                    st.session_state["dataset_name"] = uploaded.name
                    st.success(f"Fichier chargé — {df.shape[0]} lignes × {df.shape[1]} colonnes")

    return (
        st.session_state.get("X"),
        st.session_state.get("y"),
        st.session_state.get("task", "classification"),
    )


def show_overview(X: pd.DataFrame, y: pd.Series | None):
    st.subheader("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{X.shape[0]:,}")
    c2.metric("Colonnes", f"{X.shape[1]:,}")
    c3.metric("Valeurs manquantes", f"{X.isnull().sum().sum():,}")
    c4.metric("Mémoire", f"{X.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    tab1, tab2, tab3 = st.tabs(["Aperçu", "Types", "Statistiques"])

    with tab1:
        n_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
        df_display = X.copy()
        if y is not None:
            df_display["🎯 target"] = y.values
        st.dataframe(df_display.head(n_rows), use_container_width=True)

    with tab2:
        dtypes_df = pd.DataFrame({
            "Colonne": X.dtypes.index,
            "Type": X.dtypes.astype(str).values,
            "Valeurs uniques": [X[c].nunique() for c in X.columns],
            "Manquantes": X.isnull().sum().values,
            "Manquantes %": (X.isnull().mean() * 100).round(2).values,
        })
        st.dataframe(dtypes_df, use_container_width=True)

    with tab3:
        num_cols = X.select_dtypes(include=np.number)
        if not num_cols.empty:
            st.dataframe(num_cols.describe().round(4).T, use_container_width=True)


def show_distributions(X: pd.DataFrame, y: pd.Series | None):
    st.subheader("Distributions")
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        st.markdown("**Variables numériques**")
        col_select = st.multiselect("Colonnes", num_cols, default=num_cols[:min(4, len(num_cols))])
        if col_select:
            n_cols = min(3, len(col_select))
            rows = (len(col_select) + n_cols - 1) // n_cols
            for r in range(rows):
                cols = st.columns(n_cols)
                for i, col_name in enumerate(col_select[r * n_cols:(r + 1) * n_cols]):
                    with cols[i]:
                        color_by = y.astype(str) if y is not None else None
                        fig = px.histogram(
                            x=X[col_name],
                            color=color_by,
                            nbins=30,
                            title=col_name,
                            template="plotly_dark",
                            marginal="box",
                        )
                        fig.update_layout(height=300, showlegend=bool(color_by is not None))
                        st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.markdown("**Variables catégorielles**")
        for col_name in cat_cols[:6]:
            vc = X[col_name].value_counts().reset_index()
            vc.columns = [col_name, "count"]
            fig = px.bar(vc, x=col_name, y="count", title=col_name, template="plotly_dark")
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)


def show_correlations(X: pd.DataFrame):
    st.subheader("Corrélations")
    num_df = X.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        st.info("Pas assez de colonnes numériques pour calculer les corrélations.")
        return

    method = st.radio("Méthode", ["pearson", "spearman", "kendall"], horizontal=True)
    corr = num_df.corr(method=method)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title=f"Matrice de corrélation ({method})",
        template="plotly_dark",
        text_auto=".2f",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Paires les plus corrélées**")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["Feature 1", "Feature 2", "Corrélation"]
    pairs["|Corrélation|"] = pairs["Corrélation"].abs()
    pairs = pairs.sort_values("|Corrélation|", ascending=False).head(10)
    st.dataframe(pairs.round(4), use_container_width=True)


def show_missing(X: pd.DataFrame):
    st.subheader("Valeurs manquantes")
    missing = X.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("Aucune valeur manquante détectée !")
        return

    missing_df = pd.DataFrame({
        "Colonne": missing.index,
        "Manquantes": missing.values,
        "Pourcentage": (missing / len(X) * 100).round(2).values,
    }).sort_values("Manquantes", ascending=False)

    fig = px.bar(
        missing_df, x="Colonne", y="Pourcentage",
        color="Pourcentage",
        color_continuous_scale="Reds",
        title="Taux de valeurs manquantes par colonne",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missing_df, use_container_width=True)


def show_target_analysis(y: pd.Series, task: str):
    st.subheader("Analyse de la variable cible")

    if task == "classification":
        vc = y.value_counts().reset_index()
        vc.columns = ["Classe", "Effectif"]
        vc["Proportion %"] = (vc["Effectif"] / vc["Effectif"].sum() * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(vc, values="Effectif", names="Classe",
                        title="Distribution des classes",
                        template="plotly_dark", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(vc, x="Classe", y="Effectif",
                        color="Proportion %",
                        template="plotly_dark",
                        title="Effectifs par classe")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc, use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(x=y, nbins=40, title="Distribution de la cible",
                              template="plotly_dark", marginal="violin")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(y=y, title="Boxplot de la cible", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


def show_validation(X: pd.DataFrame, y: pd.Series | None):
    st.subheader("Rapport de validation")
    validator = DataValidator()
    report = validator.validate(X, y)

    status = "✅ Données valides" if report.passed else "❌ Problèmes détectés"
    st.markdown(f"**Statut :** {status}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", f"{report.stats['n_rows']:,}")
    c2.metric("Colonnes", f"{report.stats['n_cols']:,}")
    c3.metric("Doublons", f"{report.stats.get('duplicates', 0):,}")

    if report.errors:
        for err in report.errors:
            st.error(f"🚫 {err}")
    if report.warnings:
        for warn in report.warnings:
            st.warning(f"⚠️ {warn}")
    if not report.errors and not report.warnings:
        st.success("Aucun problème détecté. Vos données sont prêtes pour la modélisation.")


def main():
    X, y, task = load_data()

    if X is None:
        st.info("Chargez un dataset pour commencer l'exploration.")
        st.markdown("---")
        st.subheader("Datasets disponibles")
        for name, desc in SAMPLE_DATASETS.items():
            st.markdown(f"- **{name}** : {desc}")
        return

    tabs = st.tabs(["Vue d'ensemble", "Distributions", "Corrélations", "Manquantes", "Cible", "Validation"])

    with tabs[0]:
        show_overview(X, y)
    with tabs[1]:
        show_distributions(X, y)
    with tabs[2]:
        show_correlations(X)
    with tabs[3]:
        show_missing(X)
    with tabs[4]:
        if y is not None:
            show_target_analysis(y, task)
        else:
            st.info("Aucune variable cible définie.")
    with tabs[5]:
        show_validation(X, y)


main()
