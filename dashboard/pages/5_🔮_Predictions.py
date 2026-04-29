"""Page de prédictions sur nouvelles données."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datalabsn.models.predictor import Predictor

st.set_page_config(page_title="Prédictions · DataLabSn", page_icon="🔮", layout="wide")
st.title("🔮 Prédictions")
st.caption("Utilisez votre modèle entraîné pour prédire de nouvelles données.")


def manual_input_form(feature_names: list[str], X_ref: pd.DataFrame) -> pd.DataFrame:
    """Formulaire de saisie manuelle."""
    st.subheader("Saisie manuelle")
    st.markdown(f"Saisissez les valeurs pour les {len(feature_names)} features :")

    num_cols_per_row = 3
    values = {}
    rows = (len(feature_names) + num_cols_per_row - 1) // num_cols_per_row

    for r in range(rows):
        cols = st.columns(num_cols_per_row)
        for i, feat in enumerate(feature_names[r * num_cols_per_row:(r + 1) * num_cols_per_row]):
            with cols[i]:
                if X_ref is not None and feat in X_ref.columns:
                    col_data = X_ref[feat]
                    if pd.api.types.is_numeric_dtype(col_data):
                        default = float(col_data.mean())
                        values[feat] = st.number_input(feat, value=default, key=f"inp_{feat}")
                    else:
                        options = col_data.dropna().unique().tolist()
                        values[feat] = st.selectbox(feat, options, key=f"inp_{feat}")
                else:
                    values[feat] = st.number_input(feat, value=0.0, key=f"inp_{feat}")

    return pd.DataFrame([values])


def batch_prediction(predictor: Predictor, preprocessor=None):
    """Prédictions en lot depuis un fichier."""
    st.subheader("Prédictions en lot")
    uploaded = st.file_uploader(
        "Chargez un fichier à prédire",
        type=["csv", "xlsx", "json"],
        help="Le fichier doit avoir les mêmes colonnes que les données d'entraînement",
    )

    if uploaded is None:
        return

    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_json(uploaded)

    st.info(f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("🔮 Prédire", type="primary", key="batch_predict"):
        with st.spinner("Prédiction en cours..."):
            try:
                X_input = df.copy()
                if preprocessor:
                    X_input = preprocessor.transform(X_input)

                result_df = predictor.predict_with_confidence(X_input)
                preds = predictor.predict(X_input)
                result_df = df.copy()
                result_df["🎯 prediction"] = preds
                probas = predictor.predict_proba(X_input)
                if probas is not None:
                    result_df["📊 confiance"] = probas.max(axis=1).round(4)

                st.success(f"Prédictions terminées : {len(preds)} lignes")
                st.dataframe(result_df, use_container_width=True)

                # Distribution des prédictions
                fig = px.histogram(
                    x=preds.astype(str),
                    title="Distribution des prédictions",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export
                csv = result_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Télécharger les résultats (CSV)",
                    csv, "predictions_lot.csv", "text/csv",
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.exception(e)


def main():
    predictor: Predictor = st.session_state.get("predictor")
    result = st.session_state.get("training_result")
    X_ref = st.session_state.get("X_processed", st.session_state.get("X"))
    preprocessor = st.session_state.get("preprocessor")
    task = st.session_state.get("task", "classification")

    if predictor is None or result is None:
        st.warning("Entraînez d'abord un modèle dans **🤖 Model Training**.")
        return

    st.success(
        f"Modèle actif : **{result.model_name}** | "
        f"Score test : **{result.test_score:.4f}** | "
        f"Features : {len(result.feature_names)}"
    )

    mode = st.radio("Mode", ["Saisie manuelle", "Prédictions en lot"], horizontal=True)

    if mode == "Saisie manuelle":
        input_df = manual_input_form(result.feature_names, X_ref)

        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            with st.spinner("Prédiction..."):
                try:
                    preds = predictor.predict(input_df.values)
                    probas = predictor.predict_proba(input_df.values)

                    st.divider()
                    st.subheader("Résultat")

                    if task == "classification":
                        pred_label = preds[0]
                        st.metric("Classe prédite", str(pred_label))

                        if probas is not None:
                            n_classes = probas.shape[1]
                            conf_df = pd.DataFrame({
                                "Classe": [str(i) for i in range(n_classes)],
                                "Probabilité": probas[0].round(4),
                            }).sort_values("Probabilité", ascending=False)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confiance", f"{probas[0].max():.1%}")
                                st.dataframe(conf_df, use_container_width=True)
                            with col2:
                                fig = px.bar(
                                    conf_df, x="Classe", y="Probabilité",
                                    color="Probabilité",
                                    color_continuous_scale="Purples",
                                    title="Probabilités par classe",
                                    template="plotly_dark",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        pred_val = preds[0]
                        y_ref = st.session_state.get("y")
                        st.metric("Valeur prédite", f"{pred_val:.4f}")
                        if y_ref is not None:
                            percentile = (y_ref < pred_val).mean() * 100
                            st.metric("Percentile", f"{percentile:.1f}%",
                                     help="Position de la prédiction dans la distribution des valeurs réelles")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.exception(e)

    else:
        batch_prediction(predictor, preprocessor)

    # Sauvegarde du modèle
    st.divider()
    st.subheader("Sauvegarde du modèle")
    col1, col2 = st.columns(2)
    with col1:
        model_filename = st.text_input("Nom du fichier", value=f"{result.model_name.replace(' ', '_').lower()}.joblib")
    with col2:
        if st.button("💾 Sauvegarder le modèle"):
            save_path = Path("models/saved") / model_filename
            predictor.save(save_path)
            st.success(f"Modèle sauvegardé : `{save_path}`")

    # Sauvegarder en mémoire pour téléchargement
    import joblib, io as _io
    buf = _io.BytesIO()
    joblib.dump({"model": predictor.model, "feature_names": predictor.feature_names}, buf)
    buf.seek(0)
    st.download_button(
        "⬇ Télécharger le modèle (.joblib)",
        buf.getvalue(),
        model_filename,
        "application/octet-stream",
    )


main()
