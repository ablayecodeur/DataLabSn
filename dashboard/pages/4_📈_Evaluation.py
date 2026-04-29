"""Page d'évaluation des modèles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from datalab.evaluation import Evaluator

st.set_page_config(page_title="Évaluation · DataLab", page_icon="📈", layout="wide")
st.title("📈 Évaluation des modèles")
st.caption("Analysez les performances de vos modèles en profondeur.")


def plot_confusion_matrix(cm, labels=None):
    labels = labels or [str(i) for i in range(cm.shape[0])]
    fig = ff.create_annotated_heatmap(
        cm,
        x=labels, y=labels,
        colorscale="Purples",
        showscale=True,
    )
    fig.update_layout(
        title="Matrice de confusion",
        template="plotly_dark",
        xaxis_title="Prédiction",
        yaxis_title="Réalité",
        height=450,
    )
    fig["data"][0]["showscale"] = True
    return fig


def plot_roc_curve(y_test, y_proba, n_classes):
    fig = go.Figure()

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                name=f"ROC (AUC = {roc_auc:.4f})",
                                line=dict(color="#6C63FF", width=2)))
    else:
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                    name=f"Classe {cls} (AUC={roc_auc:.3f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                            line=dict(dash="dash", color="#A0AEC0"),
                            name="Aléatoire", showlegend=True))
    fig.update_layout(
        title="Courbe ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_precision_recall(y_test, y_proba):
    fig = go.Figure()
    if y_proba.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                name="Precision-Recall",
                                line=dict(color="#3ECFCF", width=2)))
    fig.update_layout(
        title="Courbe Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                            marker=dict(color="#6C63FF", opacity=0.6),
                            name="Résidus"))
    fig.add_hline(y=0, line_dash="dash", line_color="#F6AD55")
    fig.update_layout(
        title="Résidus vs Prédictions",
        xaxis_title="Valeurs prédites",
        yaxis_title="Résidus",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                            marker=dict(color="#6C63FF", opacity=0.6),
                            name="Prédictions"))
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                            line=dict(dash="dash", color="#F6AD55"),
                            name="Parfait"))
    fig.update_layout(
        title="Réel vs Prédit",
        xaxis_title="Valeurs réelles",
        yaxis_title="Valeurs prédites",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_learning_curve(result, X, y, task):
    with st.spinner("Calcul des courbes d'apprentissage..."):
        evaluator = Evaluator()
        lc = evaluator.learning_curve_data(result.model, X.values, y.values, task)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc["train_size"], y=lc["train_mean"],
        mode="lines+markers", name="Score entraînement",
        line=dict(color="#6C63FF"),
        error_y=dict(type="data", array=lc["train_std"], visible=True),
    ))
    fig.add_trace(go.Scatter(
        x=lc["train_size"], y=lc["val_mean"],
        mode="lines+markers", name="Score validation",
        line=dict(color="#3ECFCF"),
        error_y=dict(type="data", array=lc["val_std"], visible=True),
    ))
    fig.update_layout(
        title="Courbes d'apprentissage",
        xaxis_title="Taille du set d'entraînement",
        yaxis_title="Score",
        template="plotly_dark",
        height=400,
    )
    return fig


def main():
    result = st.session_state.get("training_result")
    X = st.session_state.get("X_processed", st.session_state.get("X"))
    y = st.session_state.get("y")
    task = st.session_state.get("task", "classification")

    if result is None:
        st.warning("Entraînez d'abord un modèle dans **🤖 Model Training**.")
        return

    st.success(
        f"Modèle actif : **{result.model_name}** | "
        f"Tâche : {'Classification' if task == 'classification' else 'Régression'} | "
        f"Score test : **{result.test_score:.4f}**"
    )

    evaluator = Evaluator()
    y_test = result.y_test
    y_pred = result.y_pred
    y_proba = result.y_proba

    if task == "classification":
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)

        # Métriques clés
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
        cols[1].metric("F1-Score", f"{metrics['f1_score']:.4f}")
        cols[2].metric("Precision", f"{metrics['precision']:.4f}")
        cols[3].metric("Recall", f"{metrics['recall']:.4f}")
        if "roc_auc" in metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        tabs = st.tabs(["Matrice de confusion", "ROC Curve", "Precision-Recall",
                        "Rapport complet", "Courbes d'apprentissage"])

        with tabs[0]:
            n_classes = len(np.unique(y_test))
            labels = [str(i) for i in range(n_classes)]
            st.plotly_chart(plot_confusion_matrix(metrics["confusion_matrix"], labels),
                          use_container_width=True)

        with tabs[1]:
            if y_proba is not None:
                st.plotly_chart(plot_roc_curve(y_test, y_proba, n_classes),
                               use_container_width=True)
            else:
                st.info("Ce modèle ne fournit pas de probabilités.")

        with tabs[2]:
            if y_proba is not None and n_classes == 2:
                st.plotly_chart(plot_precision_recall(y_test, y_proba),
                               use_container_width=True)
            else:
                st.info("Disponible uniquement pour la classification binaire avec probabilités.")

        with tabs[3]:
            st.text(metrics["classification_report"])

        with tabs[4]:
            if X is not None and y is not None:
                fig_lc = plot_learning_curve(result, X, y, task)
                st.plotly_chart(fig_lc, use_container_width=True)

    else:
        metrics = evaluator.evaluate_regression(y_test, y_pred)

        cols = st.columns(4)
        cols[0].metric("R²", f"{metrics['r2_score']:.4f}")
        cols[1].metric("RMSE", f"{metrics['rmse']:.4f}")
        cols[2].metric("MAE", f"{metrics['mae']:.4f}")
        cols[3].metric("MAPE (%)", f"{metrics['mape']:.2f}")

        tabs = st.tabs(["Réel vs Prédit", "Résidus", "Distribution résidus", "Courbes d'apprentissage"])

        with tabs[0]:
            st.plotly_chart(plot_actual_vs_predicted(y_test, y_pred), use_container_width=True)

        with tabs[1]:
            st.plotly_chart(plot_residuals(y_test, y_pred), use_container_width=True)

        with tabs[2]:
            residuals = y_test - y_pred
            fig = px.histogram(x=residuals, nbins=40, title="Distribution des résidus",
                              template="plotly_dark", marginal="violin")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            if X is not None and y is not None:
                fig_lc = plot_learning_curve(result, X, y, task)
                st.plotly_chart(fig_lc, use_container_width=True)

    # Export des résultats
    st.divider()
    st.subheader("Export")
    results_df = pd.DataFrame({
        "y_réel": y_test,
        "y_prédit": y_pred,
    })
    if y_proba is not None and task == "classification":
        results_df["confiance"] = y_proba.max(axis=1)

    csv = results_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Télécharger les prédictions (CSV)",
        csv, "predictions.csv", "text/csv",
    )


main()
