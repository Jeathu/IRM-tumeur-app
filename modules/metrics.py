"""
Calcul et affichage des métriques du modèle :
    - Rapport de classification
    - Matrice de confusion
    - Courbes d'apprentissage (loss / accuracy)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_LABELS_FR,
    HISTORY_PATH,
    PARAMS_PATH,
)
from .model_loader import load_model
from importlib import import_module

ImageDataGenerator = import_module(
    "tensorflow.keras.preprocessing.image"
).ImageDataGenerator


@st.cache_data(show_spinner="Calcul des métriques sur le jeu de test...")
def compute_metrics():
    """Calcule les métriques du modèle sur le jeu de test

    @ Chargement du modèle (best_brain_tumor_cnn.keras)
    @ Prépare les images de test en niveaux de gris et normalisées
    @ Prédiction les classes labeles avec données de test
    @ Calcule le rapport de classification, la matrice de confusion et l'exactitude

    @Returns:
        Dictionnaire contenant le rapport, la matrice et l'exactitude.
    """

    # Callback function (model_loader.py)
    model = load_model()

    # métrique sont calculées sur test data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )

    # predictions
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    return {
        "report": report_dict,
        "cm": cm,
        "accuracy": float(np.mean(y_pred == y_true)),
    }


def render():
    """Affiche la page des métriques du modèle dans Streamlit

    @ Calcule les métriques sur le jeu de test
    @ Affichage le rapport de classification
    @Affichage l'accuracy globale
    @ Matrice de confusion.
    @ Les courbes d'apprentissage.
    """
    st.header("Métriques du modèle")

    metrics = compute_metrics()

    # Rapport de classification
    st.subheader("Rapport de classification")
    report_df = pd.DataFrame(metrics["report"]).T
    report_df.index = [CLASS_LABELS_FR.get(c, c) for c in report_df.index]
    st.dataframe(report_df.round(4), use_container_width=True)
    st.metric("Accuracy globale", f"{metrics['accuracy']:.2%}")

    # Matrice de confusion
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels_fr = [CLASS_LABELS_FR[c] for c in CLASS_NAMES]

    ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"], display_labels=labels_fr
    ).plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Matrice de confusion")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Courbes d'apprentissage
    st.subheader("Courbes d'apprentissage")
    _plot_learning_curves() # Callback


def _plot_learning_curves():
    """Affiche les courbes d'apprentissage (loss et accuracy par epoch)
    à partir de history.json

       @Fichier history.json généré lors de l'entraînement du meilleur modèle.
    """
    if not HISTORY_PATH.exists():
        st.info(f"Historique d'entraînement introuvable : {HISTORY_PATH}")
        return

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            hist = json.load(f)

        hist = hist.get("history", hist)

        epochs = range(1, len(hist["loss"]) + 1)
        acc_key = "accuracy" if "accuracy" in hist else "acc"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, hist["loss"], marker="o", label="Train")
        if "val_loss" in hist:
            ax1.plot(epochs, hist["val_loss"], marker="o", label="Validation")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, hist[acc_key], marker="o", label="Train")
        if f"val_{acc_key}" in hist:
            ax2.plot(epochs, hist[f"val_{acc_key}"], marker="o", label="Validation")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Impossible d'afficher les courbes : {e}")
