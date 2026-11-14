"""
Calcul et affichage des métriques du modèle :
        - Rapport de classification
        - Matrice de confusion
        - Courbes d'apprentissage (loss / accuracy)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import TEST_DIR, IMG_SIZE, BATCH_SIZE, CLASS_NAMES, CLASS_LABELS_FR
from .model_loader import load_model
from importlib import import_module

ImageDataGenerator = import_module(
    "tensorflow.keras.preprocessing.image"
).ImageDataGenerator


@st.cache_data(show_spinner="Calcul des métriques sur le jeu de test…")
def compute_metrics():
    """Calcule les métriques du modèle sur le jeu de test

    @ Charge le modèle entraîné
    @ Prépare les images de test en niveaux de gris et normalisées
    @ Prédiction les classes
    @ Calcule le rapport de classification, la matrice de confusion et l'exactitude

    @Returns:
        Dictionnaire contenant le rapport, la matrice et l'exactitude.
    """

    model = load_model()

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

    # Callback function (affichage des courbes d'apprentissage)
    _plot_learning_curves()


def _plot_learning_curves():
    """Affiche les courbes d'apprentissage ou un message d'aide

    @ Affichage l'historique de loss et d'accuracy.
    """
    try:
        model = load_model()

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

        batch_losses, batch_accs = [], []
        for _ in range(len(test_gen)):
            x_batch, y_batch = next(test_gen)
            res = model.evaluate(x_batch, y_batch, verbose=0, return_dict=True)
            batch_losses.append(res.get("loss", np.nan))
            batch_accs.append(res.get("accuracy", res.get("acc", np.nan)))

        batches = range(1, len(batch_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(batches, batch_losses, marker="o", label="Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Batch")
        ax1.legend()

        ax2.plot(batches, batch_accs, marker="o", label="Accuracy")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Batch")
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Impossible d'afficher les courbes : {e}")
