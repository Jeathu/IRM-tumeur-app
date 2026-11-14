"""
Chargement du modèle et pipeline d'inférence
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import io

from config import IMG_SIZE, CLASS_NAMES, CLASS_LABELS_FR, MODEL_PATH




@st.cache_resource
def load_model():
    """Charge le modèle Keras en cache pour la session.

    Vérifie que le fichier du modèle existe.
    Affiche une erreur si le fichier manque.

    @Returns:
        Modèle Keras chargé depuis `MODEL_PATH`.
    """
    if not MODEL_PATH.exists():
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model





def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prépare une image pour la prédiction du modèle.

    Convertit l'image en niveaux de gris.
    Redimensionne l'image à la taille attendue.
    Normalise les valeurs entre 0 et 1.
    Ajoute les dimensions batch et canal.

    @Args:
        image: Image PIL à prétraiter

    @Returns:
        Tableau NumPy prêt pour le modèle, de forme (1, IMG_SIZE, IMG_SIZE, 1)
    """
    img = image.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0 
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr






@st.cache_data(show_spinner=False)
def predict(image_bytes: bytes) -> dict:
    """Prédit la classe d'une image et met le résultat en cache.

    Charge le modèle en cache.
    Ouvre l'image depuis les octets reçus.
    Prétraite l'image et lance la prédiction.

    @Args:
        image_bytes: Contenu binaire de l'image à analyser

    @Returns:
        Dictionnaire contenant `probs`, `class`, `class_fr` et `confidence`
    """
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes))
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]

    return {
        "probs": probs,
        "class": CLASS_NAMES[int(np.argmax(probs))],
        "class_fr": CLASS_LABELS_FR[CLASS_NAMES[int(np.argmax(probs))]],
        "confidence": float(np.max(probs)),
    }





def get_model_summary() -> str:
    """Récupère le résumé textuel du modèle Keras.

    Charge le modèle en cache.
    Capture chaque ligne du résumé.
    Assemble les lignes en un seul texte.

    @Returns:
        Résumé complet du modèle formaté sur plusieurs lignes
    """
    model = load_model()
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)