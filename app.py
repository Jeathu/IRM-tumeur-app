"""
Point d'entrée principal de l'application Streamlit
    Orchestre la navigation  :
        - Exploration des données
        - Chargement du modèle
        - Test sur une image
        - Génération du rapport PDF
"""

# streamlit run app.py


import io
import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path

from config import check_paths, CLASS_LABELS_FR, CLASS_NAMES, TEST_DIR
from modules import data_explorer, model_loader, metrics, pdf_report

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Classification IRM cérébrales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


paths_ok = check_paths()
if not all(paths_ok.values()):
    st.error("Ressources manquantes :")
    for k, v in paths_ok.items():
        if not v:
            st.text(f" - {k}")
    st.stop()


# ---------------------------------------------------------------------------
# Barre de navigation
# ---------------------------------------------------------------------------
col1, col2 = st.sidebar.columns([1, 5], vertical_alignment="center")
col1.image("images/2491298.png", width=50)
col2.subheader("IRM Brain Tumor")
st.sidebar.markdown(
    "Application de classification d'IRM cérébrales en 4 classes labels."
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Exploration des données",
        "Chargement du modèle",
        "Test sur une image",
        "Rapport PDF",
    ],
)


if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ---------------------------------------------------------------------------
# Exploration des données
# ---------------------------------------------------------------------------
if page == "Exploration des données":
    data_explorer.render()


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
elif page == "Chargement du modèle":
    st.header("Modèle chargé")

    try:
        model = model_loader.load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Architecture")
    st.code(model_loader.get_model_summary())

    # Affichage des hyperparamètres
    st.subheader("Hyperparamètres")
    params = model_loader.load_params()  # Callback
    if params:
        st.json(params)
    else:
        st.info("params.json introuvable")

    # Affichage matrice de confusion et rapport de classification
    st.subheader("Métriques sur le jeu de test")
    metrics.render()


# ---------------------------------------------------------------------------
# Test sur une image
# ---------------------------------------------------------------------------
elif page == "Test sur une image":
    st.header("Test sur une image")
    st.markdown(
        "Sélectionnez une image depuis votre ordinateur ou depuis le dataset de test."
    )

    source = st.radio("Source de l'image", ["Upload", "Depuis le dataset test"])
    image_bytes = None

    if source == "Upload":
        uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bytes = uploaded.read()
    else:

        selected_class = st.selectbox(
            "Classe", CLASS_NAMES, format_func=lambda c: CLASS_LABELS_FR[c]
        )

        # On test prédiction sur des image test
        class_dir = TEST_DIR / selected_class

        if not class_dir.exists():
            st.warning(f"Dossier introuvable : {class_dir}")
        else:
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        if images:
            selected = st.selectbox("Image", [p.name for p in images])
            image_bytes = (class_dir / selected).read_bytes()
        else:
            st.info("Aucune image dans ce dossier.")

    if image_bytes is not None:
        st.session_state.last_image = image_bytes
        st.image(
            Image.open(io.BytesIO(image_bytes)), caption="Image à analyser", width=300
        )

        with st.spinner("Analyse en cours…"):
            prediction = model_loader.predict(image_bytes)
            st.session_state.last_prediction = prediction

        st.subheader("Résultats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Classe prédite",
                prediction["class_fr"],
            )
        with col2:
            st.metric("Confiance", f"{prediction['confidence']:.2%}")

        # Affichage des probabilités avec DataFrame pandas
        df_probs = pd.DataFrame(
            {
                "Classe": [CLASS_LABELS_FR[c] for c in CLASS_NAMES],
                "Probabilité": prediction["probs"],
            }
        )
        st.bar_chart(df_probs.set_index("Classe"), use_container_width=True)


# ---------------------------------------------------------------------------
# Rapport PDF
# ---------------------------------------------------------------------------
elif page == "Rapport PDF":
    if st.session_state.last_image is None:
        st.warning(
            "Veuillez d'abord tester une image dans l'onglet "
            "**'Test sur une image'**."
        )
    else:
        metrics_data = metrics.compute_metrics()
        pdf_report.render(
            st.session_state.last_image,
            st.session_state.last_prediction,
            metrics_data,
        )
