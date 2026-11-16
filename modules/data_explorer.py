"""
Exploration du dataset : navigation par classe, grille d'images,
agrandissement de l'image sélectionnée.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config import TRAIN_DIR, CLASS_NAMES, CLASS_LABELS_FR


def _list_images(class_dir: Path) -> list[Path]:
    """Liste les images par différants classe de tumeur

    @Args:
        class_dir: Dossier contenant les images

    @Returns:
        Liste triée
    """
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])


def render():
    """Affichage la page d'exploration.

    @ Vérifie la présence du dossier d'entraînement.
    @ choisir une classe tumeur.
    @ Affichage les images en grille de 4 colonnes.
    @ Agrandir une image sélectionnée.
    """
    st.header("Exploration du dataset")
    st.markdown(
        "Parcourez les images du dataset d'entraînement par classe. "
        "Cliquez sur une image pour l'agrandir."
    )

    if not TRAIN_DIR.exists():
        st.error(f"Répertoire d'entraînement introuvable : {TRAIN_DIR}")
        return

    selected_class = st.selectbox(
        "Sélectionnez une classe",
        options=CLASS_NAMES,
        format_func=lambda c: CLASS_LABELS_FR[c],
    )

    class_dir = TRAIN_DIR / selected_class
    images = _list_images(class_dir)

    if not images:
        st.warning("Aucune image trouvée pour cette classe.")
        return

    st.caption(f"{len(images)} images disponibles")

    # Grille 4 colonnes
    cols = st.columns(4)
    for idx, img_path in enumerate(images):
        col = cols[idx % 4]
        with col:
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
            if st.button(
                f"{img_path.name}",
                key=f"zoom_{img_path.name}",
                use_container_width=True,
            ):
                st.session_state["zoom_image"] = img_path

    # Agrandissement
    if "zoom_image" in st.session_state:
        st.divider()
        st.subheader("Image agrandie")
        img = Image.open(st.session_state["zoom_image"])
        st.image(img, use_container_width=True)
        st.caption(st.session_state["zoom_image"].name)
        if st.button("Fermer l'agrandissement"):
            del st.session_state["zoom_image"]
            st.rerun()
