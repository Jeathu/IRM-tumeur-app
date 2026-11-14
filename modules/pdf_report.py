"""
Génération d'un rapport PDF contenant :
- L'image testée
- Les probabilités de prédiction
- Les métriques du modèle
- Les paramètres du modèle
"""

import io
from io import BytesIO
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from config import CLASS_NAMES, CLASS_LABELS_FR


def generate_pdf(image_bytes: bytes, prediction: dict, metrics: dict) -> bytes:
    """
    Génère le rapport PDF et retourne son contenu sous forme de bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=6
    )

    story = []

    # --- Titre ---
    story.append(Paragraph("Rapport d'analyse IRM cérébrale", title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            ParagraphStyle("Date", alignment=TA_CENTER, textColor=colors.grey),
        )
    )
    story.append(Spacer(1, 1 * cm))

    # --- Image testée ---
    story.append(Paragraph("Image analysée", heading_style))
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
    img_path = BytesIO()
    img.save(img_path, format="PNG")
    img_path.seek(0)
    story.append(RLImage(img_path, width=6 * cm, height=6 * cm))
    story.append(Spacer(1, 0.5 * cm))

    # --- Prédiction ---
    story.append(Paragraph("Prédiction", heading_style))
    story.append(
        Paragraph(
            f"<b>Classe prédite :</b> {prediction['class_fr']}",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Confiance :</b> {prediction['confidence']:.2%}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Tableau des probabilités
    prob_data = [["Classe", "Probabilité"]]
    for cls, prob in zip(CLASS_NAMES, prediction["probs"]):
        prob_data.append([CLASS_LABELS_FR[cls], f"{prob:.2%}"])
    prob_table = Table(prob_data, colWidths=[8 * cm, 5 * cm])
    prob_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]
        )
    )
    story.append(prob_table)
    story.append(Spacer(1, 0.5 * cm))

    # --- Métriques ---
    story.append(Paragraph("Métriques du modèle", heading_style))
    story.append(
        Paragraph(
            f"<b>Accuracy globale :</b> {metrics['accuracy']:.2%}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Matrice de confusion dans le PDF
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_fr = [CLASS_LABELS_FR[c] for c in CLASS_NAMES]
    ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"], display_labels=labels_fr
    ).plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Matrice de confusion")
    plt.xticks(rotation=45)

    cm_buf = BytesIO()
    fig.savefig(cm_buf, format="PNG", dpi=150, bbox_inches="tight")
    plt.close(fig)
    cm_buf.seek(0)
    story.append(RLImage(cm_buf, width=12 * cm, height=10 * cm))
    story.append(Spacer(1, 0.5 * cm))

    # --- Paramètres du modèle ---
    story.append(Paragraph("Paramètres du modèle", heading_style))
    params = [
        ["Paramètre", "Valeur"],
        ["Architecture", "CNN 4 couches convolutives"],
        ["Taille d'entrée", "128 x 128 x 1 (niveaux de gris)"],
        ["Nombre de classes", "4"],
        ["Activation sortie", "Softmax"],
        ["Optimiseur", "Adam"],
        ["Fonction de perte", "Categorical Crossentropy"],
    ]
    params_table = Table(params, colWidths=[6 * cm, 7 * cm])
    params_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(params_table)

    doc.build(story)
    return buffer.getvalue()


def render(image_bytes: bytes, prediction: dict, metrics: dict):
    """Affiche le bouton de téléchargement du PDF dans Streamlit."""
    st.header("📄 Rapport PDF")
    st.markdown(
        "Générez un rapport PDF complet contenant l'image testée, "
        "les prédictions, les métriques et les paramètres du modèle."
    )

    if image_bytes is None:
        st.warning("Veuillez d'abord tester une image dans l'onglet 'Test'.")
        return

    pdf_bytes = generate_pdf(image_bytes, prediction, metrics)

    st.download_button(
        label="⬇️ Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"rapport_irm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )
