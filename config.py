"""
Configuration centralisée de l'application.
Tous les chemins et hyperparamètres sont définis ici..
"""

from pathlib import Path






BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODEL_PATH = BASE_DIR / "models" / "best_brain_tumor_cnn.keras"




# ---------------------------------------------------------------------------
# Paramètres du modèle 
# ---------------------------------------------------------------------------
IMG_SIZE = 128
COLOR_MODE = "grayscale"          # 1 canal, comme à l'entraînement
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS_FR = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Absence de tumeur",
    "pituitary": "Tumeur pituitaire",
}

BATCH_SIZE = 32
SEED = 42





def check_paths() -> dict[str, bool]:
    """Retourne un dictionnaire indiquant la présence des ressources."""
    return {
        "DATA_DIR": DATA_DIR.exists(),
        "TRAIN_DIR": TRAIN_DIR.exists(),
        "TEST_DIR": TEST_DIR.exists(),
        "MODEL_PATH": MODEL_PATH.exists(),
    }