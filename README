## Installation et lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/Jeathu/IRM-tumeur-app.git
cd IRM-tumeur-app
```

### 2. Préparer le dataset
Télécharge le dataset **Brain Tumor MRI Dataset :**   

- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset   

sur Kaggle, puis organise les dossiers à la racine du projet comme suit :

```text
Data/
├── Training/
└── Testing/
```

### 3. Créer et activer l'environnement virtuel
```bash
python -m venv venv
```

Active ensuite l'environnement virtuel :
* **Windows (Command Prompt / PowerShell) :**
  ```bash
  venv\Scripts\activate
  ```
* **macOS / Linux :**
  ```bash
  source venv/bin/activate
  ```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```


### 5. Entraîner les modèles
Ouvre le dossier `notebook` et exécute `EDA.ipynb` pour entraîner les modèles. À l'issue de l'exécution, place les deux fichiers générés dans le dossier `models/` à la racine du projet :

```text
models/
├── best_brain_tumor_cnn.keras
└── cnn_reference.keras
```



### 6. Lancer l'application
```bash
streamlit run app.py
```