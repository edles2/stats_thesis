# 1) Créer un environnement (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate       # sur macOS / Linux
# .venv\Scripts\activate        # sur Windows

# 2) Installer les dépendances
pip install -r requirements.txt

# 3) Lancer l'app Streamlit
streamlit run streamlit.py