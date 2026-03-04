from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Primary dataset
RAW_RECIPES_CSV = DATA_DIR / "RAW_recipes.csv"

# Trained artifacts
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
KNN_PATH = MODELS_DIR / "knn_index.joblib"
RECIPE_LOOKUP_PATH = MODELS_DIR / "recipe_lookup.csv"