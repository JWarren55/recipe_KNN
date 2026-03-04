from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

from .data_load import load_raw_recipes, build_recipe_docs_from_raw
from .config import MODELS_DIR, VECTORIZER_PATH, KNN_PATH, RECIPE_LOOKUP_PATH


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ## load data
    raw = load_raw_recipes()
    recipes = build_recipe_docs_from_raw(raw)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
    )

    X = vectorizer.fit_transform(recipes["doc"])

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(X)

    # Save artifacts
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(knn, KNN_PATH)
    recipes.to_csv(RECIPE_LOOKUP_PATH, index=False)

    print("build done")
    print(f"recipes indexed: {len(recipes):,}")


if __name__ == "__main__":
    main()