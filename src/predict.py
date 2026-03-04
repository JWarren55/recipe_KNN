from typing import List, Tuple
import joblib
import pandas as pd
import re
import difflib
from .config import RAW_RECIPES_CSV, VECTORIZER_PATH, KNN_PATH, RECIPE_LOOKUP_PATH

## predict the recipe
def predict_knn(ingredients: List[str] | str, topk: int = 5) -> pd.DataFrame:
    
    if isinstance(ingredients, list):
        q = " ".join([str(x).strip().lower() for x in ingredients if str(x).strip()])
    else:
        q = str(ingredients).strip().lower()
    if not q:
        raise ValueError("No ingredients provided.")
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    knn = joblib.load(KNN_PATH)
    recipes = pd.read_csv(RECIPE_LOOKUP_PATH)
    
    full = pd.read_csv(RAW_RECIPES_CSV, usecols=["id"])
    full_ids = set(full["id"].astype(int))

    ## looks for bad input hen can be better
    known, unknown = analyze_ingredients_against_vocab(ingredients, vectorizer)

    if unknown:
        print("\n[!] Some inputs were not recognized in the training vocabulary (they won’t help matching):")
        for bad, sugg in unknown.items():
            if sugg:
                print(f"  - '{bad}'  → did you mean: {', '.join(sugg)} ?")
            else:
                print(f"  - '{bad}'  → no close match found (try simpler wording, singular form, or remove brand/quantity)")
        print()

    q_vec = vectorizer.transform([q])

    ## Gets the IDs that are also in csv file for recipe printing
    distances, indices = knn.kneighbors(q_vec, n_neighbors=min(len(recipes), topk * 10))

    idxs = indices[0].tolist()
    dists = distances[0].tolist()

    out = recipes.iloc[idxs].copy()
    out["cosine_distance"] = dists
    out["similarity"] = [1.0 - d for d in dists]

    out["id"] = out["id"].astype(int)
    out = out[out["id"].isin(full_ids)]
    out = out.sort_values("similarity", ascending=False).head(topk).reset_index(drop=True)

    return out[["id", "similarity", "n_tokens"]]

#######################################################################
def analyze_ingredients_against_vocab(ingredients: List[str] | str, vectorizer, max_suggestions: int = 5):
    """
    Returns:
      known: list[str]
      unknown: dict[str, list[str]]   # unknown ingredient -> suggestions
    """
    # Turn input into a list of ingredient phrases
    if isinstance(ingredients, list):
        items = [str(x) for x in ingredients]
    else:
        # allow comma-separated string
        items = [x.strip() for x in str(ingredients).split(",")]

    # Basic normalization: lowercase + keep letters/spaces only
    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z\s-]", "", s)       # drop numbers/punct (keeps letters, space, hyphen)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    items = [norm(x) for x in items if norm(x)]
    vocab = set(vectorizer.get_feature_names_out())  # includes unigrams + bigrams if you trained that way

    known = []
    unknown = {}

    for ing in items:
        tokens = ing.split()

        is_known = (ing in vocab) or any(t in vocab for t in tokens)
        if is_known:
            known.append(ing)
        else:
            # closest matches from vocab (good for typos/variants)
            unigram_vocab = [v for v in vocab if " " not in v]
            suggestions = difflib.get_close_matches(ing, unigram_vocab, n=max_suggestions, cutoff=0.75)

            # If phrase has multiple tokens, also suggest for the last token (often the main noun)
            if not suggestions and tokens:
                suggestions = difflib.get_close_matches(tokens[-1], unigram_vocab, n=max_suggestions, cutoff=0.75)

            unknown[ing] = suggestions

    return known, unknown


    
## uses the id to print the food
import ast

def print_recipe_by_id(recipe_id: int, similarity: float | None = None):
    full_recipes = pd.read_csv(RAW_RECIPES_CSV)

    row = full_recipes.loc[full_recipes["id"] == recipe_id]
    if row.empty:
        print(f"No recipe found for id={recipe_id}")
        return

    r = row.iloc[0]

    print("\n" + "=" * 60)
    print(f"{r['name']}  (id={recipe_id})")

    if similarity is not None:
        print(f"Similarity: {similarity:.4f}")

    print(f"Minutes: {r['minutes']}")
    print("-" * 60)

    print("\nIngredients:")
    try:
        ing_list = ast.literal_eval(r["ingredients"]) if isinstance(r["ingredients"], str) else []
    except Exception:
        ing_list = []
    for ing in ing_list:
        s = str(ing).strip()
        if s:
            print(" .", s)

    print("\nSteps:")
    try:
        steps_list = ast.literal_eval(r["steps"]) if isinstance(r["steps"], str) else []
    except Exception:
        steps_list = []
    for i, step in enumerate(steps_list, 1):
        s = str(step).strip()
        if s:
            print(f" {i}. {s}")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    res = predict_knn(["pistachio", "peanut", "chicken"], topk=5)
    for _, row in res.iterrows():
        print_recipe_by_id(int(row["id"]), similarity=row["similarity"])