import ast
import re
from typing import List

import pandas as pd

from .config import RAW_RECIPES_CSV

## load data
def load_raw_recipes(path=RAW_RECIPES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize id to int where possible
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

    return df


def _parse_list_cell(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    s = str(cell).strip()
    
    if not s:
        return []

    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val if str(x).strip()]
        if isinstance(val, str):
            return [val]
    except Exception:
        pass

    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


_UNITS = {
    "teaspoon", "teaspoons", "tsp",
    "tablespoon", "tablespoons", "tbsp",
    "cup", "cups",
    "pint", "pints",
    "quart", "quarts",
    "gallon", "gallons",
    "ounce", "ounces", "oz",
    "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g",
    "kilogram", "kilograms", "kg",
    "ml", "l", "liter", "liters",
    "pinch", "dash",
    "clove", "cloves",
    "slice", "slices",
    "can", "cans",
    "package", "packages", "pkg",
    "stick", "sticks",
}


def _clean_ingredient(ing: str) -> str:
    
    s = ing.lower().strip()

    # Remove parenthetical notes
    s = re.sub(r"\([^\)]*\)", " ", s)

    # Remove digits/fractions/punct (keep letters, spaces, hyphen)
    s = re.sub(r"[^a-z\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Drop common unit words
    parts = [p for p in s.split() if p and p not in _UNITS]
    return " ".join(parts).strip()

## id = #### doc = ingredients, n_tokens = number of tokens
def build_recipe_docs_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in raw_df.columns or "ingredients" not in raw_df.columns:
        raise ValueError("RAW_recipes.csv must include 'id' and 'ingredients' columns.")

    docs = []
    for _, row in raw_df[["id", "ingredients"]].iterrows():
        rid = int(row["id"])
        ing_list = _parse_list_cell(row["ingredients"])

        cleaned = []
        for ing in ing_list:
            c = _clean_ingredient(str(ing))
            if c:
                cleaned.append(c)

        doc = " ".join(cleaned)
        docs.append((rid, doc, len(cleaned)))

    out = pd.DataFrame(docs, columns=["id", "doc", "n_tokens"])
    out = out[out["doc"].astype(str).str.strip() != ""].reset_index(drop=True)
    return out