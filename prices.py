"""Ingredient price lookup backed by a Trader Joe's product CSV.

The CSV has the columns: sku, retail_price, item_title, inserted_at, store_code, availability.
Only item_title and retail_price are used.
"""
import csv
import re
import statistics
from functools import lru_cache
from pathlib import Path

CSV_PATH = Path(__file__).parent / "traderjoes.csv"
COL_NAME = "item_title"
COL_PRICE = "retail_price"

# Rough conversions to grams (ml is treated as ~1 g; fine for an estimate).
UNIT_G = {
    "g": 1, "gram": 1, "grams": 1, "kg": 1000,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.6, "lbs": 453.6, "pound": 453.6, "pounds": 453.6,
    "ml": 1, "l": 1000, "liter": 1000, "floz": 29.57,
    "tbsp": 15, "tbs": 15, "tablespoon": 15, "tablespoons": 15,
    "tsp": 5, "teaspoon": 5, "teaspoons": 5,
    "cup": 240, "cups": 240,
}

# Assumed to already be in the kitchen, so they cost $0 in the estimate.
PANTRY_STAPLES = {
    "salt", "sea salt", "pepper", "black pepper", "water",
    "olive oil", "vegetable oil", "oil", "cooking oil",
}

# Map TheMealDB / British wording to what Trader Joe's calls things.
ALIASES = {
    "spring onions": "green onion",
    "coriander": "cilantro",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "plain flour": "flour",
    "double cream": "heavy cream",
    "minced beef": "ground beef",
    "egg yolks": "eggs",
    "egg whites": "eggs"
}

# Whole-item units: "2", "2 large", "3 fillets". Things like "cloves" or "slices" are NOT
# included because a title's "3 ct" of garlic means 3 bulbs, not 3 cloves.
COUNT_UNITS = {"", "large", "medium", "small", "whole", "egg", "eggs", "piece", "pieces",
               "fillet", "fillets", "breast", "breasts"}
TITLE_COUNT_RE = re.compile(r"(\d+)\s*(?:ct|count|pk|pack|pcs|pieces|eggs)\b", re.I)

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|oz|lbs?|kg|g|ml|l)\b", re.I)


def _num(s: str) -> float:
    s = s.strip()
    if " " in s and "/" in s:  # "1 1/2"
        whole, frac = s.split()
        n, d = frac.split("/")
        return float(whole) + float(n) / float(d)
    if "/" in s:
        n, d = s.split("/")
        return float(n) / float(d)
    return float(s)


def parse_measure(measure: str):
    """'200g' / '1 1/2 cups' / '2 tbsp' -> grams, or None for things like '2 cloves'."""
    m = re.match(r"\s*(\d+\s+\d+/\d+|\d+/\d+|\d*\.?\d+)\s*([a-zA-Z]+)?", measure or "")
    if not m:
        return None
    unit = (m.group(2) or "").lower()
    return _num(m.group(1)) * UNIT_G[unit] if unit in UNIT_G else None


def parse_count(measure: str):
    """'2' / '2 large' / '3 fillets' -> number of whole items, else None."""
    m = re.match(r"\s*(\d+\s+\d+/\d+|\d+/\d+|\d*\.?\d+)\s*([a-zA-Z]*)\s*$", measure or "")
    if not m or m.group(2).lower() not in COUNT_UNITS:
        return None
    return _num(m.group(1))


def title_count(text: str):
    """'Large Eggs, 12 ct' -> 12, else None."""
    m = TITLE_COUNT_RE.search(text or "")
    return float(m.group(1)) if m else None


def parse_size_g(text: str):
    m = SIZE_RE.search(text or "")
    if not m:
        return None
    unit = re.sub(r"[.\s]", "", m.group(2).lower())
    return float(m.group(1)) * UNIT_G[unit] if unit in UNIT_G else None


def _tokens(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude singularize
        out.add(w)
    return out


class CatalogError(RuntimeError):
    """The price file is unreadable or has the wrong columns."""


def _to_price(raw):
    try:
        value = float(re.sub(r"[^\d.]", "", raw or ""))
    except ValueError:
        return None
    return value if value > 0 else None


@lru_cache(maxsize=1)
def load_catalog():
    """One entry per distinct item_title.

    The CSV can hold the same product many times (different stores, different dates), so the
    price used is the MEDIAN retail_price across all rows for that title. Rows with a blank
    title or a missing/zero price are skipped.
    """
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        head = f.readline()
        f.seek(0)
        delimiter = max([",", "\t", ";", "|"], key=head.count)  # copes with CSV or TSV
        reader = csv.DictReader(f, delimiter=delimiter)
        reader.fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = {COL_NAME, COL_PRICE} - set(reader.fieldnames)
        if missing:
            raise CatalogError(f"Price file is missing column(s) {sorted(missing)}; "
                               f"found {reader.fieldnames}.")

        by_title: dict = {}
        for row in reader:
            title = (row.get(COL_NAME) or "").strip()
            price = _to_price(row.get(COL_PRICE))
            if not title or price is None:
                continue
            entry = by_title.setdefault(title.lower(), {"name": title, "prices": []})
            entry["prices"].append(price)

    items = []
    for entry in by_title.values():
        items.append({
            "name": entry["name"],
            "price": round(statistics.median(entry["prices"]), 2),
            "size_g": parse_size_g(entry["name"]),    # e.g. "Chicken Breasts, 16 oz"
            "size_ct": title_count(entry["name"]),    # e.g. "Large Eggs, 12 ct"
            "tokens": _tokens(entry["name"]),
        })
    if not items:
        raise CatalogError("Price file contained no usable rows.")
    return items


def is_staple(ingredient: str) -> bool:
    return ingredient.strip().lower() in PANTRY_STAPLES


def find_product(ingredient: str):
    """Best catalog match for an ingredient name, or None.

    Scoring: share of the ingredient's words found in the product name,
    with a small penalty for long names so plain items beat prepared ones.
    """
    ingredient = ALIASES.get(ingredient.strip().lower(), ingredient)
    want = _tokens(ingredient)
    if not want:
        return None
    best, best_score = None, 0.0
    for p in load_catalog():
        hit = len(want & p["tokens"]) / len(want)
        if hit == 0:
            continue
        score = hit - 0.01 * len(p["tokens"])
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 0.6 else None
