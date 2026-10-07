"""Tools for Pantry Pal, the weekly meal-planning agent.

1. search_recipes         -> TheMealDB (free, no key)                  [external API]
2. estimate_recipe_cost   -> Trader Joe's price CSV + kitchen tracking [original]
3. suggest_leftover_meals -> reuses leftovers, prices only what's new  [original]
4. weekly_summary         -> running total + cost-per-meal for the week [original]
5. suggest_wine_pairing   -> WineAPI (api.wineapi.io)                  [external API]
6. update_pantry          -> groceries the user already owns (cost $0) [original]
7. suggest_pantry_meals   -> recipes built around what the user owns   [original]

app.py hands the model TOOLS (the schemas) and calls run_tool(name, args) for each request
the model makes. run_tool always returns a JSON string. When something goes wrong it returns
{"error": ..., "hint": ...} instead of raising, so the model can read what failed and recover
or explain it to the user.
"""
import json
import math
import os
import re
from contextvars import ContextVar

import requests

from prices import CatalogError, find_product, is_staple, parse_count, parse_measure

MEALDB = "https://www.themealdb.com/api/json/v1/1"
TIMEOUT = 10

# ----------------------------------------------------------------------------
# Per-session kitchen. app.py binds a session's state before each /chat call, so
# every browser session has its own leftovers and cart.
# ----------------------------------------------------------------------------
_STATE: ContextVar = ContextVar("state", default=None)
_FALLBACK_STATE = {"stock": {}, "meals": [], "pantry": {}}  # used outside a web request (tests)


def new_state() -> dict:
    """stock: ingredient -> {"g": grams left, "ct": whole items left, "some": a package is
    left but the amount is unknown}. meals: meals committed to the cart. pantry: groceries
    the user already owns -> how many more meals each can be used in (free until it hits 0)."""
    return {"stock": {}, "meals": [], "pantry": {}}


def begin_request(state: dict):
    return _STATE.set(state)


def end_request(token):
    _STATE.reset(token)


def _state() -> dict:
    s = _STATE.get()
    return _FALLBACK_STATE if s is None else s


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _get_meal(meal_id: str):
    r = requests.get(f"{MEALDB}/lookup.php", params={"i": meal_id}, timeout=TIMEOUT)
    r.raise_for_status()
    meals = r.json().get("meals")
    return meals[0] if meals else None


def _ingredients(meal: dict):
    out = []
    for i in range(1, 21):
        name = (meal.get(f"strIngredient{i}") or "").strip()
        if name:
            out.append((name, (meal.get(f"strMeasure{i}") or "").strip()))
    return out


def _owned_match(ingredient: str, pantry: dict):
    """The pantry item that covers this recipe ingredient, or None.

    A pantry item covers an ingredient when all of the item's words appear in the ingredient's
    name: "rice" covers "Basmati Rice", "eggs" covers "Eggs". Only that direction is used, so
    "chicken stock" does not wrongly cover "Chicken". Items with no meals left are skipped.
    """
    ing = _words(ingredient)
    for item, meals_left in pantry.items():
        words = _words(item)
        if meals_left > 0 and words and words <= ing:
            return item
    return None


ASSUMED_FRACTION_USED = 0.5  # when a recipe's amount can't be measured, assume half a package
RECIPE_SERVES = 4  # TheMealDB doesn't say how many a recipe serves; most feed about 4


def _plan(measure: str, product: dict):
    """How to measure this ingredient against the package: ("g"|"ct", amount, package_size)."""
    grams = parse_measure(measure)
    if grams is not None and product["size_g"]:
        return "g", grams, product["size_g"]
    count = parse_count(measure)
    if count is not None and product["size_ct"]:
        return "ct", count, product["size_ct"]
    return None


def _has_left(entry: dict) -> bool:
    return entry["g"] > 5 or entry["ct"] >= 1 or entry["some"]


def _leftover_view(stock: dict) -> dict:
    out = {}
    for name, e in stock.items():
        parts = []
        if e["g"] > 5:
            parts.append(f"{round(e['g'])} g")
        if e["ct"] >= 1:
            parts.append(f"{round(e['ct'])} ct")
        if e["some"]:
            parts.append("some left (amount unknown)")
        if parts:
            out[name] = ", ".join(parts)
    return out


def _leftover_names(stock: dict, limit: int = 6) -> list:
    names = [k for k, e in stock.items() if _has_left(e)]
    names.sort(key=lambda k: (-stock[k]["g"], -stock[k]["ct"]))
    return names[:limit]


def _price_meal(meal: dict, servings: int, commit: bool = False) -> dict:
    """Price one recipe against the Trader Joe's catalog, using leftovers first.

    cash_to_spend - what you'd pay at the register (whole packages, minus what's
                    already in the kitchen)
    value_used    - the prorated cost of just the amount the recipe uses

    Ingredients are measured in grams or whole items when the recipe amount and the package
    size allow it. Otherwise the amount is unknown: the first time, one package is bought and
    about half is assumed used (the rest becomes an "unknown amount" leftover); a later recipe
    that needs the same ingredient reuses that leftover for free. Those lines are marked low
    confidence.
    """
    stock = _state()["stock"]
    pantry = _state().get("pantry", {})
    matched = set()  # pantry items this meal uses (each counts once per meal)
    working = {k: dict(v) for k, v in stock.items()}  # copy, so commit=False has no side effects
    lines, cash, used_total, unmatched, owned = [], 0.0, 0.0, [], []

    for name, measure in _ingredients(meal):
        key = name.lower()
        if is_staple(name):
            lines.append({"ingredient": name, "measure": measure, "spend": 0.0,
                          "value_used": 0.0, "note": "pantry staple, assumed on hand"})
            continue

        match = _owned_match(name, pantry)
        if match:
            owned.append(name)
            matched.add(match)
            lines.append({"ingredient": name, "measure": measure, "spend": 0.0, "value_used": 0.0,
                          "note": f"already owned (matched your '{match}', "
                                  f"{pantry[match]} meal(s) of use left, counting this one)"})
            continue

        product = find_product(name)
        if not product:
            unmatched.append(name)
            lines.append({"ingredient": name, "measure": measure, "spend": 0.0,
                          "value_used": 0.0, "note": "no match in the price list"})
            continue

        entry = working.setdefault(key, {"g": 0.0, "ct": 0.0, "some": False})
        plan = _plan(measure, product)
        note = None
        if plan:
            dim, need, size = plan
            need *= servings / RECIPE_SERVES  # scale the recipe's amounts to the party size
            to_buy = max(0.0, need - entry[dim])
            packs = math.ceil(to_buy / size) if to_buy > 0 else 0
            spend = packs * product["price"]
            value_used = product["price"] * need / size
            entry[dim] += packs * size - need
            confidence = "high"
        elif entry["some"]:
            spend = 0.0
            value_used = product["price"] * ASSUMED_FRACTION_USED
            entry["some"] = False
            confidence, note = "low", "reused a leftover from an earlier meal; amount assumed"
        else:
            spend = product["price"]
            value_used = product["price"] * ASSUMED_FRACTION_USED
            entry["some"] = True
            confidence, note = "low", ("amount couldn't be measured; assumed about half a "
                                       "package is used")

        cash += spend
        used_total += value_used
        line = {"ingredient": name, "measure": measure, "product": product["name"],
                "package_price": product["price"], "spend": round(spend, 2),
                "value_used": round(value_used, 2), "confidence": confidence}
        if plan:
            line["amount_for_servings"] = f"{round(need)} {'g' if dim == 'g' else 'ct'}"
        if note:
            line["note"] = note
        lines.append(line)

    if commit:
        stock.clear()
        stock.update(working)
        for item in matched:  # one use of each owned item per committed meal
            pantry[item] = max(0, pantry[item] - 1)

    return {
        "lines": lines,
        "cash_to_spend": round(cash, 2),
        "cash_per_serving": round(cash / servings, 2),
        "value_used": round(used_total, 2),
        "value_used_per_serving": round(used_total / servings, 2),
        "unmatched_ingredients": unmatched,
        "owned_ingredients": owned,
        "pantry_meals_left": dict(pantry),
    }


# ----------------------------------------------------------------------------
# tool 1: search_recipes (external API)
# ----------------------------------------------------------------------------
def search_recipes(query: str, search_by: str = "name", max_results: int = 5) -> dict:
    endpoints = {"name": ("search.php", "s"), "ingredient": ("filter.php", "i"),
                 "cuisine": ("filter.php", "a"), "category": ("filter.php", "c")}
    if search_by not in endpoints:
        return {"error": f"Unknown search_by '{search_by}'.",
                "hint": "Use one of: name, ingredient, cuisine, category."}
    path, param = endpoints[search_by]
    q = query.strip().replace(" ", "_") if path == "filter.php" else query.strip()
    r = requests.get(f"{MEALDB}/{path}", params={param: q}, timeout=TIMEOUT)
    r.raise_for_status()
    meals = r.json().get("meals") or []
    if not meals:
        return {"results": [],
                "hint": f"No recipes matched '{query}' by {search_by}. Try a broader word, "
                        "a single main ingredient (search_by='ingredient'), or a different "
                        "search_by."}
    max_results = max(1, min(int(max_results), 10))
    return {"results": [{"meal_id": m["idMeal"], "name": m["strMeal"],
                         "category": m.get("strCategory"), "cuisine": m.get("strArea")}
                        for m in meals[:max_results]]}


# ----------------------------------------------------------------------------
# tool 2: estimate_recipe_cost (original)
# ----------------------------------------------------------------------------
def estimate_recipe_cost(meal_id: str, servings: int = 2, add_to_cart: bool = False) -> dict:
    servings = int(servings)
    if servings < 1:
        return {"error": "servings must be at least 1.", "hint": "Retry with servings=1 or more."}
    meal = _get_meal(meal_id)
    if not meal:
        return {"error": f"No recipe found with meal_id '{meal_id}'.",
                "hint": "Call search_recipes to get a valid meal_id."}
    priced = _price_meal(meal, servings, commit=add_to_cart)
    state = _state()
    if add_to_cart:
        state["meals"].append({"meal": meal["strMeal"], "servings": servings,
                               "cash_spent": priced["cash_to_spend"]})
    priced.update({
        "headline": (f"For {servings} serving(s): ${priced['cash_to_spend']:.2f} to spend at the "
                     f"register, which is ${priced['cash_per_serving']:.2f} per serving."),
        "meal": meal["strMeal"], "servings": servings, "added_to_cart": add_to_cart,
        "assumed_recipe_serves": RECIPE_SERVES,
        "leftovers": _leftover_view(state["stock"]),
        "note": ("Estimates from a Trader Joe's price list. The recipe's amounts were scaled "
                 f"from an assumed {RECIPE_SERVES} servings to {servings}. 'low' confidence "
                 "lines have amounts that could not be measured, so they are rough."),
    })
    return priced


# ----------------------------------------------------------------------------
# shared by tools 3 and 7: find recipes built around ingredients the user has on hand
# ----------------------------------------------------------------------------
def _recipe_ids_with(name: str) -> list:
    """TheMealDB recipe ids that use an ingredient. Tries the singular/plural spelling too."""
    tries = [name]
    tries.append(name[:-1] if name.endswith("s") else name + "s")
    for term in tries:
        r = requests.get(f"{MEALDB}/filter.php", params={"i": term.replace(" ", "_")},
                         timeout=TIMEOUT)
        r.raise_for_status()
        meals = r.json().get("meals") or []
        if meals:
            return [m["idMeal"] for m in meals]
    return []


def _candidates(names: list, servings: int) -> list:
    votes: dict = {}
    for ing in names[:6]:
        for mid in _recipe_ids_with(ing):
            votes[mid] = votes.get(mid, 0) + 1
    out = []
    for mid in sorted(votes, key=votes.get, reverse=True)[:8]:
        meal = _get_meal(mid)
        if meal:  # what-if pricing only, nothing is committed
            out.append({"meal_id": mid, "meal": meal, "votes": votes[mid],
                        "priced": _price_meal(meal, servings)})
    return out


# ----------------------------------------------------------------------------
# tool 3: suggest_leftover_meals (original)
# ----------------------------------------------------------------------------
def suggest_leftover_meals(max_results: int = 3, servings: int = 2) -> dict:
    servings = int(servings)
    if servings < 1:
        return {"error": "servings must be at least 1.", "hint": "Retry with servings=1 or more."}
    stock = _state()["stock"]
    leftovers = _leftover_names(stock)
    if not leftovers:
        return {"error": "No leftovers are tracked yet.",
                "hint": "Call estimate_recipe_cost with add_to_cart=true for the meal the user "
                        "chose, then call this tool again."}

    view = _leftover_view(stock)
    shown = {k: view[k] for k in leftovers if k in view}
    scored = [{"meal_id": c["meal_id"], "name": c["meal"]["strMeal"],
               "leftover_ingredients_used": c["votes"],
               "extra_cash_per_serving": c["priced"]["cash_per_serving"],
               "extra_cash_total": c["priced"]["cash_to_spend"],
               "unmatched_ingredients": c["priced"]["unmatched_ingredients"]}
              for c in _candidates(leftovers, servings)]
    if not scored:
        return {"leftovers": shown, "suggestions": [],
                "hint": "No recipes in the database use these leftovers. Tell the user, and "
                        "offer to search for a new recipe instead."}

    scored.sort(key=lambda s: (s["extra_cash_per_serving"], -s["leftover_ingredients_used"]))
    max_results = max(1, min(int(max_results), 5))
    return {"leftovers": shown, "suggestions": scored[:max_results]}


# ----------------------------------------------------------------------------
# tool 6: update_pantry (original)
# ----------------------------------------------------------------------------
MAX_PANTRY = 50


def _clean_items(items) -> list:
    if isinstance(items, str):
        items = [items]
    out = []
    for x in items or []:
        if isinstance(x, str) and x.strip():
            out.append(x.strip().lower()[:40])
    return out


DEFAULT_PANTRY_MEALS = 2  # each owned item can be used in this many more meals unless told otherwise


def update_pantry(add_items=None, remove_items=None, max_meals: int = DEFAULT_PANTRY_MEALS) -> dict:
    pantry = _state().setdefault("pantry", {})
    try:
        meals = max(1, min(int(max_meals), 10))
    except (TypeError, ValueError):
        meals = DEFAULT_PANTRY_MEALS
    added, removed, not_found = [], [], []
    for item in _clean_items(add_items):
        if item not in pantry and len(pantry) >= MAX_PANTRY:
            return {"error": f"The pantry is full ({MAX_PANTRY} items).",
                    "hint": "Ask the user which items to remove first."}
        pantry[item] = meals  # adding an item again counts as restocking it
        added.append(item)
    for item in _clean_items(remove_items):
        if item in pantry:
            del pantry[item]
            removed.append(item)
        else:
            not_found.append(item)
    result = {"pantry_meals_left": dict(pantry), "added": added, "removed": removed}
    if added:
        result["each_added_item_covers_meals"] = meals
    if not_found:
        result["not_found"] = not_found
    if not any(v > 0 for v in pantry.values()):
        result["hint"] = ("Nothing usable is in the pantry. Ask the user what groceries they "
                          "already have.")
    return result


# ----------------------------------------------------------------------------
# tool 7: suggest_pantry_meals (original)
# ----------------------------------------------------------------------------
def suggest_pantry_meals(max_results: int = 3, servings: int = 2) -> dict:
    servings = int(servings)
    if servings < 1:
        return {"error": "servings must be at least 1.", "hint": "Retry with servings=1 or more."}
    pantry = _state().get("pantry", {})
    if not pantry:
        return {"error": "The user hasn't told you what groceries they already have.",
                "hint": "Ask what they have at home, record it with update_pantry, then call "
                        "this tool again."}
    active = [item for item, meals_left in pantry.items() if meals_left > 0]
    if not active:
        return {"error": "Every grocery the user owns has been used up.",
                "hint": "Tell the user, and ask whether they have restocked anything. If so, "
                        "record it with update_pantry."}

    scored = [{"meal_id": c["meal_id"], "name": c["meal"]["strMeal"],
               "owned_ingredients": c["priced"]["owned_ingredients"],
               "cash_per_serving": c["priced"]["cash_per_serving"],
               "cash_to_spend": c["priced"]["cash_to_spend"],
               "unmatched_ingredients": c["priced"]["unmatched_ingredients"]}
              for c in _candidates(active, servings)]
    if not scored:
        return {"pantry_meals_left": dict(pantry), "suggestions": [],
                "hint": "No recipes in the database were found for these items. Tell the user, "
                        "and offer to search by cuisine or category instead."}

    scored.sort(key=lambda s: (s["cash_per_serving"], -len(s["owned_ingredients"])))
    max_results = max(1, min(int(max_results), 5))
    return {"pantry_meals_left": dict(pantry), "suggestions": scored[:max_results]}


# ----------------------------------------------------------------------------
# tool 4: weekly_summary (original)
# ----------------------------------------------------------------------------
def weekly_summary() -> dict:
    state = _state()
    meals = state["meals"]
    if not meals:
        return {"error": "No meals have been added to the cart yet.",
                "hint": "Help the user pick a recipe, then call estimate_recipe_cost with "
                        "add_to_cart=true."}
    total = round(sum(m["cash_spent"] for m in meals), 2)
    servings = sum(m["servings"] for m in meals)
    return {"meals": meals, "total_cash_spent": total, "total_servings": servings,
            "average_cash_per_serving": round(total / servings, 2),
            "leftovers": _leftover_view(state["stock"]),
            "pantry_meals_left": dict(state.get("pantry", {}))}


# ----------------------------------------------------------------------------
# tool 5: suggest_wine_pairing (external API) - WineAPI, https://api.wineapi.io
# ----------------------------------------------------------------------------
# WineAPI cannot search by food. It searches wines by name, winery, or appellation, and
# each wine has a list of foods it pairs with. So the model picks a wine style for the dish,
# and this tool (1) searches for real wines of that style, then (2) checks each wine's own
# food-pairing list against the dish. Free plan = 100 calls/day, so results are cached and
# each use costs at most 1 search + 3 pairing lookups.
WINEAPI = "https://api.wineapi.io"
MAX_WINES = 3
_STOPWORDS = {"and", "with", "the", "for", "from", "alla", "alle", "del", "della", "con"}
_pairing_cache: dict = {}  # wine_id -> pairings list (shared across sessions)
_wine_cache: dict = {}     # (meal_id, wine_query) -> finished result


def _words(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude singularize
        if len(w) >= 3 and w not in _STOPWORDS:
            out.add(w)
    return out


def _wine_get(path: str, params: dict | None = None):
    r = requests.get(f"{WINEAPI}{path}", headers={"X-API-Key": os.environ["WINEAPI_KEY"]},
                     params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _pairings(wine_id: str) -> list:
    if wine_id not in _pairing_cache:
        _pairing_cache[wine_id] = _wine_get(f"/wines/{wine_id}/pairings").get("pairings") or []
    return _pairing_cache[wine_id]


def suggest_wine_pairing(meal_id: str, wine_query: str) -> dict:
    if not os.getenv("WINEAPI_KEY"):
        return {"error": "Wine pairing is not configured on the server (no WINEAPI_KEY).",
                "hint": "Tell the user wine pairing is unavailable right now and carry on."}
    wine_query = (wine_query or "").strip()
    if len(wine_query) < 2:
        return {"error": "wine_query is empty.",
                "hint": "Choose a wine style or appellation that suits the dish, such as "
                        "'Chianti' or 'Sancerre', and call again."}

    cache_key = (str(meal_id), wine_query.lower())
    if cache_key in _wine_cache:
        return _wine_cache[cache_key]

    meal = _get_meal(meal_id)
    if not meal:
        return {"error": f"No recipe found with meal_id '{meal_id}'.",
                "hint": "Call search_recipes to get a valid meal_id."}

    # Words that describe the dish: its name, category, and non-staple ingredients.
    dish_terms = _words(meal["strMeal"]) | _words(meal.get("strCategory") or "")
    for name, _ in _ingredients(meal):
        if not is_staple(name):
            dish_terms |= _words(name)

    found = _wine_get("/wines/search", {"q": wine_query, "limit": 10}).get("results") or []
    if not found:
        return {"wines": [],
                "hint": f"No wines matched '{wine_query}'. The wine database searches wine "
                        "names, wineries, and appellations, not foods or moods. Try an "
                        "appellation or well-known wine name (for example 'Chianti', 'Rioja', "
                        "'Sancerre') and call again."}

    # Best-rated distinct wines first (the same wine can appear once per vintage).
    picks, seen = [], set()
    for w in sorted(found[:8], key=lambda w: -(w.get("averageRating") or 0)):
        name = (w.get("name") or "").lower()
        if name not in seen:
            seen.add(name)
            picks.append(w)
        if len(picks) == MAX_WINES:
            break

    wines, rate_limited = [], False
    for w in picks:
        pairings = None
        if not rate_limited:
            try:
                pairings = _pairings(w["id"])
            except requests.HTTPError as err:
                rate_limited = err.response is not None and err.response.status_code == 429
            except requests.RequestException:
                pass
        entry = {"name": w.get("name"), "vintage": w.get("vintage"), "type": w.get("type"),
                 "winery": w.get("winery"), "region": w.get("region"),
                 "country": w.get("country"), "average_rating": w.get("averageRating"),
                 "ratings_count": w.get("ratingsCount")}
        if pairings is None:
            entry.update({"pairings_checked": False, "pairing_matches": [],
                          "pairing_match_count": 0})
        else:
            hits = [p for p in pairings if _words(p.get("food")) & dish_terms]
            rest = [p.get("food") for p in pairings if p not in hits][:4]
            entry.update({
                "pairings_checked": True,
                "pairing_matches": [{"food": p.get("food"), "notes": p.get("notes")}
                                    for p in hits],
                "pairing_match_count": len(hits),
                "other_listed_pairings": rest,
            })
        wines.append(entry)

    wines.sort(key=lambda e: (-e["pairing_match_count"], -(e["average_rating"] or 0)))
    result = {
        "meal": meal["strMeal"], "searched_for": wine_query, "wines": wines,
        "how_to_read": ("pairing_matches are foods the wine database lists for that wine that "
                        "share a word with this dish or its main ingredients. No matches means "
                        "no overlapping pairing is listed, not that the wine is a bad choice."),
    }
    if rate_limited:
        result["note"] = ("The wine service's daily limit was reached part-way, so some wines "
                          "have no pairing data. Say so if you mention them.")
    else:
        _wine_cache[cache_key] = result
    return result


# ----------------------------------------------------------------------------
# schemas the model sees (OpenAI function-calling format, which LiteLLM passes to Gemini)
# ----------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_recipes",
        "description": (
            "Search a recipe database (TheMealDB) and return matching recipes with their "
            "meal_ids. Use this first, whenever the user mentions a craving, dish, ingredient, "
            "or cuisine. The meal_id in each result is what estimate_recipe_cost and "
            "suggest_wine_pairing need."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "The search term, for example 'chicken', 'Italian', "
                                     "'Seafood', or 'carbonara'."},
            "search_by": {"type": "string", "enum": ["name", "ingredient", "cuisine", "category"],
                          "description": "How to interpret query: 'name' for a dish name, "
                                         "'ingredient' for one main ingredient, 'cuisine' for "
                                         "a style such as Italian, Mexican, Indian, Thai, "
                                         "Japanese, Chinese or French, 'category' for a type "
                                         "such as Beef, Chicken, Pasta, Seafood, Vegetarian, "
                                         "Vegan or Dessert. Defaults to 'name'."},
            "max_results": {"type": "integer",
                            "description": "How many recipes to return, from 1 to 10. "
                                           "Defaults to 5."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "estimate_recipe_cost",
        "description": (
            "Price a recipe using Trader Joe's prices. Returns each ingredient's cost, "
            "cash_to_spend (whole packages the user would buy, minus leftovers already in "
            "their kitchen), the prorated value_used, and per-serving figures. Call it with "
            "add_to_cart=false to preview a recipe. Once the user commits to cooking it, call "
            "it again with add_to_cart=true so its leftovers are tracked and later meals are "
            "priced against them."),
        "parameters": {"type": "object", "properties": {
            "meal_id": {"type": "string", "description": "The meal_id from search_recipes."},
            "servings": {"type": "integer",
                         "description": "How many servings to divide the cost by (at least 1). "
                                        "Defaults to 2."},
            "add_to_cart": {"type": "boolean",
                            "description": "True only after the user has chosen this meal. "
                                           "Defaults to false."},
        }, "required": ["meal_id"]}}},
    {"type": "function", "function": {
        "name": "suggest_leftover_meals",
        "description": (
            "Suggest other recipes that reuse leftover ingredients from meals already in the "
            "cart. Each suggestion shows extra_cash_per_serving, which is what the user would "
            "still have to buy, since leftovers cost nothing more. Only works after at least "
            "one meal was committed with estimate_recipe_cost(add_to_cart=true). Use it to "
            "show how the cost per meal drops as the week goes on."),
        "parameters": {"type": "object", "properties": {
            "max_results": {"type": "integer",
                            "description": "How many suggestions to return, from 1 to 5. "
                                           "Defaults to 3."},
            "servings": {"type": "integer",
                         "description": "How many servings to divide the extra cost by (at "
                                        "least 1). Defaults to 2."},
        }}}},
    {"type": "function", "function": {
        "name": "update_pantry",
        "description": (
            "Record groceries the user already has at home, or remove ones they no longer have. "
            "Owned ingredients cost nothing in a cost breakdown, but each one only covers a "
            "limited number of meals (2 by default, one use per meal you commit with "
            "add_to_cart=true); after that it is priced normally. Call it whenever the user "
            "mentions what they already have, BEFORE searching or pricing recipes. Adding an "
            "item again restocks it. Returns the pantry with the meals each item has left. Call "
            "it with no arguments to just see the pantry."),
        "parameters": {"type": "object", "properties": {
            "add_items": {"type": "array", "items": {"type": "string"},
                          "description": "Groceries the user already has, one per entry, plain "
                                         "ingredient names without quantities, for example "
                                         "['eggs', 'rice', 'chicken breast']."},
            "remove_items": {"type": "array", "items": {"type": "string"},
                             "description": "Groceries to take out of the pantry, spelled the "
                                            "same way they were added."},
            "max_meals": {"type": "integer",
                          "description": "How many more meals each item in add_items can be "
                                         "used in, from 1 to 10. Defaults to 2. Only set it if "
                                         "the user says otherwise, for example 'a big bag of "
                                         "rice, enough for 5 meals'."},
        }}}},
    {"type": "function", "function": {
        "name": "suggest_pantry_meals",
        "description": (
            "Suggest recipes built around the groceries the user already owns (recorded with "
            "update_pantry). Each suggestion lists which ingredients are already owned and what "
            "the user would still need to buy per serving, cheapest first. Use it when the user "
            "asks what they can make with what they have or wants the cheapest option."),
        "parameters": {"type": "object", "properties": {
            "max_results": {"type": "integer",
                            "description": "How many suggestions to return, from 1 to 5. "
                                           "Defaults to 3."},
            "servings": {"type": "integer",
                         "description": "How many servings to divide the cost by (at least 1). "
                                        "Defaults to 2."},
        }}}},
    {"type": "function", "function": {
        "name": "weekly_summary",
        "description": (
            "Summarize every meal committed to the cart this session: what each meal cost at "
            "the register, the total, the average cost per serving, and what is left in the "
            "kitchen. Use it when the user asks how their week is looking, what they have "
            "spent, or for a recap."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "suggest_wine_pairing",
        "description": (
            "Find real wines to serve with a recipe. The wine database cannot search by food, "
            "so first choose a wine style or appellation that suits the dish, then pass it as "
            "wine_query. The tool finds matching wines and checks each one's listed food "
            "pairings against the dish. Only call it when the user asks about wine or says yes "
            "to your offer (the service has a small daily limit). Only recommend wines that "
            "appear in the result."),
        "parameters": {"type": "object", "properties": {
            "meal_id": {"type": "string",
                        "description": "The meal_id from search_recipes of the meal to pair a "
                                       "wine with."},
            "wine_query": {"type": "string",
                           "description": "A wine style, appellation, or well-known wine name "
                                          "that you think suits this dish, for example "
                                          "'Chianti', 'Pinot Grigio', 'Sancerre', or 'Rioja'. "
                                          "It searches wine names, wineries, and appellations, "
                                          "not foods."},
        }, "required": ["meal_id", "wine_query"]}}},
]

_FUNCS = {
    "search_recipes": search_recipes,
    "estimate_recipe_cost": estimate_recipe_cost,
    "suggest_leftover_meals": suggest_leftover_meals,
    "update_pantry": update_pantry,
    "suggest_pantry_meals": suggest_pantry_meals,
    "weekly_summary": weekly_summary,
    "suggest_wine_pairing": suggest_wine_pairing,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool and return its result as a JSON string. Never raises."""
    fn = _FUNCS.get(name)
    if fn is None:
        result = {"error": f"There is no tool named '{name}'.",
                  "hint": f"Use one of: {', '.join(_FUNCS)}."}
    else:
        try:
            result = fn(**(args or {}))
        except (FileNotFoundError, CatalogError) as e:
            detail = ("The Trader Joe's price file is missing on the server."
                      if isinstance(e, FileNotFoundError) else str(e))
            result = {"error": detail,
                      "hint": "Tell the user prices are unavailable right now. "
                              "You can still search recipes and suggest wine."}
        except requests.Timeout:
            result = {"error": "The external service timed out.",
                      "hint": "Retry once. If it fails again, tell the user and carry on "
                              "without it."}
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "unknown"
            if code == 429:
                result = {"error": "The external service's rate limit was reached.",
                          "hint": "Tell the user this feature is unavailable for now and carry "
                                  "on without it. Do not retry."}
            elif code in (401, 403):
                result = {"error": f"The external service rejected our credentials (HTTP {code}).",
                          "hint": "Tell the user this feature is unavailable and carry on."}
            elif code == 404:
                result = {"error": "The external service could not find that item (HTTP 404).",
                          "hint": "Check the id you passed. For recipes, call search_recipes "
                                  "to get a valid meal_id."}
            else:
                result = {"error": f"The external service returned HTTP {code}.",
                          "hint": "Retry once, then tell the user and carry on."}
        except requests.RequestException as e:
            result = {"error": f"Network problem: {type(e).__name__}.",
                      "hint": "Retry once, then tell the user the service is unreachable."}
        except TypeError as e:  # wrong or unexpected arguments from the model
            result = {"error": f"Bad arguments for {name}: {e}",
                      "hint": "Check the tool's parameters and call it again."}
        except Exception as e:  # never crash the chat because of one tool
            result = {"error": f"{type(e).__name__}: {e}",
                      "hint": "Check the arguments and try again, or explain the problem "
                              "to the user."}
    return json.dumps(result, default=str)
