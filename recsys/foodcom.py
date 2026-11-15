"""Read the Food.com recipes (Kaggle, shuyangli94) into the frame the app uses.

The file is ``RAW_recipes.csv`` from "Food.com Recipes and Interactions". It
is not in the repository: download it into ``data/raw/kaggle/``. Rows come out
with the same columns as :func:`recsys.data.load_recipes`, so the two sets can
be joined.

The source is lower-cased and tokenised ("mom s chili", "stir well , then
cool"), so names, steps and descriptions are tidied for display. Cuisine and
course come from the recipe's tags. There are no photos and no quantities.
"""
from __future__ import annotations

import ast
import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import ROOT, canonical

FOODCOM = ROOT / "data" / "raw" / "kaggle" / "RAW_recipes.csv"
# Ids of the first dataset are its row numbers (under 7,000). Food.com ids go
# up to about 540,000, so this keeps the two apart and both readable.
ID_OFFSET = 1_000_000
MAX_MINUTES = 3 * 24 * 60

# Tag -> cuisine label. The labels are listed in recsys.taxonomy.CUISINE_MAP.
# Broad tags are used only when a recipe has no narrower one.
_BROAD_CUISINES = {
    "north-american": "North American", "european": "European", "asian": "Asian", "african": "African",
    "south-american": "South American", "central-american": "Central American",
    "middle-eastern": "Middle Eastern", "south-west-pacific": "South West Pacific", "caribbean": "Caribbean",
    "scandinavian": "Scandinavian", "american": "American", "canadian": "Canadian",
}
_CUISINES = {
    "southern-united-states": "Southern United States", "southwestern-united-states": "Southwestern United States",
    "midwestern": "Midwestern", "northeastern-united-states": "Northeastern United States",
    "pacific-northwest": "Pacific Northwest", "californian": "Californian", "cajun": "Cajun", "creole": "Creole",
    "tex-mex": "Tex Mex", "hawaiian": "Hawaiian", "native-american": "Native American",
    "pennsylvania-dutch": "Pennsylvania Dutch", "amish-mennonite": "Amish Mennonite", "soul": "Soul Food",
    "ontario": "Ontario", "quebec": "Quebec", "british-columbian": "British Columbian",
    "cuban": "Cuban", "puerto-rican": "Puerto Rican",
    "mexican": "Mexican", "oaxacan": "Oaxacan", "baja": "Baja",
    "brazilian": "Brazilian", "argentine": "Argentine", "peruvian": "Peruvian", "chilean": "Chilean",
    "colombian": "Colombian", "venezuelan": "Venezuelan", "ecuadorean": "Ecuadorean",
    "costa-rican": "Costa Rican", "guatemalan": "Guatemalan", "honduran": "Honduran",
    "italian": "Italian", "french": "French", "greek": "Greek", "english": "English", "scottish": "Scottish",
    "irish": "Irish", "welsh": "Welsh", "german": "German", "spanish": "Spanish", "portuguese": "Portuguese",
    "swedish": "Swedish", "danish": "Danish", "norwegian": "Norwegian", "finnish": "Finnish",
    "icelandic": "Icelandic", "polish": "Polish", "russian": "Russian", "hungarian": "Hungarian",
    "czech": "Czech", "austrian": "Austrian", "swiss": "Swiss", "belgian": "Belgian", "dutch": "Dutch",
    "georgian": "Georgian",
    "chinese": "Chinese", "szechuan": "Sichuan", "cantonese": "Cantonese", "hunan": "Hunan", "beijing": "Beijing",
    "mongolian": "Mongolian", "japanese": "Japanese", "korean": "Korean",
    "thai": "Thai", "vietnamese": "Vietnamese", "indonesian": "Indonesian", "malaysian": "Malaysian",
    "filipino": "Filipino", "cambodian": "Cambodian", "laotian": "Laotian",
    "indian": "Indian", "pakistani": "Pakistani", "nepalese": "Nepalese",
    "lebanese": "Lebanese", "turkish": "Turkish", "iranian-persian": "Persian", "saudi-arabian": "Saudi Arabian",
    "iraqi": "Iraqi", "palestinian": "Palestinian", "jewish-ashkenazi": "Jewish", "jewish-sephardi": "Jewish",
    "moroccan": "Moroccan", "south-african": "South African", "ethiopian": "Ethiopian", "egyptian": "Egyptian",
    "nigerian": "Nigerian", "libyan": "Libyan", "sudanese": "Sudanese", "angolan": "Angolan",
    "congolese": "Congolese", "somalian": "Somalian", "namibian": "Namibian",
    "australian": "Australian", "new-zealand": "New Zealand", "polynesian": "Polynesian",
    "micro-melanesia": "Micronesian",
}

# Tag -> course label, most telling first: an omelette tagged both
# "breakfast" and "main-dish" is a breakfast.
_COURSES = (
    ("desserts", "Dessert"), ("beverages", "Drink"), ("breakfast", "Breakfast"), ("soups-stews", "Soup"),
    ("salads", "Salad"), ("main-dish", "Main Course"), ("side-dishes", "Side Dish"),
    ("appetizers", "Appetizer"), ("snacks", "Snack"), ("breads", "Bread"),
    ("condiments-etc", "Sauce & Condiment"), ("brunch", "Brunch"), ("lunch", "Lunch"),
)

# Meat, fish and what is made from them. Eggs count as non-vegetarian, as in
# the first dataset. Matched as whole words on the canonical ingredient name,
# so "graham cracker" and "eggplant" are not hits.
_NON_VEG = frozenset("""
chicken hen poultry turkey duck goose quail pheasant cornish beef steak veal brisket sirloin chuck ribeye
oxtail pork ham bacon pancetta prosciutto sausage salami pepperoni chorizo kielbasa bratwurst knockwurst
liverwurst andouille mortadella bologna pastrami capicola guanciale lardon frankfurter wiener hotdog
hamburger mignon lamb mutton goat venison deer elk moose rabbit bison buffalo boar meat meatball
liver giblet gizzard kidney tripe tongue lard suet tallow dripping gelatin gelatine jello marshmallow
fish salmon tuna cod haddock halibut tilapia trout bass snapper catfish flounder sole mackerel sardine
anchovy anchovie herring swordfish mahi grouper perch pollock whitefish roughy caviar roe surimi eel
shrimp prawn crab crabmeat lobster crawfish crayfish scallop clam mussel oyster squid calamari octopus
conch shellfish seafood escargot snail bouillon broth stock consomme worcestershire egg eggnog
mayonnaise mayo rib roast tenderloin wing drumstick thigh breast cutlet loin shank hock burger jerky spam
""".split())
_NON_VEG_PHRASES = ("hot dog", "jell o")
# A word here says the ingredient is a meat-free version, whatever else it says.
_MEAT_FREE = frozenset("vegetarian vegan veggie meatless eggless soy tofu tempeh seitan".split())
# Names that hold one of the words above and are not meat. They are cut out
# of the name before it is checked.
_NOT_MEAT = (
    "vegetable stock", "vegetable broth", "vegetable bouillon", "vegetable consomme", "mushroom broth",
    "mushroom stock", "coconut meat", "nut meat", "celery rib", "rib celery", "rib of celery",
    "swedish fish", "duck sauce", "oyster mushroom", "oyster cracker", "poultry seasoning",
    "steak seasoning", "steak sauce", "sausage seasoning", "meat tenderizer", "burger bun", "hamburger bun",
    "hot dog bun", "hamburger roll", "hot dog roll", "wing sauce", "buffalo sauce", "buffalo mozzarella",
    "kidney bean", "crab apple", "bass ale", "hen of the wood", "chicken of the wood", "goat cheese",
    "goat milk", "goat s milk", "goat s cheese", "beef steak tomato", "imitation bacon", "egg free",
    "egg replacer", "bacon salt",
)

# Tags that say the same. Some recipes leave the meat out of the ingredient list.
_NON_VEG_TAGS = frozenset("""
meat poultry chicken chicken-breasts chicken-thighs-legs whole-chicken chicken-livers wings turkey
turkey-breasts whole-turkey turkey-burgers duck duck-breasts whole-duck goose quail pheasant beef
ground-beef roast-beef steaks steak beef-ribs beef-sausage beef-liver beef-kidney beef-organ-meats veal
pork pork-chops pork-loins pork-loin pork-ribs pork-sausage ham bacon lamb-sheep wild-game deer elk moose
rabbit bear seafood fish saltwater-fish freshwater-fish shellfish salmon tuna cod halibut tilapia trout
bass catfish perch whitefish mahi-mahi orange-roughy sole-and-flounder pickeral shrimp crab lobster
crawfish scallops clams mussels oysters squid octopus eggs omelets-and-frittatas meatloaf meatballs
gelatin main-dish-beef main-dish-pork main-dish-chicken main-dish-seafood
""".split())

_SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to",
          "with", "n"}
_SPACES = re.compile(r"\s+")
_POSSESSIVE = re.compile(r"\b(\w{2,}) s\b")
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?)])")
_SPACE_AFTER = re.compile(r"([(])\s+")
_SENTENCE = re.compile(r"(^|[.!?]\s+)([a-z])")
_LONE_I = re.compile(r"\bi\b")
_DEGREES = re.compile(r"\b(degrees?|deg) ([fc])\b")


def tidy_name(name: str) -> str:
    """'mom s  best chili' -> "Mom's Best Chili"."""
    words = _POSSESSIVE.sub(r"\1's", _SPACES.sub(" ", str(name)).strip()).split()
    out = []
    for i, word in enumerate(words):
        out.append(word if i and word in _SMALL else word[:1].upper() + word[1:])
    return " ".join(out)


def tidy_text(text: str) -> str:
    """Put spaces and capitals back in a lower-cased, tokenised sentence."""
    text = _SPACES.sub(" ", str(text)).strip()
    text = _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))
    text = _LONE_I.sub("I", text)
    text = _DEGREES.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
    return _SENTENCE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def _instructions(steps: list[str]) -> str:
    """Steps as one text, a sentence each, the way the first dataset has them."""
    out = []
    for step in steps:
        step = tidy_text(step).rstrip(" .")
        if step:
            out.append(step)
    return ". ".join(out) + ("." if out else "")


def _display_ingredient(name: str) -> str:
    """Capital first letter, and no commas: the list is stored comma-joined."""
    name = _SPACES.sub(" ", str(name).replace(",", " ")).strip()
    return name[:1].upper() + name[1:]


def is_vegetarian(keys: list[str]) -> bool:
    """True when no ingredient is meat, fish, egg or made from them."""
    for key in keys:
        for phrase in _NOT_MEAT:
            if phrase in key:
                key = key.replace(phrase, " ")
        words = set(key.split())
        if words & _MEAT_FREE:
            continue
        if words & _NON_VEG or any(p in key for p in _NON_VEG_PHRASES):
            return False
    return True


def _literal(cell: str) -> list:
    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _cuisine(tags: list[str], counts: Counter) -> str:
    """The narrowest cuisine tag: the one fewest recipes carry."""
    for table in (_CUISINES, _BROAD_CUISINES):
        found = [t for t in tags if t in table]
        if found:
            return table[min(found, key=lambda t: (counts[t], t))]
    return ""


def _course(tags: set[str]) -> str:
    for tag, label in _COURSES:
        if tag in tags:
            return label
    return ""


def load_foodcom(path: str | Path | None = None) -> pd.DataFrame:
    """One row per Food.com recipe, in the columns of ``load_recipes``."""
    path = Path(path) if path else FOODCOM
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    tag_lists = [[str(t) for t in _literal(r["tags"])] for r in raw]
    counts = Counter(t for tags in tag_lists for t in tags)

    rows = []
    for r, tags in zip(raw, tag_lists):
        name = tidy_name(r["name"])
        shown = list(dict.fromkeys(n for n in map(_display_ingredient, _literal(r["ingredients"])) if n))
        keys = list(dict.fromkeys(k for k in map(canonical, shown) if k))
        if not name or not keys:
            continue
        try:
            minutes = float(r["minutes"])
        except ValueError:
            minutes = 0.0
        tag_set = set(tags)
        veg = is_vegetarian(keys) and not (tag_set & _NON_VEG_TAGS)
        rows.append({
            "name": name,
            "description": tidy_text(r["description"]),
            "cuisine": _cuisine(tags, counts),
            "course": _course(tag_set),
            "diet": "Vegetarian" if veg else "Non Vegetarian",
            "ingredients_name": ",".join(shown),
            "ingredients_quantity": "",
            "prep_time": float("nan"),
            "cook_time": float("nan"),
            "instructions": _instructions([str(s) for s in _literal(r["steps"])]),
            "image_url": "",
            "raw_id": ID_OFFSET + int(r["id"]),
            "ingredients_raw": shown,
            "ingredients": keys,
            "is_hindi": False,
            "veg": veg,
            "total_time": minutes if 0 < minutes <= MAX_MINUTES else float("nan"),
        })
    return pd.DataFrame(rows)
