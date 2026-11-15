import pandas as pd

from recsys import foodcom as fc
from recsys.taxonomy import is_mapped


def test_tidy_name_restores_capitals_and_possessives():
    assert fc.tidy_name("mom s  best chili") == "Mom's Best Chili"
    assert fc.tidy_name("cream of the crop  soup") == "Cream of the Crop Soup"
    assert fc.tidy_name("a day at the beach") == "A Day at the Beach"


def test_tidy_text_fixes_spacing_and_sentence_starts():
    got = fc.tidy_text("stir well , then cool .  i bake at 350 degrees f for 10 minutes ( or more )")
    assert got == "Stir well, then cool. I bake at 350 degrees F for 10 minutes (or more)"


def test_vegetarian_rule():
    assert fc.is_vegetarian(["graham cracker", "butter", "eggplant", "vegetable stock", "kidney bean"])
    assert fc.is_vegetarian(["goat cheese", "celery rib", "veggie burger", "soy chorizo"])
    for meat in ("chicken breast", "egg", "beef broth", "worcestershire sauce", "hot dog", "anchovie",
                 "unflavored gelatin", "stock"):
        assert not fc.is_vegetarian(["onion", meat]), meat


def test_every_label_is_in_the_taxonomy():
    cuisines = set(fc._CUISINES.values()) | set(fc._BROAD_CUISINES.values())
    assert all(is_mapped(cuisine=c) for c in cuisines)
    assert all(is_mapped(course=c) for _, c in fc._COURSES)


def test_load_foodcom(tmp_path):
    path = tmp_path / "RAW_recipes.csv"
    pd.DataFrame([
        {"name": "aunt may s  pasta", "id": 42, "minutes": 25,
         "tags": "['italian', 'european', 'main-dish', 'breakfast']",
         "steps": "['boil the pasta', 'add sauce , then serve']", "description": "quick and easy.",
         "ingredients": "['penne', 'tomatoes', 'olive oil']"},
        {"name": "pork chops", "id": 43, "minutes": 0, "tags": "['pork', 'north-american']",
         "steps": "['grill']", "description": "", "ingredients": "['salt', 'pepper']"},
        {"name": "nothing", "id": 44, "minutes": 5, "tags": "[]", "steps": "[]", "description": "",
         "ingredients": "[]"},
    ]).to_csv(path, index=False)
    df = fc.load_foodcom(path)
    assert len(df) == 2  # the recipe with no ingredients is dropped
    first = df.iloc[0]
    assert first["name"] == "Aunt May's Pasta" and first["raw_id"] == fc.ID_OFFSET + 42
    assert first["cuisine"] == "Italian"  # the narrower tag wins over "european"
    assert first["course"] == "Breakfast"
    assert first["instructions"] == "Boil the pasta. Add sauce, then serve."
    assert first["ingredients_name"] == "Penne,Tomatoes,Olive oil"
    assert first["ingredients"] == ["penne", "tomato", "olive oil"]
    assert first["veg"] and first["total_time"] == 25
    second = df.iloc[1]
    assert not second["veg"]  # the tag says pork though no ingredient does
    assert second["cuisine"] == "North American" and pd.isna(second["total_time"])
