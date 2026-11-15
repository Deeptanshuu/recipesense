"""The recommender the API talks to.

Loads the catalogue once, keeps every lookup in memory and builds rankers on
first use. The catalogue is the packed one in ``data/catalogue/`` when it is
there (see recsys.catalogue), else the first dataset read from its CSV. Learned, embedding and semantic parts are optional at run time: each
is built inside try/except and the service falls back to TF-IDF when one fails.
Brute force is not offered here; it is only an offline evaluation baseline.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import catalogue
from .data import Vocabulary, load_recipes, split_ingredients
from .palette import Palette
from .registry import build_rankers
from .taxonomy import (COURSE_GROUPS, CUISINE_GROUPS, course_group, course_options, cuisine_groups,
                       cuisine_options, resolve_course, resolve_cuisines)

# _pick sorts only this many of the best rows unless it runs out of them.
PICK_POOL = 2000

# Boosts are added to a ranker score that is scaled to 0..1, so a cuisine or
# course match is worth 20 or 30 percent of a perfect ingredient match.
CUISINE_BOOST = 0.2
COURSE_BOOST = 0.3
TASTE_BOOST = 0.3

# Models offered, in order. The first is the default. All are built by
# recsys.registry.build_rankers.
MODELS = [
    {"id": "learned", "label": "Learned ranker"},
    {"id": "embedding", "label": "Ingredient embeddings"},
    {"id": "overlap", "label": "Ingredient overlap"},
    {"id": "tfidf", "label": "TF-IDF"},
]
MODEL_IDS = [m["id"] for m in MODELS]
DEFAULT_MODEL = "learned"
FALLBACK_MODEL = "tfidf"


class RecipeNotFound(LookupError):
    pass


def _minutes(value) -> int | None:
    """Whole minutes, or None when the value is missing."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else int(round(value))


def _scale(scores: np.ndarray) -> np.ndarray:
    """Scale to 0..1 with the best score at 1.

    The learned ranker's raw scores can be negative, so those are shifted to
    start at 0 first. Rankers that never go below 0 are only divided by the
    best score, which keeps 0 meaning "no match".
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not scores.size:
        return scores
    low = min(float(scores.min()), 0.0)
    span = float(scores.max()) - low
    return (scores - low) / span if span > 0 else np.zeros_like(scores)


class FrameStore:
    """A catalogue held as a pandas frame: the first dataset, or the rows a
    test passes in. Same surface as :class:`recsys.catalogue.Catalogue`."""

    def __init__(self, recipes: pd.DataFrame):
        self.recipes = df = recipes.reset_index(drop=True)
        self.vocab = Vocabulary.build(df)
        self.n = len(df)
        self.ids = df["raw_id"].to_numpy()
        self.veg = df["veg"].to_numpy(dtype=bool)
        self.names = [str(n) for n in df["name"]]

    def times(self, pos: int) -> tuple[int | None, int | None, int | None]:
        row = self.recipes.iloc[pos]
        return _minutes(row["prep_time"]), _minutes(row["cook_time"]), _minutes(row["total_time"])

    def cuisine(self, pos: int) -> str:
        return self.recipes["cuisine"].iat[pos]

    def course(self, pos: int) -> str:
        return self.recipes["course"].iat[pos]

    def image(self, pos: int) -> str:
        return self.recipes["image_url"].iat[pos]

    def keys_of(self, pos: int) -> list[str]:
        return self.recipes["ingredients"].iat[pos]

    def detail(self, pos: int) -> tuple[str, str, str]:
        row = self.recipes.iloc[pos]
        names = str(row["ingredients_name"]) if pd.notna(row["ingredients_name"]) else ""
        return row["description"], row["instructions"], names

    def cuisine_masks(self) -> dict[str, np.ndarray]:
        groups = [cuisine_groups(c) for c in self.recipes["cuisine"]]
        return {g: np.array([g in s for s in groups]) for g in CUISINE_GROUPS}

    def course_masks(self) -> dict[str, np.ndarray]:
        courses = [course_group(c) for c in self.recipes["course"]]
        return {g: np.array([c == g for c in courses]) for g in COURSE_GROUPS}

    def options(self) -> dict:
        return {"cuisines": cuisine_options(self.recipes), "courses": course_options(self.recipes)}

    def build_ranker(self, name: str):
        return build_rankers(self.recipes, self.vocab, only=[name])[name]

    def semantic(self):
        from .semantic import SemanticSearch

        return SemanticSearch(self.recipes)

    def name_search(self):
        """TF-IDF over name and description. The search fallback."""
        docs = [f"{n} {n} {d}" for n, d in zip(self.recipes["name"], self.recipes["description"])]
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        return vec, vec.fit_transform(docs)


class Recommender:
    def __init__(self, recipes: pd.DataFrame | None = None, path=None, store=None):
        if store is not None:
            self.store = store
        elif recipes is not None:
            self.store = FrameStore(recipes)
        elif path is None and catalogue.exists():
            self.store = catalogue.Catalogue()
        else:
            self.store = FrameStore(load_recipes(path))
        store = self.store
        self.recipes = getattr(store, "recipes", None)
        self.vocab = store.vocab
        self._pos = {int(i): p for p, i in enumerate(store.ids)}
        self._veg = np.asarray(store.veg, dtype=bool)
        self._palette = Palette(self.vocab, store.n)
        self._names = [n.strip().lower() for n in store.names]
        self._cuisine_mask = store.cuisine_masks()
        self._course_mask = store.course_masks()
        self._lock = threading.RLock()
        self._rankers: dict = {}
        self._failed: set[str] = set()
        self._text_search = None
        self._semantic = None
        self._semantic_failed = False
        self._combined = None  # (key, matrix) for "more like this" and taste
        self._ready = {"ranker": False, "search": False}
        self._meta = None

    # ----- rows -----

    def _card(self, pos: int) -> dict:
        store = self.store
        return {
            "id": int(store.ids[pos]),
            "title": store.names[pos],
            "total_time": store.times(pos)[2],
            "image": store.image(pos),
            "veg": bool(self._veg[pos]),
            "cuisine": store.cuisine(pos),
            "course": store.course(pos),
            "palette": self._palette.for_recipe(store.keys_of(pos)),
        }

    def _cards(self, positions, scores) -> list[dict]:
        return [{**self._card(int(p)), "score": round(float(s), 4)} for p, s in zip(positions, scores)]

    def _position(self, recipe_id: int) -> int:
        try:
            return self._pos[int(recipe_id)]
        except (KeyError, TypeError, ValueError):
            raise RecipeNotFound(f"No recipe with id {recipe_id}") from None

    def _pick(self, scores: np.ndarray, allowed: np.ndarray, limit: int, skip=()) -> tuple[list[int], list[float]]:
        """Best positions by score. Drops zeros, repeated titles and ``skip``."""
        cand = np.flatnonzero(allowed & (scores > 0))
        pool = max(PICK_POOL, 4 * limit)
        while True:
            # A full sort of a big catalogue is slow, and the answer is near the top.
            if len(cand) > pool:
                kth = len(cand) - pool
                bound = np.partition(scores[cand], kth)[kth]
                part = cand[scores[cand] >= bound]
            else:
                part = cand
            order = part[np.lexsort((part, -scores[part]))]
            seen, pos, out = set(skip), [], []
            for p in order:
                name = self._names[p]
                if name in seen:
                    continue
                seen.add(name)
                pos.append(int(p))
                out.append(float(scores[p]))
                if len(pos) >= limit:
                    break
            if len(pos) >= limit or len(part) == len(cand):
                return pos, out
            pool *= 8

    # ----- models -----

    def models(self) -> list[dict]:
        """Models the API can offer right now."""
        return [m for m in MODELS if m["id"] not in self._failed]

    def _ranker(self, name: str):
        with self._lock:
            if name in self._rankers:
                return self._rankers[name]
            if name in self._failed:
                return None
            try:
                self._rankers[name] = self.store.build_ranker(name)
            except Exception:
                if name == FALLBACK_MODEL:
                    raise
                self._failed.add(name)
                return None
            return self._rankers[name]

    def resolve_model(self, name: str) -> str:
        """The model that will really run. Falls back to tfidf when the asked one
        cannot be built."""
        if name not in MODEL_IDS:
            raise ValueError(f"Unknown model {name!r}")
        return name if self._ranker(name) is not None else FALLBACK_MODEL

    # ----- warm-up -----

    def warm(self) -> None:
        """Build the default ranker and the semantic index now, not on the first request."""
        try:
            self.resolve_model(DEFAULT_MODEL)
        finally:
            self._ready["ranker"] = True
        try:
            index = self._semantic_index()
            if index is not None:
                index.search("warm up", 1)  # loads the sentence model
            self._vectors()
        finally:
            self._ready["search"] = True

    def ready(self) -> dict:
        """Set by warm(); a request that builds the parts itself does not flip it."""
        with self._lock:
            return {
                "ranker": self._ready["ranker"],
                "search": self._ready["search"],
            }

    # ----- masks and boosts -----

    def _allowed(self, veg: bool) -> np.ndarray:
        return self._veg.copy() if veg else np.ones(len(self._veg), dtype=bool)

    def _group_boost(self, cuisine, course) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._veg)
        cmask = np.zeros(n, dtype=bool)
        for g in resolve_cuisines(cuisine or []):
            cmask |= self._cuisine_mask[g]
        group = resolve_course(course or "")
        kmask = self._course_mask[group] if group else np.zeros(n, dtype=bool)
        return cmask, kmask

    def _taste(self, liked_pos: list[int]) -> np.ndarray:
        """How close each recipe is to what the person liked, 0..1."""
        matrix = self._vectors()
        if matrix is not None:
            try:
                from .similar import TasteProfile

                profile = TasteProfile(matrix)
                for p in liked_pos:
                    profile.like(p)
                return _scale(np.clip(profile.similarity(), 0, None))
            except Exception:
                pass
        # Fallback: mean TF-IDF vector of the liked recipes.
        ranker = self._ranker(FALLBACK_MODEL)
        centroid = np.asarray(ranker.matrix[liked_pos].mean(axis=0))
        sims = np.asarray(ranker.matrix @ centroid.T).ravel()
        return _scale(sims)

    # ----- public API -----

    def recommend(self, ingredients, cuisine=None, course=None, veg=False, model=DEFAULT_MODEL, liked=None,
                  limit=24) -> list[dict]:
        ingredients = [i for i in (ingredients or []) if str(i).strip()]
        liked_pos = [self._pos[int(i)] for i in (liked or []) if int(i) in self._pos]
        cmask, kmask = self._group_boost(cuisine, course)
        if not (ingredients or cmask.any() or kmask.any() or liked_pos):
            return []

        scores = np.zeros(len(self._veg))
        if ingredients:
            used = self.resolve_model(model)
            scores += _scale(self._ranker(used).score(ingredients))
        scores += CUISINE_BOOST * cmask + COURSE_BOOST * kmask
        if liked_pos:
            scores += TASTE_BOOST * self._taste(liked_pos)

        allowed = self._allowed(veg)
        allowed[liked_pos] = False
        pos, vals = self._pick(scores, allowed, limit)
        return self._cards(pos, vals)

    def search(self, text: str, veg: bool = False, limit: int = 24, cuisine=None, course=None) -> list[dict]:
        allowed = self._allowed(veg)
        cmask, kmask = self._group_boost(cuisine, course)
        boosted = cmask.any() or kmask.any()
        # With a boost, look at more candidates so a matching recipe a little
        # further down the text ranking can still come up.
        pool = max(limit * 3, 300) if boosted else limit * 3
        found = self._semantic_search(text, pool, allowed)
        if found is None:
            scores = self._text_scores(text)
        else:
            scores = np.zeros(len(allowed))
            scores[found[0]] = found[1]
        if boosted:
            base = _scale(np.where(allowed, scores, 0.0))
            scores = base + (CUISINE_BOOST * cmask + COURSE_BOOST * kmask) * (base > 0)
        pos, vals = self._pick(scores, allowed, limit)
        return self._cards(pos, vals)

    def recipe(self, recipe_id: int) -> dict:
        return self._detail(self._position(recipe_id))

    def similar(self, recipe_id: int, limit: int = 12) -> list[dict]:
        pos = self._position(recipe_id)
        allowed = np.ones(len(self._veg), dtype=bool)
        allowed[pos] = False
        found = self._similar_found(pos, limit * 3, allowed)
        if found is None:
            ranker = self._ranker(FALLBACK_MODEL)
            scores = np.asarray((ranker.matrix @ ranker.matrix[pos].T).todense()).ravel()
        else:
            scores = np.zeros(len(allowed))
            scores[found[0]] = found[1]
        picked, vals = self._pick(scores, allowed, limit, skip=[self._names[pos]])
        return self._cards(picked, vals)

    def meta(self) -> dict:
        if self._meta is None:
            self._meta = {**self.store.options(), "recipes": self.store.n}
        return {**self._meta, "models": self.models()}

    # ----- detail -----

    def _detail(self, pos: int) -> dict:
        store = self.store
        card = self._card(pos)
        description, instructions, names = store.detail(pos)
        prep, cook, total = store.times(pos)
        return {
            "id": card["id"],
            "name": card["title"],
            "description": description,
            "veg": card["veg"],
            "image": card["image"],
            "prep_time": prep,
            "cook_time": cook,
            "total_time": total,
            "course": card["course"],
            "cuisine": card["cuisine"],
            "cooking_ingredients": names,
            "cooking_instruction": instructions,
            "palette": card["palette"],
            # One entry per ingredient in cooking_ingredients, in the same order.
            "ingredient_colors": self._palette.colors(split_ingredients(names)),
        }

    # ----- semantic parts -----

    def _semantic_index(self):
        """The SemanticSearch index, built once. None if it cannot be built."""
        with self._lock:
            if self._semantic is None and not self._semantic_failed:
                try:
                    self._semantic = self.store.semantic()
                except Exception:
                    self._semantic_failed = True
            return self._semantic

    def _semantic_search(self, text: str, k: int, mask: np.ndarray):
        """(positions, scores) from the semantic index, or None to use TF-IDF."""
        index = self._semantic_index()
        if index is None:
            return None
        try:
            idx, sc = index.search(text, k, mask)
            return np.asarray(idx, dtype=int), np.asarray(sc, dtype=float)
        except Exception:
            return None

    def _vectors(self):
        """L2-normalised recipe matrix: ingredient vectors joined with text
        vectors when both exist, else whichever one does. None if neither."""
        ranker = self._ranker("embedding")
        ingredient = getattr(ranker, "recipe_vectors", None)
        index = self._semantic_index()
        text = getattr(index, "matrix", None)
        key = (id(ingredient), id(text))
        with self._lock:
            if self._combined is not None and self._combined[0] == key:
                return self._combined[1]
            try:
                if ingredient is not None and text is not None:
                    from .similar import combine

                    matrix = combine([ingredient, text])
                else:
                    matrix = ingredient if ingredient is not None else text
            except Exception:
                matrix = ingredient
            self._combined = (key, matrix)
            return matrix

    def _similar_found(self, pos: int, k: int, mask: np.ndarray):
        """(positions, scores) from the combined vectors, or None to use TF-IDF."""
        matrix = self._vectors()
        if matrix is None:
            return None
        try:
            from .similar import similar_recipes

            idx, sc = similar_recipes(matrix, pos, k, mask)
            return np.asarray(idx, dtype=int), np.asarray(sc, dtype=float)
        except Exception:
            return None

    def _text_scores(self, text: str) -> np.ndarray:
        """TF-IDF cosine over the recipe text. The search fallback."""
        with self._lock:
            if self._text_search is None:
                self._text_search = self.store.name_search()
            vec, matrix = self._text_search
        q = vec.transform([text])
        return np.asarray((matrix @ q.T).todense()).ravel()
