"""Ingredient embeddings and the rankers that work on canonical ingredients.

Word2Vec skip-gram is trained with each recipe's ingredient list as one
sentence. Order inside a recipe means nothing, so the window covers the whole
list and every pair of ingredients in a recipe counts as context.

A recipe vector is the IDF-weighted mean of its ingredient vectors. A query is
turned into a vector the same way, after resolving what the person typed to
canonical ingredients. Strings the vocabulary does not know go through a
character n-gram matcher instead of being dropped.
"""
from __future__ import annotations

import difflib
import json
import zlib
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Vocabulary, canonical

# pandas is only needed to read the source data, not to serve the packed catalogue.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORS_PATH = MODELS_DIR / "ingredient_vectors.npz"

SEED = 7
# Chosen on the tune split. Bigger vectors (128, 256) and more epochs help the
# embedding ranker on its own but not the learned ranker, and they give worse
# neighbours, so the small setting wins. min_count=1 so every ingredient a
# person can type has a vector.
W2V_PARAMS = {"vector_size": 64, "epochs": 40, "min_count": 1, "negative": 10, "sample": 0.001}
FUZZY_THRESHOLD = 0.7
SPELLING_CUTOFF = 0.8  # difflib ratio: one slip in a five-letter word
MIN_COMMON = 20
OVERLAP_WEIGHT = 0.4


def _stable_hash(text: str) -> int:
    # gensim seeds each word vector with hash(word + seed). The built-in hash
    # changes between runs, so use one that does not.
    return zlib.crc32(text.encode("utf-8"))


class IngredientVectors:
    """A plain key -> vector table, so the app does not need gensim at runtime."""

    def __init__(self, keys: list[str], vectors: np.ndarray, params: dict | None = None):
        self.keys = list(keys)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.params = dict(params or {})
        self.index = {k: i for i, k in enumerate(self.keys)}

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def __len__(self) -> int:
        return len(self.keys)

    def get(self, key: str) -> np.ndarray | None:
        i = self.index.get(key)
        return None if i is None else self.vectors[i]


def train_vectors(recipes: pd.DataFrame, seed: int = SEED, **params) -> IngredientVectors:
    """Train skip-gram vectors on the catalogue's ingredient lists."""
    from gensim.models import Word2Vec

    params = {**W2V_PARAMS, **params}
    sentences = [list(items) for items in recipes["ingredients"]]
    window = max(len(s) for s in sentences)
    model = Word2Vec(
        sentences,
        sg=1,
        window=window,
        shrink_windows=False,  # every ingredient in the recipe is equally close
        workers=1,
        seed=seed,
        hashfxn=_stable_hash,
        **params,
    )
    keys = list(model.wv.index_to_key)
    return IngredientVectors(keys, model.wv.vectors.copy(), {**params, "seed": seed, "window": window})


def save_vectors(vectors: IngredientVectors, path: str | Path = VECTORS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # float16 halves the file and changes cosines in the fourth decimal.
    np.savez_compressed(path, keys=np.array(vectors.keys), vectors=vectors.vectors.astype(np.float16),
                        params=np.array(json.dumps(vectors.params)))
    return path


def load_vectors(path: str | Path = VECTORS_PATH) -> IngredientVectors:
    with np.load(path) as data:
        return IngredientVectors(list(data["keys"]), data["vectors"].astype(np.float32),
                                 json.loads(str(data["params"])))


class FuzzyMatcher:
    """Nearest canonical ingredient or alias for a string the vocabulary misses.

    Two passes. Spelling first: the closest common ingredient by matching
    characters (difflib), which is what catches "chiken", "tomatoe" and
    "brocolli". Then character n-gram TF-IDF over every name and alias, which
    catches word-level slips: "curry leave", "green chilly". Below the
    thresholds the string is left unresolved.

    The spelling pass only looks at ingredients in at least ``MIN_COMMON``
    recipes. A big catalogue holds other people's typos too, and "tomatoe"
    should become "tomato", not the rare "dried tomatoe".
    """

    def __init__(self, vocab: Vocabulary, threshold: float = FUZZY_THRESHOLD):
        names = list(vocab.counts) + list(vocab.alias_to_key)
        self.targets = list(vocab.counts) + list(vocab.alias_to_key.values())
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(names)
        self._counts = vocab.counts
        self._common = [k for k, n in vocab.counts.items() if n >= MIN_COMMON]
        self._cache: dict[str, tuple[str, float] | None] = {}

    def _spelling(self, key: str) -> tuple[str, float] | None:
        close = difflib.get_close_matches(key, self._common, n=5, cutoff=SPELLING_CUTOFF)
        if not close:
            return None
        ratio = {c: difflib.SequenceMatcher(None, key, c).ratio() for c in close}
        best = max(close, key=lambda c: (round(ratio[c], 2), self._counts[c]))
        return best, ratio[best]

    def nearest(self, text: str) -> tuple[str, float] | None:
        key = canonical(text)
        if not key:
            return None
        if key not in self._cache:
            hit = self._spelling(key)
            if hit is None:
                sims = (self.matrix @ self.vectorizer.transform([key]).T).toarray().ravel()
                best = int(sims.argmax())
                hit = (self.targets[best], float(sims[best])) if sims[best] > 0 else None
            self._cache[key] = hit
        return self._cache[key]

    def match(self, text: str) -> str | None:
        hit = self.nearest(text)
        return hit[0] if hit and hit[1] >= self.threshold else None


class IngredientIndex:
    """Recipe x ingredient incidence, IDF and query resolution, shared by the
    rankers below and by the learned ranker's features."""

    def __init__(self, recipes: pd.DataFrame | None, vocab: Vocabulary, fuzzy_threshold: float = FUZZY_THRESHOLD,
                 rows: tuple[np.ndarray, np.ndarray] | None = None):
        """``rows`` is a ready table ``(indptr, indices)`` in CSR layout, one row per
        recipe, with ingredient ids that follow ``sorted(vocab.counts)``. With it
        ``recipes`` is not read, which is how the packed catalogue starts fast."""
        self.vocab = vocab
        self.keys = sorted(vocab.counts)
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        if rows is None:
            lists = [[self.key_index[k] for k in items if k in self.key_index] for items in recipes["ingredients"]]
            lens = np.array([len(x) for x in lists], dtype=np.int64)
            indptr = np.concatenate([[0], np.cumsum(lens)])
            indices = np.array([i for idx in lists for i in idx], dtype=np.int32)
        else:
            indptr = np.asarray(rows[0], dtype=np.int64)
            indices = np.asarray(rows[1], dtype=np.int32)
            lens = np.diff(indptr)
        n = len(lens)
        counts = np.array([vocab.counts[k] for k in self.keys], dtype=np.float64)
        self.idf = np.log(np.maximum(n, counts) / counts).astype(np.float32)

        data = np.ones(len(indices), dtype=np.float32)
        # Copies, so sorting the matrix leaves the caller's recipe order alone.
        self.incidence = sparse.csr_matrix((data, indices.copy(), indptr.copy()), shape=(n, len(self.keys)))
        self.incidence.sum_duplicates()
        self.incidence_csc = self.incidence.tocsc()
        self.recipe_len = lens.astype(np.float32)
        self.recipe_idf = np.asarray(self.incidence @ self.idf).ravel().astype(np.float32)
        # Padded ingredient ids per recipe, for max-over-ingredients lookups.
        # The pad points at an extra column that callers fill with a low value.
        width = max(1, int(lens.max())) if n else 1
        small = np.uint16 if len(self.keys) < np.iinfo(np.uint16).max else np.int32
        self.padded = np.full((n, width), len(self.keys), dtype=small)
        row_of = np.repeat(np.arange(n), lens)
        self.padded[row_of, np.arange(len(indices)) - indptr[:-1][row_of]] = indices
        self.fuzzy = FuzzyMatcher(vocab, fuzzy_threshold)

    def resolve(self, query: list[str]) -> tuple[list[int], int]:
        """Column ids for the typed strings, plus how many could not be placed."""
        out: list[int] = []
        missing = 0
        for item in query:
            key = self.vocab.resolve(item) or self.fuzzy.match(item)
            i = self.key_index.get(key) if key else None
            if i is None:
                missing += 1
            elif i not in out:
                out.append(i)
        return out, missing

    def matched(self, cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Per recipe: how many query ingredients it has, and their IDF sum."""
        if not cols:
            zero = np.zeros(self.incidence.shape[0], dtype=np.float32)
            return zero, zero.copy()
        sub = self.incidence_csc[:, cols]
        count = np.asarray(sub.sum(axis=1)).ravel().astype(np.float32)
        idf = np.asarray(sub @ self.idf[cols]).ravel().astype(np.float32)
        return count, idf


class OverlapRanker:
    """Lexical, but on canonical ingredients: share of the query's IDF mass the
    recipe covers, with a small bonus for covering more of the recipe so that
    a short exact match beats a long recipe that happens to contain it all."""

    name = "overlap"

    def __init__(self, recipes: pd.DataFrame, vocab: Vocabulary, index: IngredientIndex | None = None,
                 recipe_weight: float = OVERLAP_WEIGHT):
        self.index = index or IngredientIndex(recipes, vocab)
        self.recipe_weight = recipe_weight

    def score_matched(self, cols: list[int], matched_idf: np.ndarray) -> np.ndarray:
        if not cols:
            return np.zeros(self.index.incidence.shape[0], dtype=np.float32)
        total = float(self.index.idf[cols].sum()) or 1.0
        recipe = matched_idf / np.maximum(self.index.recipe_idf, 1e-6)
        return (matched_idf / total + self.recipe_weight * recipe).astype(np.float32)

    def score(self, query: list[str]) -> np.ndarray:
        cols, _ = self.index.resolve(query)
        _, idf = self.index.matched(cols)
        return self.score_matched(cols, idf)


class EmbeddingRanker:
    """Cosine between the query vector and IDF-weighted recipe vectors."""

    name = "embedding"

    def __init__(self, recipes: pd.DataFrame, vocab: Vocabulary, vectors: IngredientVectors | None = None,
                 index: IngredientIndex | None = None, **train_params):
        self.index = index or IngredientIndex(recipes, vocab)
        self.vectors = vectors or train_vectors(recipes, **train_params)
        dim = self.vectors.vectors.shape[1]
        table = np.zeros((len(self.index.keys), dim), dtype=np.float32)
        for i, key in enumerate(self.index.keys):
            vec = self.vectors.get(key)
            if vec is not None:
                table[i] = vec
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        self.has_vector = norms.ravel() > 0
        self.table = table / np.maximum(norms, 1e-12)
        weighted = self.index.incidence.multiply(self.index.idf[None, :]).tocsr()
        self.recipe_vectors = _l2(np.asarray(weighted @ self.table, dtype=np.float32))

    def query_vector(self, cols: list[int]) -> np.ndarray:
        if not cols:
            return np.zeros(self.table.shape[1], dtype=np.float32)
        vec = self.index.idf[cols] @ self.table[cols]
        return _l2(vec[None, :])[0]

    def score_cols(self, cols: list[int]) -> np.ndarray:
        return self.recipe_vectors @ self.query_vector(cols)

    def score(self, query: list[str]) -> np.ndarray:
        cols, _ = self.index.resolve(query)
        return self.score_cols(cols)

    def most_similar_ingredients(self, name: str, k: int = 10, min_count: int = 5) -> list[tuple[str, float]]:
        """Nearest ingredients to ``name`` by cosine, as display names.

        Ingredients seen in fewer than ``min_count`` recipes are skipped. A
        one-off ingredient's vector is mostly the recipe it came from, so it
        sits close to everything else in that recipe and crowds the list.
        """
        cols, _ = self.index.resolve([name])
        if not cols or not self.has_vector[cols[0]]:
            return []
        sims = self.table @ self.table[cols[0]]
        sims[cols[0]] = -np.inf
        sims[~self.has_vector] = -np.inf
        counts = self.index.vocab.counts
        sims[[counts[key] < min_count for key in self.index.keys]] = -np.inf
        top = np.argsort(-sims)[:k]
        display = self.index.vocab.display
        return [(display.get(self.index.keys[i], self.index.keys[i]), float(sims[i])) for i in top]


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, 1e-12)).astype(np.float32)
