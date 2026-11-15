"""The packed catalogue: every recipe, laid out so the API starts fast.

The first dataset is a 20 MB CSV that pandas reads in a second. With the
Food.com recipes the catalogue is about 240,000 rows and 200 MB of text, which
is too slow to parse and clean on every cold start and too big to hold in
memory as a frame. So it is built once, offline, into ``data/catalogue/``:

- ``rows.npz``      ids, flags, times, label codes and the recipe x ingredient table
- ``names.json``    recipe names and the cuisine and course labels
- ``vocab.json``    canonical ingredients, counts, display names and aliases
- ``tfidf.npz``     the fitted TF-IDF matrix, with its terms in ``tfidf.json``
- ``details-N.bin`` description, steps and ingredient lines, compressed in
                    blocks and read one block at a time (``details.npy`` says where)
- ``text.npy``      text embeddings, reduced to 192 dimensions and stored as int8
                    (``text.npz`` has the projection and the scales)
- ``ingredient_vectors.npz``, ``ranker.txt``, ``ranker.json``  the models,
                    trained on this catalogue

Build it with ``python -m recsys.catalogue`` (needs requirements-dev.txt and the
Food.com file, see recsys.foodcom). The build also writes the ingredient names
the web app suggests. :class:`Catalogue` reads it back.
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .data import ROOT, Vocabulary, load_recipes
from .taxonomy import (COURSE_GROUPS, CUISINE_GROUPS, course_group, course_options_from, cuisine_groups,
                       cuisine_options_from)

# pandas is only needed to read the source data, not to serve the packed catalogue.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

CATALOGUE_DIR = ROOT / "data" / "catalogue"
VERSION = 1
BLOCK = 256                 # recipes per compressed block of details
SHARD_BYTES = 40 << 20      # keep every file well under GitHub's 100 MB limit
TEXT_DIM = 192
RAW_TEXT = ROOT / "data" / "raw" / "text_embeddings_full.npy"  # cache of the 384-d vectors, not shipped
NO_TIME = -1
SUGGESTIONS = ROOT / "web" / "src" / "ingredients.json"  # what the search box offers as you type
MIN_SUGGESTED = 20


def load_sources() -> pd.DataFrame:
    """Both datasets as one frame. Food.com is added when its file is there."""
    import pandas as pd

    from .foodcom import FOODCOM, load_foodcom

    frames = [load_recipes()]
    if FOODCOM.exists():
        frames.append(load_foodcom())
    return pd.concat(frames, ignore_index=True)


def exists(path: Path = CATALOGUE_DIR) -> bool:
    return (Path(path) / "meta.json").is_file()


# ----- writing -----

def _minutes(series: pd.Series) -> np.ndarray:
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(values) & (values >= 0) & (values < 32000)
    return np.where(ok, np.round(np.where(ok, values, 0)), NO_TIME).astype(np.int16)


def _codes(values: pd.Series) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(values))
    index = {label: i for i, label in enumerate(labels)}
    return labels, np.array([index[v] for v in values], dtype=np.uint16)


def _write_details(recipes: pd.DataFrame, out: Path) -> int:
    """Compressed blocks of [description, instructions, ingredient lines]. Returns the number of shards."""
    for old in out.glob("details-*.bin"):
        old.unlink()
    where = []  # (shard, offset, length) per block
    shard, offset, fh = 0, 0, open(out / "details-0.bin", "wb")
    cols = [recipes[c].fillna("").astype(str).tolist() for c in ("description", "instructions", "ingredients_name")]
    for start in range(0, len(recipes), BLOCK):
        block = [list(row) for row in zip(*(c[start:start + BLOCK] for c in cols))]
        data = lzma.compress(json.dumps(block, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), preset=6)
        if offset and offset + len(data) > SHARD_BYTES:
            fh.close()
            shard, offset = shard + 1, 0
            fh = open(out / f"details-{shard}.bin", "wb")
        fh.write(data)
        where.append((shard, offset, len(data)))
        offset += len(data)
    fh.close()
    np.save(out / "details.npy", np.array(where, dtype=np.int64))
    return shard + 1


def build_rows(recipes: pd.DataFrame, out: Path = CATALOGUE_DIR) -> Vocabulary:
    """Everything except the models and the text embeddings."""
    from .baselines import TfidfRanker

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.build(recipes)
    keys = sorted(vocab.counts)
    key_index = {k: i for i, k in enumerate(keys)}

    lists = [[key_index[k] for k in items] for items in recipes["ingredients"]]
    indptr = np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int32)
    indices = np.array([i for idx in lists for i in idx], dtype=np.uint16 if len(keys) < 65536 else np.uint32)
    cuisines, cuisine_code = _codes(recipes["cuisine"])
    courses, course_code = _codes(recipes["course"])
    np.savez_compressed(
        out / "rows.npz",
        ids=recipes["raw_id"].to_numpy(dtype=np.int32),
        veg=recipes["veg"].to_numpy(dtype=bool),
        total=_minutes(recipes["total_time"]), prep=_minutes(recipes["prep_time"]),
        cook=_minutes(recipes["cook_time"]),
        cuisine=cuisine_code, course=course_code, indptr=indptr, indices=indices,
    )
    (out / "names.json").write_text(json.dumps(
        {"names": recipes["name"].tolist(), "cuisines": cuisines, "courses": courses}, ensure_ascii=False))
    (out / "vocab.json").write_text(json.dumps({
        "keys": keys, "counts": [vocab.counts[k] for k in keys], "display": [vocab.display[k] for k in keys],
        "aliases": vocab.alias_to_key,
    }, ensure_ascii=False))

    tfidf = TfidfRanker(recipes)
    terms, idf = tfidf.parts()
    sparse.save_npz(out / "tfidf.npz", tfidf.matrix.astype(np.float32).tocsr())
    (out / "tfidf.json").write_text(json.dumps({"terms": terms, "idf": idf.tolist()}, ensure_ascii=False))

    shards = _write_details(recipes, out)
    (out / "meta.json").write_text(json.dumps({
        "version": VERSION, "rows": len(recipes), "ingredients": len(keys), "block": BLOCK,
        "detail_shards": shards,
    }, indent=2))
    return vocab


def build_text(recipes: pd.DataFrame, out: Path = CATALOGUE_DIR, dim: int = TEXT_DIM,
               workers: int | None = None, raw: Path = RAW_TEXT) -> dict:
    """Embed every recipe, reduce to ``dim`` and store as int8.

    The full 384-d matrix is cached in ``raw`` (not shipped), so changing
    ``dim`` does not embed everything again.
    """
    from .semantic import MODEL_NAME, encode_parallel, fit_projection, project, recipe_texts

    out = Path(out)
    full = None
    if raw.exists():
        cached = np.load(raw)
        if len(cached) == len(recipes):
            full = cached.astype(np.float32)
    if full is None:
        texts = recipe_texts(recipes)
        full = encode_parallel(texts, workers or max(1, (os.cpu_count() or 2) - 2))
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw, full.astype(np.float16))
    components = fit_projection(full, dim)
    small = project(full, components)
    # One scale per dimension: the first directions are much wider than the last.
    scale = 127.0 / np.minimum(np.abs(small).max(axis=0), 5 * small.std(axis=0))
    np.save(out / "text.npy", np.clip(np.round(small * scale), -127, 127).astype(np.int8))
    np.savez(out / "text.npz", components=components, scale=scale.astype(np.float32))
    info = {"model": MODEL_NAME, "rows": int(len(small)), "dim": int(dim)}
    (out / "text.json").write_text(json.dumps(info, indent=2))
    return info


def build_models(recipes: pd.DataFrame, vocab: Vocabulary, out: Path = CATALOGUE_DIR,
                 queries: int = 8000) -> dict:
    """Ingredient vectors and the learned ranker, trained on this catalogue.

    The ranker learns from ``queries`` tune-split recipes (three query styles
    each). Every query scores the whole catalogue, so all of them would take hours.
    """
    from .embeddings import EmbeddingRanker, IngredientIndex, save_vectors, train_vectors
    from .evaluate import build_queries
    from .ranker import N_CANDIDATES, FeatureBuilder, save_ranker, train_ranker

    out = Path(out)
    vectors = train_vectors(recipes)
    save_vectors(vectors, out / "ingredient_vectors.npz")
    embedding = EmbeddingRanker(recipes, vocab, vectors, IngredientIndex(recipes, vocab))
    features = FeatureBuilder(recipes, vocab, embedding, n_candidates=N_CANDIDATES)
    tune = build_queries(recipes, vocab, "tune", sample=queries)
    booster, info = train_ranker(features, [q for qs in tune.values() for q in qs])
    save_ranker(booster, info, N_CANDIDATES, out / "ranker.txt", out / "ranker.json")
    return info


def write_suggestions(vocab: Vocabulary, path: Path = SUGGESTIONS, min_count: int = MIN_SUGGESTED) -> int:
    """Ingredient names for the web app's search box, most used first.

    Only ingredients in at least ``min_count`` recipes: the rest are mostly
    one-off spellings, and anything can still be typed in by hand.
    """
    rows = sorted(((n, vocab.display[k]) for k, n in vocab.counts.items() if n >= min_count),
                  key=lambda r: (-r[0], r[1]))
    names = list(dict.fromkeys(name for _, name in rows))
    Path(path).write_text(json.dumps({"ingredients": names}, indent=0, ensure_ascii=False) + "\n")
    return len(names)


# ----- reading -----

class Catalogue:
    """The packed catalogue, read back. Arrays load at once; recipe text is
    read a block at a time when a recipe page asks for it."""

    def __init__(self, path: str | Path = CATALOGUE_DIR):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        if meta.get("version") != VERSION:
            raise ValueError(f"catalogue version {meta.get('version')} is not {VERSION}")
        self.block = int(meta["block"])
        with np.load(self.path / "rows.npz") as rows:
            self.ids = rows["ids"]
            self.veg = rows["veg"]
            self._total, self._prep, self._cook = rows["total"], rows["prep"], rows["cook"]
            self._cuisine, self._course = rows["cuisine"], rows["course"]
            self.indptr, self.indices = rows["indptr"], rows["indices"]
        self.n = len(self.ids)
        text = json.loads((self.path / "names.json").read_text())
        self.names: list[str] = text["names"]
        self._cuisines: list[str] = text["cuisines"]
        self._courses: list[str] = text["courses"]
        v = json.loads((self.path / "vocab.json").read_text())
        self.keys: list[str] = v["keys"]
        self.vocab = Vocabulary(counts=Counter(dict(zip(v["keys"], v["counts"]))), alias_to_key=v["aliases"],
                                display=dict(zip(v["keys"], v["display"])))
        self._where = np.load(self.path / "details.npy")
        self._parts: dict = {}

    def __len__(self) -> int:
        return self.n

    # ----- one recipe -----

    @staticmethod
    def _time(value) -> int | None:
        return None if value == NO_TIME else int(value)

    def times(self, pos: int) -> tuple[int | None, int | None, int | None]:
        """(prep, cook, total) in minutes; None where the source has none."""
        return self._time(self._prep[pos]), self._time(self._cook[pos]), self._time(self._total[pos])

    def cuisine(self, pos: int) -> str:
        return self._cuisines[self._cuisine[pos]]

    def course(self, pos: int) -> str:
        return self._courses[self._course[pos]]

    def image(self, pos: int) -> str:
        return ""

    def keys_of(self, pos: int) -> list[str]:
        """Canonical ingredients of a recipe, in the order the recipe lists them."""
        return [self.keys[i] for i in self.indices[self.indptr[pos]:self.indptr[pos + 1]]]

    def detail(self, pos: int) -> tuple[str, str, str]:
        """(description, instructions, comma-joined ingredient lines)."""
        shard, offset, length = (int(x) for x in self._where[pos // self.block])
        with open(self.path / f"details-{shard}.bin", "rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
        return tuple(json.loads(lzma.decompress(data))[pos % self.block])

    # ----- filters -----

    def cuisine_masks(self) -> dict[str, np.ndarray]:
        groups = [cuisine_groups(label) for label in self._cuisines]
        return {g: np.array([g in s for s in groups])[self._cuisine] for g in CUISINE_GROUPS}

    def course_masks(self) -> dict[str, np.ndarray]:
        groups = [course_group(label) for label in self._courses]
        return {g: np.array([c == g for c in groups])[self._course] for g in COURSE_GROUPS}

    def options(self) -> dict:
        cuisine = np.bincount(self._cuisine, minlength=len(self._cuisines))
        course = np.bincount(self._course, minlength=len(self._courses))
        return {"cuisines": cuisine_options_from(zip(self._cuisines, cuisine)),
                "courses": course_options_from(zip(self._courses, course))}

    # ----- rankers and search -----

    def _part(self, name: str, make):
        if name not in self._parts:
            self._parts[name] = make()
        return self._parts[name]

    def index(self):
        from .embeddings import IngredientIndex

        return self._part("index", lambda: IngredientIndex(None, self.vocab, rows=(self.indptr, self.indices)))

    def tfidf(self):
        from .baselines import TfidfRanker

        def make():
            fit = json.loads((self.path / "tfidf.json").read_text())
            return TfidfRanker.from_parts(fit["terms"], np.array(fit["idf"], dtype=np.float32),
                                          sparse.load_npz(self.path / "tfidf.npz"))
        return self._part("tfidf", make)

    def overlap(self):
        from .embeddings import OverlapRanker

        return self._part("overlap", lambda: OverlapRanker(None, self.vocab, self.index()))

    def embedding(self):
        from .embeddings import EmbeddingRanker, load_vectors

        return self._part("embedding", lambda: EmbeddingRanker(
            None, self.vocab, load_vectors(self.path / "ingredient_vectors.npz"), self.index()))

    def learned(self):
        from .ranker import FeatureBuilder, LearnedRanker, load_ranker

        def make():
            loaded = load_ranker(self.path / "ranker.txt", self.path / "ranker.json")
            if loaded is None:
                raise FileNotFoundError("no ranker in the catalogue")
            trees, meta = loaded
            features = FeatureBuilder(None, self.vocab, self.embedding(), n_candidates=meta["n_candidates"],
                                      tfidf=self.tfidf(), overlap=self.overlap())
            return LearnedRanker(features, trees, self.n)
        return self._part("learned", make)

    def build_ranker(self, name: str):
        return {"tfidf": self.tfidf, "overlap": self.overlap, "embedding": self.embedding,
                "learned": self.learned}[name]()

    def semantic(self):
        """The text index. Raises when the catalogue was built without it."""
        from .semantic import PackedSemantic

        with np.load(self.path / "text.npz") as p:
            components, scale = p["components"], p["scale"]
        packed = np.load(self.path / "text.npy")
        # In place: a second copy of this matrix is 180 MB the function does not have.
        matrix = np.empty(packed.shape, dtype=np.float32)
        np.divide(packed, scale, out=matrix)
        del packed
        norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
        matrix /= np.maximum(norms, 1e-12)[:, None]
        if len(matrix) != self.n:
            raise ValueError("text embeddings do not match the catalogue")
        return PackedSemantic(matrix, components)

    def name_search(self):
        """TF-IDF over recipe names: the search fallback when the text index is missing."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        def make():
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            return vec, vec.fit_transform(self.names)
        return self._part("names", make)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="*", default=["rows", "text", "models", "web"],
                        help="what to build: rows, text, models, web (default: all four)")
    parser.add_argument("--out", default=str(CATALOGUE_DIR))
    parser.add_argument("--dim", type=int, default=TEXT_DIM, help="size of the stored text vectors")
    parser.add_argument("--queries", type=int, default=8000, help="tune recipes the ranker learns from")
    parser.add_argument("--workers", type=int, help="processes for the text embeddings")
    args = parser.parse_args()
    out = Path(args.out)

    start = time.time()
    recipes = load_sources()
    print(f"{len(recipes)} recipes read in {time.time() - start:.0f}s")
    vocab = None
    if "rows" in args.steps:
        start = time.time()
        vocab = build_rows(recipes, out)
        print(f"rows: {len(vocab.counts)} ingredients, {time.time() - start:.0f}s")
    if "text" in args.steps:
        start = time.time()
        print(f"text: {build_text(recipes, out, args.dim, args.workers)}, {time.time() - start:.0f}s")
    if "models" in args.steps:
        start = time.time()
        info = build_models(recipes, vocab or Vocabulary.build(recipes), out, args.queries)
        print(f"models: {info['best_iteration']} trees, {info['train_queries']} train / {info['valid_queries']} "
              f"held-out queries, candidate recall {info['candidate_recall']:.3f}, held-out {info['valid']}, "
              f"{time.time() - start:.0f}s")
    if "web" in args.steps:
        print(f"web: {write_suggestions(vocab or Vocabulary.build(recipes))} ingredient names in {SUGGESTIONS.name}")
    total = 0
    for path in sorted(out.iterdir()):
        total += path.stat().st_size
        print(f"{path.name:28s} {path.stat().st_size / 1e6:8.1f} MB")
    print(f"{'total':28s} {total / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
