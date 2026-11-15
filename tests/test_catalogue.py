import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys import catalogue
from recsys.baselines import QueryTerms
from recsys.service import FrameStore, Recommender


@pytest.fixture(scope="module")
def small(recipes):
    return recipes.head(400).reset_index(drop=True)


@pytest.fixture(scope="module")
def packed(small, tmp_path_factory):
    out = tmp_path_factory.mktemp("catalogue")
    catalogue.build_rows(small, out)
    return catalogue.Catalogue(out)


def test_rows_round_trip(small, packed):
    frame = FrameStore(small)
    assert packed.n == len(small) and list(packed.ids) == list(frame.ids)
    assert packed.names == frame.names and list(packed.veg) == list(frame.veg)
    for pos in (0, 7, 255, 256, 399):  # both sides of a block edge
        assert packed.keys_of(pos) == frame.keys_of(pos)
        assert packed.detail(pos) == frame.detail(pos)
        assert packed.times(pos) == frame.times(pos)
        assert packed.cuisine(pos) == frame.cuisine(pos) and packed.course(pos) == frame.course(pos)
    assert packed.options() == frame.options()
    for group, mask in frame.cuisine_masks().items():
        assert (packed.cuisine_masks()[group] == mask).all()
    for group, mask in frame.course_masks().items():
        assert (packed.course_masks()[group] == mask).all()


def test_query_terms_match_sklearn():
    docs = ["red onion tomato", "onion garlic", "Fresh Tomato, basil leaves", "a b"]
    fitted = TfidfVectorizer(binary=True).fit(docs)
    terms = sorted(fitted.vocabulary_, key=fitted.vocabulary_.get)
    mine = QueryTerms(terms, fitted.idf_)
    for query in ("Onion tomato tomato", "basil, garlic & unknown", "nothing here"):
        want = fitted.transform([query]).toarray()
        assert np.allclose(mine.transform([query]).toarray(), want, atol=1e-6)


@pytest.mark.parametrize("model", ["tfidf", "overlap"])
def test_packed_rankers_score_like_the_frame(small, packed, model):
    query = ["onion", "tomato", "curd", "haldi"]
    want = FrameStore(small).build_ranker(model).score(query)
    got = packed.build_ranker(model).score(query)
    assert np.allclose(got, want, atol=1e-5)


def test_recommender_on_a_packed_catalogue(small, packed):
    frame = Recommender(recipes=small)
    rec = Recommender(store=packed)
    for r in (frame, rec):
        r._semantic_failed = True
        r._failed.update({"embedding", "learned"})
    body = dict(ingredients=["onion", "tomato"], cuisine=["Indian"], veg=True, model="tfidf", limit=12)
    want = frame.recommend(**body)
    got = rec.recommend(**body)
    assert [c["id"] for c in got] == [c["id"] for c in want]
    assert {k: v for k, v in got[0].items() if k != "image"} == {k: v for k, v in want[0].items() if k != "image"}
    rid = want[0]["id"]
    a, b = rec.recipe(rid), frame.recipe(rid)
    assert {k: v for k, v in a.items() if k != "image"} == {k: v for k, v in b.items() if k != "image"}
    assert rec.meta()["recipes"] == len(small)
    assert rec.search("spinach curry", limit=3)  # falls back to TF-IDF over names
    assert rec.similar(rid, limit=4)


def test_pick_matches_a_full_sort(small, monkeypatch):
    rec = Recommender(recipes=small)
    scores = np.random.default_rng(3).random(len(small))
    allowed = np.ones(len(small), dtype=bool)
    full = rec._pick(scores, allowed, 30)
    monkeypatch.setattr("recsys.service.PICK_POOL", 5)
    assert rec._pick(scores, allowed, 30) == full
