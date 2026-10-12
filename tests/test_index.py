"""
The index: corpus parity with the FYP and the distance metric.
Uses a tiny fake embedding so it runs offline; scripts/parity_check.py does
the real MiniLM check.
"""
import itertools

import chromadb
import pytest

from rag import config
from rag.index import build_collection, load_corpus


def test_corpus_has_23_files_and_1551_chunks():
    pairs = list(load_corpus())
    assert len(pairs) == 23
    assert sum(len(c) for _, c in pairs) == config.EXPECTED_CHUNKS == 1551


def test_corpus_files_are_lf_only():
    for p in config.CORPUS_DIR.iterdir():
        assert b"\r" not in p.read_bytes(), f"{p.name} has CR characters"


def test_corpus_includes_the_readme():
    names = [n for n, _ in load_corpus()]
    assert "README.md" in names
    assert "GPT-4.txt" in names


def test_built_collection_is_cosine_and_complete(tiny_embedding):
    coll = build_collection(embedding_function=tiny_embedding, name="t_full")
    assert (coll.metadata or {}).get("hnsw:space") == "cosine"
    assert coll.count() == 1551


def test_wrong_chunk_count_fails_loudly(tiny_embedding):
    with pytest.raises(RuntimeError, match="expected 9999"):
        build_collection(embedding_function=tiny_embedding, expected=9999, name="t_bad")


# From the FYP's tests/test_distance_metric.py: Chroma defaults to squared L2.
NEAR, FAR = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
_names = itertools.count()


def _distances(metadata):
    client = chromadb.EphemeralClient()
    kw = {"name": f"probe_{next(_names)}"}
    if metadata is not None:
        kw["metadata"] = metadata
    coll = client.create_collection(**kw)
    coll.add(ids=["near", "far"], embeddings=[NEAR, FAR], documents=["n", "f"])
    res = coll.query(query_embeddings=[NEAR], n_results=2)
    return [round(float(d), 4) for d in res["distances"][0]]


def test_cosine_space_puts_orthogonal_vectors_at_one():
    assert _distances({"hnsw:space": "cosine"})[1] == pytest.approx(1.0, abs=1e-3)


def test_chroma_default_is_squared_l2_which_is_why_cosine_is_explicit():
    assert _distances(None)[1] == pytest.approx(2.0, abs=1e-3)
