"""
Shared fixtures. Ported from the FYP's tests/conftest.py.

1. No language model is called: rag.llm.chat is replaced by a spy wherever
   generation could be reached, so tests can assert the model was NOT called.
2. No real embedding model is loaded: the threshold tests use a fake
   collection with fixed distances, and the index tests use a tiny fake
   embedding function. The real MiniLM run is scripts/parity_check.py.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import llm, pipeline  # noqa: E402


class FakeCollection:
    def __init__(self, distances, documents=None, sources=None):
        self.distances = list(distances)
        n = len(self.distances)
        self.documents = list(documents) if documents else [f"chunk {i}" for i in range(n)]
        self.sources = list(sources) if sources else ["doc.txt"] * n
        self.calls = []

    def query(self, query_texts, n_results):
        self.calls.append({"query_texts": query_texts, "n_results": n_results})
        return {
            "documents": [self.documents],
            "metadatas": [[{"source": s} for s in self.sources]],
            "distances": [self.distances],
        }


class ChatSpy:
    """Stands in for rag.llm.chat and records every prompt."""

    def __init__(self, reply="A generated answer about perplexity."):
        self.reply = reply
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return {"answer": self.reply, "prompt_tokens": 10, "completion_tokens": 5,
                "finish_reason": "stop"}

    @property
    def called(self):
        return bool(self.calls)

    @property
    def last_prompt(self):
        return self.calls[-1]


@pytest.fixture
def chat_spy(monkeypatch):
    spy = ChatSpy()
    monkeypatch.setattr(llm, "chat", spy)
    return spy


@pytest.fixture
def collection_at(monkeypatch):
    def _install(distances, documents=None, sources=None):
        fake = FakeCollection(distances, documents, sources)
        monkeypatch.setattr(pipeline, "get_collection", lambda: fake)
        return fake

    return _install


class TinyEmbedding:
    """A deterministic 8-dim bag-of-letters embedding: no download, no torch."""

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 8
            for ch in text.lower():
                v[ord(ch) % 8] += 1.0
            out.append(v)
        return out

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "tiny-test-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return TinyEmbedding()

    def is_legacy(self):
        return False

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]


@pytest.fixture
def tiny_embedding():
    return TinyEmbedding()
