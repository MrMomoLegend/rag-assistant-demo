"""
Build the in-memory vector index from corpus/ at startup.

Same embedding call as the FYP (Chroma's SentenceTransformerEmbeddingFunction
with all-MiniLM-L6-v2, main.py l.113 / ingest.py l.57) and the same metric,
set explicitly: Chroma defaults to squared L2, and the 0.53 threshold is a
cosine distance. Nothing is persisted; rebuilding takes well under a minute
on CPU and the hosted app's disk isn't persistent anyway.
"""
import hashlib
from pathlib import Path
from typing import Optional

import chromadb

from .chunking import chunk_sentence
from .config import COLLECTION_NAME, CORPUS_DIR, EMBED_MODEL, EXPECTED_CHUNKS

SUPPORTED = {".txt", ".md"}


def load_corpus(corpus_dir: Path = CORPUS_DIR):
    """Yield (source_name, chunks) for every file, in sorted order (as ingest.py)."""
    files = sorted(p for p in Path(corpus_dir).iterdir() if p.suffix.lower() in SUPPORTED)
    for p in files:
        # Text mode, as the FYP's extract_text: any CRLF from a Windows checkout
        # is read back as LF, so the split points (and the 1,551 count) hold.
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        yield p.name, chunk_sentence(text)


def default_embedding_function():
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def build_collection(corpus_dir: Path = CORPUS_DIR, embedding_function=None,
                     expected: Optional[int] = EXPECTED_CHUNKS, name: str = COLLECTION_NAME):
    """Return a fresh in-memory Chroma collection holding the whole corpus."""
    ef = embedding_function or default_embedding_function()
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(name)  # EphemeralClient is shared per process
    except Exception:
        pass
    coll = client.create_collection(
        name=name, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    total = 0
    for source, chunks in load_corpus(corpus_dir):
        if not chunks:
            continue
        prefix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        coll.add(
            documents=chunks,
            metadatas=[{"source": source} for _ in chunks],
            ids=[f"{prefix}-sentence-{i}" for i in range(len(chunks))],
        )
        total += len(chunks)
    if expected is not None and total != expected:
        raise RuntimeError(
            f"Index holds {total} chunks, expected {expected}. Check corpus/ line endings "
            "(must be LF) and the langchain-text-splitters version."
        )
    return coll
