"""
Sentence-aware chunking, copied from the FYP (main.chunk_sentence and its
fallback). LangChain's RecursiveCharacterTextSplitter is the implementation
the FYP shipped and measured; the pure-Python fallback follows the same
separator hierarchy but adds no overlap, so it is only a safety net.
"""
from typing import List

from .config import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS


def chunk_sentence(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    return _recursive_split(text, ["\n\n", "\n", ". ", " "])


def _split_keeping_punctuation(text: str, sep: str) -> List[str]:
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    tail = sep.rstrip()
    return [p + tail if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, seps: List[str]) -> List[str]:
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    if not seps:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    sep, rest = seps[0], seps[1:]
    pieces = [p for p in _split_keeping_punctuation(text, sep) if p.strip()]
    if len(pieces) <= 1:
        return _recursive_split(text, rest)

    chunks, buf = [], ""
    for piece in pieces:
        candidate = f"{buf} {piece}".strip() if buf else piece
        if len(candidate) <= CHUNK_SIZE:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(piece) > CHUNK_SIZE:
            chunks.extend(_recursive_split(piece, rest))
        else:
            buf = piece
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]
