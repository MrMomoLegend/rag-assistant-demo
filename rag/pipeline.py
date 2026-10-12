"""
answer_question(): the FYP's main.answer_question with two changes only.
The generation call goes to rag.llm.chat (Cloudflare) instead of ollama.chat,
and the collection is the in-memory index from rag.index instead of a
persistent Chroma store. Gate, prompt, decoding and return shape are the FYP's.

    python -m rag.pipeline "When was GPT-4 released?"
"""
import json
import sys
import time
from typing import Any, Dict, List

from . import llm
from .classify import looks_like_refusal
from .config import DISTANCE_THRESHOLD, REFUSAL, TOP_K

_collection = None


def get_collection():
    """The index, built once per process. Tests and the app replace this."""
    global _collection
    if _collection is None:
        from .index import build_collection

        _collection = build_collection()
    return _collection


def set_collection(coll):
    global _collection
    _collection = coll


def build_prompt(context: str, question: str) -> str:
    # Byte for byte the FYP's prompt (main.py).
    return (
        "Answer the question using ONLY the context below. If the answer "
        f"is not in the context, say '{REFUSAL}'"
        f"\n\nContext:\n{context}\n\nQuestion: {question}"
    )


def answer_question(
    question: str,
    use_rag: bool = True,
    top_k: int = None,
    use_threshold: bool = True,
    threshold: float = None,
) -> Dict[str, Any]:
    t0 = time.time()
    top_k = TOP_K if not top_k else top_k
    threshold = DISTANCE_THRESHOLD if threshold is None else threshold

    retrieved: List[str] = []
    sources: List[str] = []
    distances: List[float] = []
    retrieve_s = 0.0
    base = {"question": question, "use_rag": use_rag, "top_k": top_k}

    if not use_rag:
        prompt = question  # the bare question, no instruction to decline (as in the FYP)
    else:
        tr = time.time()
        results = get_collection().query(query_texts=[question], n_results=top_k)
        retrieve_s = round(time.time() - tr, 3)

        retrieved = results["documents"][0]
        sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
        distances = [round(float(d), 4) for d in results["distances"][0]]

        # The gate: refuse before the model is called if nothing is close enough.
        if use_threshold and (not distances or distances[0] > threshold):
            return {
                **base,
                "answer": REFUSAL,
                "refused": True,
                "refusal_reason": "retrieval_distance",
                "sources": sources,  # kept for display; the FYP returned []
                "retrieved_chunks": retrieved,
                "distances": distances,
                "threshold": threshold,
                "retrieve_s": retrieve_s,
                "generate_s": 0.0,
                "total_s": round(time.time() - t0, 3),
                "prompt_tokens": None,
                "completion_tokens": None,
                "finish_reason": None,
            }

        prompt = build_prompt("\n\n".join(retrieved), question)

    tg = time.time()
    out = llm.chat(prompt)
    answer = out["answer"]
    refused = use_rag and looks_like_refusal(answer)

    return {
        **base,
        "answer": answer,
        "refused": refused,
        "refusal_reason": "model_judgement" if refused else None,
        "sources": sources,
        "retrieved_chunks": retrieved,
        "distances": distances,
        "threshold": threshold if (use_rag and use_threshold) else None,
        "retrieve_s": retrieve_s,
        "generate_s": round(time.time() - tg, 3),
        "total_s": round(time.time() - t0, 3),
        "prompt_tokens": out.get("prompt_tokens"),
        "completion_tokens": out.get("completion_tokens"),
        "finish_reason": out.get("finish_reason"),
    }


def _main(argv):
    import argparse

    ap = argparse.ArgumentParser(description="Ask one question from the command line.")
    ap.add_argument("question")
    ap.add_argument("--no-rag", action="store_true", help="model alone: the bare question")
    args = ap.parse_args(argv)

    from .classify import classify

    r = answer_question(args.question, use_rag=not args.no_rag)
    print(r["answer"])
    print()
    print(json.dumps({
        "outcome": classify(r),
        "refusal_reason": r["refusal_reason"],
        "nearest_distance": r["distances"][0] if r["distances"] else None,
        "top_sources": list(dict.fromkeys(r["sources"]))[:3],
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "finish_reason": r["finish_reason"],
        "retrieve_s": r["retrieve_s"],
        "generate_s": r["generate_s"],
    }, indent=2))


if __name__ == "__main__":
    _main(sys.argv[1:])
