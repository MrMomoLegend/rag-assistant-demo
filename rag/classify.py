"""
Outcome classification, copied from the FYP (main.looks_like_refusal and
eval/common.classify). The UI badge uses classify(), which separates a
hedge (refusal sentence followed by an answer anyway) from a refusal. The
FYP's own UI used the keyword check and tagged hedges "Declined"; this fixes
that.
"""

REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "doesn't contain",
    "not in the context",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not provided in the context",
    "insufficient",
    "i don't know",
    "i do not know",
)

# main.looks_like_refusal's list (no "i don't know" variants).
_MAIN_MARKERS = REFUSAL_MARKERS[:10]

CONTINUATIONS = ("however", "but ", "although", "according to", "based on",
                 "that said", "nevertheless", "it appears", "it is mentioned",
                 "it can be inferred", "from the context")


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in _MAIN_MARKERS)


def classify(result: dict) -> str:
    """Return 'refused', 'hedged' or 'answered'."""
    if result.get("refusal_reason") == "retrieval_distance":
        return "refused"

    answer = (result.get("answer") or "").strip()
    if not answer:
        return "refused"

    low = answer.lower()
    marker = next((m for m in REFUSAL_MARKERS if m in low), None)
    if marker is None:
        return "answered"

    tail = answer[low.index(marker):]
    rest = tail.split(".", 1)[1].strip() if "." in tail else ""
    if len(rest) > 100 or any(c in rest.lower()[:80] for c in CONTINUATIONS):
        return "hedged"
    return "refused"
