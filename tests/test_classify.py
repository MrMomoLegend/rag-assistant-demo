"""classify() from the FYP's eval/common.py: refused / hedged / answered."""
import pytest

from rag.classify import classify

HEDGE = (
    "The provided documents do not contain this information. However, based on the "
    "context, it appears that tf-idf weighting is calculated as the product of two statistics."
)


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"refusal_reason": "retrieval_distance", "answer": "anything"}, "refused"),
        ({"answer": ""}, "refused"),
        ({"answer": "The provided documents do not contain this information."}, "refused"),
        ({"answer": HEDGE}, "hedged"),
        ({"answer": "GPT-4 was released on March 14, 2023."}, "answered"),
        ({"answer": "I don't know."}, "refused"),
    ],
)
def test_classify(result, expected):
    assert classify(result) == expected
