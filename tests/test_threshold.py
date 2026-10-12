"""
The retrieval-confidence gate (FYP Section 4.5 / 5.4), ported from the FYP's
tests/test_threshold.py. When the gate is wrong nothing raises: the only
visible signal is whether the model was called at all.
"""
import pytest

from rag import config, pipeline
from rag.classify import looks_like_refusal


def test_shipped_threshold_is_the_fyp_value():
    assert config.DISTANCE_THRESHOLD == pytest.approx(0.53)


def test_shipped_top_k_is_ten():
    assert config.TOP_K == 10


def test_refuses_without_calling_the_model_when_nearest_chunk_is_too_far(collection_at, chat_spy):
    collection_at([0.72, 0.81, 0.90])
    r = pipeline.answer_question("Who invented the transistor?")
    assert r["refused"] is True
    assert r["refusal_reason"] == "retrieval_distance"
    assert r["answer"] == config.REFUSAL
    assert chat_spy.called is False


def test_answers_when_nearest_chunk_is_within_threshold(collection_at, chat_spy):
    collection_at([0.31, 0.44, 0.58])
    r = pipeline.answer_question("What is perplexity?")
    assert r["refused"] is False
    assert r["refusal_reason"] is None
    assert chat_spy.called is True


def test_boundary_exactly_at_the_threshold_does_not_refuse(collection_at, chat_spy):
    collection_at([0.53])
    r = pipeline.answer_question("A borderline question")
    assert r["refused"] is False
    assert chat_spy.called is True


def test_boundary_just_beyond_the_threshold_refuses(collection_at, chat_spy):
    collection_at([0.5301])
    r = pipeline.answer_question("A borderline question")
    assert r["refused"] is True
    assert chat_spy.called is False


def test_only_the_nearest_distance_decides(collection_at, chat_spy):
    collection_at([0.60, 0.10, 0.12])
    r = pipeline.answer_question("A question")
    assert r["refused"] is True
    assert chat_spy.called is False


def test_empty_retrieval_refuses(collection_at, chat_spy):
    collection_at([])
    r = pipeline.answer_question("A question against an empty corpus")
    assert r["refused"] is True
    assert chat_spy.called is False


def test_disabling_the_threshold_reaches_the_model(collection_at, chat_spy):
    collection_at([0.95])
    r = pipeline.answer_question("A question", use_threshold=False)
    assert chat_spy.called is True
    assert r["refusal_reason"] != "retrieval_distance"
    assert r["threshold"] is None


def test_a_gate_refusal_costs_no_generation_time(collection_at, chat_spy):
    collection_at([0.88])
    r = pipeline.answer_question("A question")
    assert r["generate_s"] == 0.0


def test_top_k_of_zero_falls_back_to_the_default(collection_at, chat_spy):
    fake = collection_at([0.20])
    r = pipeline.answer_question("A question", top_k=0)
    assert fake.calls[0]["n_results"] == config.TOP_K
    assert r["top_k"] == config.TOP_K


def test_threshold_of_zero_is_honoured(collection_at, chat_spy):
    collection_at([0.01])
    r = pipeline.answer_question("A question", threshold=0.0)
    assert r["refusal_reason"] == "retrieval_distance"
    assert r["distances"] == [0.01]
    assert chat_spy.called is False


def test_model_judgement_refusal_is_labelled_distinctly(collection_at, monkeypatch):
    from rag import llm

    collection_at([0.30])
    monkeypatch.setattr(llm, "chat", lambda prompt: {"answer": config.REFUSAL})
    r = pipeline.answer_question("A question")
    assert r["refused"] is True
    assert r["refusal_reason"] == "model_judgement"


def test_rag_off_never_retrieves_and_sends_the_bare_question(collection_at, chat_spy):
    fake = collection_at([0.10])
    r = pipeline.answer_question("Who wrote the CKY algorithm?", use_rag=False)
    assert fake.calls == []
    assert r["retrieved_chunks"] == []
    assert chat_spy.last_prompt == "Who wrote the CKY algorithm?"


def test_grounded_prompt_is_the_fyp_prompt_byte_for_byte(collection_at, chat_spy):
    collection_at([0.20, 0.30], documents=["first chunk", "second chunk"])
    pipeline.answer_question("What is perplexity?")
    expected = (
        "Answer the question using ONLY the context below. If the answer is not in the "
        "context, say 'The provided documents do not contain this information.'"
        "\n\nContext:\nfirst chunk\n\nsecond chunk\n\nQuestion: What is perplexity?"
    )
    assert chat_spy.last_prompt == expected


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The provided documents do not contain this information.", True),
        ("The context does not contain that.", True),
        ("I cannot answer that from the provided context.", True),
        ("The information is insufficient.", True),
        ("Perplexity is the inverse probability of the test set, normalised.", False),
        ("", False),
    ],
)
def test_refusal_detector(answer, expected):
    assert looks_like_refusal(answer) is expected
