"""
Ask the documents: a hosted demo of the retrieval-augmented assistant from
my CM3070 final project. Run locally with:  streamlit run app.py
"""
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from rag import config, llm, pipeline
from rag.classify import classify

FYP_REPO = "https://github.com/MrMomoLegend/ai-orchestrator/tree/e90dc2e"
LINKEDIN = "https://www.linkedin.com/in/mokith-maheshkumar/"

# Chosen from the parity run on Tue 13: keep only questions that behave as described.
EXAMPLES = [
    "Which researchers is the CYK algorithm named after?",                          # A03: model alone names the wrong researchers; with documents, Cocke, Younger, Kasami
    "How does the fastText subword model represent a word?",                        # A09: grounded answer (sum of subword vectors)
    "What two tasks is BERT pretrained on?",                                        # A08: grounded answer
    "What word error rate does OpenAI's Whisper model achieve on LibriSpeech?",     # C03: model alone varies: sometimes a number, sometimes a refusal; with documents it declines
    "What is the capital of Brazil?",                                               # B01: declined by the gate, no model call
]

MODES = {
    "Compare": "compare",
    "With the documents": "rag",
    "Model alone": "bare",
}

st.set_page_config(page_title="Ask the documents", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building the index (1,551 passages, about a minute on a cold start)...")
def get_index():
    return pipeline.get_collection()


@st.cache_data(show_spinner=False, max_entries=500)
def cached_answer(question: str, use_rag: bool) -> dict:
    return pipeline.answer_question(question, use_rag=use_rag)


def normalise(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def run(question: str, use_rag: bool) -> dict:
    """One answer, never raising: failures come back as a dict the UI can show."""
    key = (question, use_rag)
    seen = st.session_state.setdefault("seen", set())
    from_cache = key in seen
    try:
        r = dict(cached_answer(question, use_rag))
        seen.add(key)
        r["cached"] = from_cache
        return r
    except llm.LLMNotConfigured:
        msg = "The model isn't configured on this deployment (missing API secrets)."
    except Exception as exc:  # quota, network, bad key
        text = str(exc).lower()
        if "429" in text or "quota" in text or "limit" in text or "neuron" in text:
            msg = "Today's free model quota is used up. It resets at 05:30 IST (00:00 UTC)."
        else:
            msg = "The model service didn't respond. Try again in a minute."
    # Still show the passages for a grounded question.
    r = {"answer": "", "error": msg, "use_rag": use_rag, "sources": [], "distances": [],
         "retrieved_chunks": [], "refusal_reason": None}
    if use_rag:
        try:
            res = get_index().query(query_texts=[question], n_results=config.TOP_K)
            r["retrieved_chunks"] = res["documents"][0]
            r["sources"] = [m.get("source", "unknown") for m in res["metadatas"][0]]
            r["distances"] = [round(float(d), 4) for d in res["distances"][0]]
        except Exception:
            pass
    return r


def status_line(r: dict) -> str:
    if r.get("error"):
        return f"⚠️ {r['error']}"
    if not r.get("use_rag"):
        return ""
    outcome = classify(r)
    nearest = r["distances"][0] if r.get("distances") else None
    if r.get("refusal_reason") == "retrieval_distance":
        far = f"{nearest:.2f}" if nearest is not None else "n/a"
        return (f"🚫 **Declined before the model was called:** the closest passage is {far} away; "
                f"the limit is {config.DISTANCE_THRESHOLD}.")
    if outcome == "refused":
        return "🚫 **Declined by the model:** the passages didn't contain it."
    if outcome == "hedged":
        return ("⚠️ **Hedged:** the model said it couldn't answer, then answered anyway. "
                "Treat this as unsupported.")
    return "✅ **Answered from the retrieved passages.**"


def pretty_source(name: str) -> str:
    return name.rsplit(".", 1)[0]


def show_sources(r: dict):
    chunks, sources, dists = r.get("retrieved_chunks") or [], r.get("sources") or [], r.get("distances") or []
    if not chunks:
        return
    st.markdown("**Sources** (retrieval order, cosine distance; lower is closer)")
    order = list(dict.fromkeys(sources))
    for src in order:
        items = [(c, d) for c, s, d in zip(chunks, sources, dists) if s == src]
        best = min(d for _, d in items)
        with st.expander(f"{pretty_source(src)} · {len(items)} passage(s) · nearest {best:.3f}"):
            for c, d in items:
                st.caption(f"distance {d:.3f}")
                st.text(c)


def show_details(r: dict):
    with st.expander("How this answer was made"):
        rows = {
            "Model": config.llm_model(),
            "Provider": "Cloudflare Workers AI",
            "Temperature / seed / max tokens": f"{config.LLM_TEMPERATURE:g} / {config.LLM_SEED} / {config.MAX_TOKENS}",
        }
        if r.get("use_rag"):
            rows.update({
                "Passages retrieved (k)": config.TOP_K,
                "Decline threshold (cosine distance)": config.DISTANCE_THRESHOLD,
                "Nearest passage": r["distances"][0] if r.get("distances") else "n/a",
                "Retrieval time (s)": r.get("retrieve_s", "n/a"),
            })
        rows.update({
            "Generation time (s)": r.get("generate_s", "n/a"),
            "Tokens in / out": f"{r.get('prompt_tokens') or '-'} / {r.get('completion_tokens') or '-'}",
            "Cached": "yes" if r.get("cached") else "no",
        })
        for k, v in rows.items():
            st.markdown(f"- **{k}:** {v}")


def show_answer(r: dict, title: str):
    st.subheader(title)
    line = status_line(r)
    if line:
        st.markdown(line)
    if r.get("answer"):
        st.markdown(r["answer"])  # HTML stays escaped (unsafe_allow_html is off)
    show_sources(r)
    if not r.get("error"):
        show_details(r)


# ---------------------------------------------------------------- page
st.title("Ask the documents")
st.markdown(
    "A retrieval-augmented assistant over 22 NLP documents. It answers from them, or says it can't. "
    f"Built from my [final-year project]({FYP_REPO}) · [LinkedIn]({LINKEDIN})"
)

if "question" not in st.session_state:
    st.session_state.question = ""

st.markdown("**Try one:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.question = ex
        st.session_state.ask = True

question = st.text_area("Your question", key="question", max_chars=config.MAX_QUESTION_CHARS, height=80)
mode_label = st.radio("Mode", list(MODES), horizontal=True, index=0,
                      help="Compare shows the model alone next to the model with the documents.")
asked = st.button("Ask", type="primary") or st.session_state.pop("ask", False)

if asked:
    q = normalise(question)
    if not q:
        st.warning("Type a question first.")
    elif len(q) < 3:
        st.warning("That's too short to search for.")
    else:
        mode = MODES[mode_label]
        get_index()
        if mode == "compare":
            with st.spinner("Asking twice: with the documents and without..."):
                ctx = get_script_run_ctx()
                with ThreadPoolExecutor(max_workers=2,
                                        initializer=lambda: add_script_run_ctx(ctx=ctx)) as pool:
                    f_rag = pool.submit(run, q, True)
                    f_bare = pool.submit(run, q, False)
                    r_rag, r_bare = f_rag.result(), f_bare.result()
            left, right = st.columns(2)
            with left:
                show_answer(r_rag, "With the documents")
            with right:
                show_answer(r_bare, "Model alone (no documents, no instruction to decline)")
        else:
            with st.spinner("Thinking..."):
                r = run(q, mode == "rag")
            show_answer(r, "With the documents" if mode == "rag"
                        else "Model alone (no documents, no instruction to decline)")

st.divider()
st.caption(
    "This hosted demo sends your question and the retrieved passages to Cloudflare Workers AI "
    "to generate the answer. The original system runs entirely on a laptop. Questions are not logged."
)
st.caption(
    "Corpus: Wikipedia contributors, and Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (d2l.ai), "
    "CC BY-SA 4.0, converted from PDF to plain text."
)
