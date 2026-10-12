"""
Configuration. Every retrieval and decoding value is the FYP's (CM3070,
github.com/MrMomoLegend/ai-orchestrator at e90dc2e, main.py). Only the
generation provider and the limits marked "new" are specific to the demo.

Never retune DISTANCE_THRESHOLD for the demo: 0.53 is the midpoint of the
margin measured in the FYP (answerable max 0.4799, out-of-corpus min 0.5915),
and PARITY.md reports against it.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # local only; the hosted app gets these from Streamlit Secrets

# Retrieval (FYP)
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
DISTANCE_THRESHOLD = 0.53  # cosine distance: 0 identical, 2 opposite
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
COLLECTION_NAME = "documents_sentence"
CORPUS_DIR = ROOT / "corpus"
EXPECTED_CHUNKS = 1551

# Generation (FYP)
REFUSAL = "The provided documents do not contain this information."
LLM_TEMPERATURE = 0.0
LLM_SEED = 42

# New for the demo
MAX_TOKENS = 1024  # Workers AI defaults to 256 for Llama 3.1 8B; FYP answers ran longer
MAX_QUESTION_CHARS = 300
DEFAULT_LLM_MODEL = "@cf/meta/llama-3.1-8b-instruct-fp8"


def llm_model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
