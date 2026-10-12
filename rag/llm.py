"""
The generation call: Cloudflare Workers AI through its OpenAI-compatible
endpoint. Provider = three env vars (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).

The client is built on first use, not at import, so tests run without a .env
and a missing secret on the hosted app gives a readable error, not a crash.
"""
import os

from .config import LLM_SEED, LLM_TEMPERATURE, MAX_TOKENS, llm_model

_client = None


class LLMNotConfigured(RuntimeError):
    """LLM_BASE_URL or LLM_API_KEY is missing."""


def _get_client():
    global _client
    if _client is None:
        base_url, api_key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
        if not base_url or not api_key:
            raise LLMNotConfigured(
                "LLM_BASE_URL and LLM_API_KEY must be set (.env locally, Streamlit Secrets when hosted)."
            )
        from openai import OpenAI

        _client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=1)
    return _client


def chat(prompt: str) -> dict:
    # One user message and no system prompt: the same shape as the FYP's ollama.chat call.
    r = _get_client().chat.completions.create(
        model=llm_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        max_tokens=MAX_TOKENS,
    )
    usage = getattr(r, "usage", None)
    return {
        "answer": r.choices[0].message.content or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "finish_reason": r.choices[0].finish_reason,
    }
