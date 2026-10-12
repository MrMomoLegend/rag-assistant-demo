"""
Deterministic decoding (FYP tests/test_decoding.py), now on the hosted call.
Cloudflare's default temperature is 0.6 and its default max_tokens for this
model is 256, so all three must be sent on every call.
"""
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rag import config, llm


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content="An answer.")
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=3)
        return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_client(monkeypatch):
    comp = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return comp


def test_shipped_decoding_values():
    assert config.LLM_TEMPERATURE == 0.0
    assert config.LLM_SEED == 42
    assert config.MAX_TOKENS == 1024


def test_chat_sends_temperature_seed_and_max_tokens(fake_client):
    out = llm.chat("hello")
    sent = fake_client.calls[-1]
    assert sent["temperature"] == 0.0
    assert sent["seed"] == 42
    assert sent["max_tokens"] == 1024
    assert out["answer"] == "An answer."
    assert out["finish_reason"] == "stop"


def test_chat_sends_one_user_message_and_no_system_prompt(fake_client):
    llm.chat("the prompt")
    assert fake_client.calls[-1]["messages"] == [{"role": "user", "content": "the prompt"}]


def test_model_defaults_to_fp8_and_is_overridable(fake_client, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    llm.chat("x")
    assert fake_client.calls[-1]["model"] == "@cf/meta/llama-3.1-8b-instruct-fp8"
    monkeypatch.setenv("LLM_MODEL", "@cf/meta/llama-3.1-8b-instruct-awq")
    llm.chat("x")
    assert fake_client.calls[-1]["model"] == "@cf/meta/llama-3.1-8b-instruct-awq"


def test_missing_secrets_raise_a_readable_error(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(llm.LLMNotConfigured):
        llm.chat("x")


def test_importing_the_pipeline_needs_no_env_and_no_model():
    """A missing .env must not crash import (pytest before step 4, hosted app startup)."""
    code = (
        "import os\n"
        "for k in ('LLM_BASE_URL','LLM_API_KEY','LLM_MODEL'): os.environ.pop(k, None)\n"
        "import rag.config as c\n"
        "c.load_dotenv = lambda *a, **k: None\n"
        "import rag.pipeline, rag.llm\n"
        "assert rag.llm._client is None\n"
        "assert rag.pipeline._collection is None\n"
        "print('ok')\n"
    )
    root = str(config.ROOT)
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
