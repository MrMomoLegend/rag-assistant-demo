"""
check_llm.py: Monday's pass-through checks (BUILD_HANDOVER §8, Mon step 5).

Cloudflare's OpenAI-compatible page doesn't document `seed` or `max_tokens`.
  1. seed:       the same grounded question twice -> identical answers
  2. max_tokens: one long model-alone answer -> completion_tokens > 256 and
                 finish_reason "stop" (256 is Workers AI's default cap)
Costs ~60 neurons. Usage:  python scripts\\check_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import pipeline  # noqa: E402

q = "When was GPT-4 released?"
a = pipeline.answer_question(q)
b = pipeline.answer_question(q)
seed_ok = a["answer"] == b["answer"]
print(f"[seed]       same answer twice: {seed_ok}")
print(f"             {a['answer'][:200]!r}")

long_q = "Explain how the transformer architecture works in detail."
c = pipeline.answer_question(long_q, use_rag=False)
tok, fin = c["completion_tokens"], c["finish_reason"]
max_ok = (tok or 0) > 256 and fin == "stop"
print(f"[max_tokens] completion_tokens={tok}, finish_reason={fin}: {'OK' if max_ok else 'CHECK'}")
print(f"             prompt_tokens (grounded GPT-4 question) = {a['prompt_tokens']}")

if seed_ok and max_ok:
    print("\nBoth pass. Keep the OpenAI-compatible endpoint.")
else:
    print("\nAt least one failed. Tell Claude: rag/llm.py switches to the native /ai/run/ endpoint.")
    if not max_ok and tok is not None and tok <= 256:
        print("(If the answer is simply short, the check is inconclusive rather than failed.)")
