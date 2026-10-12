"""
parity_check.py: does the hosted demo reproduce the FYP's 30-question result?

Part 1 (--retrieval-only, no model calls, free):
    chunk count is 1,551; for each question the nearest cosine distance is
    within +/-0.002 of the FYP baseline; no question crosses the 0.53 gate.
Part 2 (default, ~1,000 Cloudflare neurons, ~10 min):
    all 30 questions with RAG off, and RAG on twice (determinism check).
    Writes results/parity_responses.csv and results/parity_summary.csv.

Usage (from the repo root, venv active):
    python scripts\\parity_check.py --retrieval-only
    python scripts\\parity_check.py
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import config, pipeline  # noqa: E402
from rag.classify import classify  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions_rag.csv"
BASELINE = ROOT / "results" / "fyp_baseline_rag_responses.csv"
TOL = 0.002


def retrieval_parity(qs, base_on):
    coll = pipeline.get_collection()  # builds the index; asserts 1,551
    print(f"Index: {coll.count()} chunks (expected {config.EXPECTED_CHUNKS})")
    rows, flips, drifts = [], [], []
    for q in qs.itertuples():
        res = coll.query(query_texts=[q.question], n_results=config.TOP_K)
        d = round(float(res["distances"][0][0]), 4)
        b = float(base_on.loc[q.qid, "nearest_distance"])
        gate_now = d > config.DISTANCE_THRESHOLD
        gate_then = b > config.DISTANCE_THRESHOLD
        delta = round(d - b, 4)
        if abs(delta) > TOL:
            drifts.append(q.qid)
        if gate_now != gate_then:
            flips.append(q.qid)
        rows.append({"qid": q.qid, "fyp_nearest": b, "demo_nearest": d, "delta": delta,
                     "fyp_gate": "decline" if gate_then else "pass",
                     "demo_gate": "decline" if gate_now else "pass"})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "parity_retrieval.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nMax |delta|: {df.delta.abs().max():.4f} (tolerance {TOL})")
    print(f"Outside tolerance: {drifts or 'none'}")
    print(f"Gate flips: {flips or 'none'}")
    print(f"-> {out}")
    if flips:
        print("\nSTOP: a question crossed the 0.53 gate. Record it in PARITY.md. Do NOT retune.")
    return not flips and not drifts


def model_run(qs):
    rows = []
    plan = [("RAG off", False, 1), ("RAG on", True, 1), ("RAG on", True, 2)]
    total = len(qs) * len(plan)
    i = 0
    for cond, use_rag, run in plan:
        for q in qs.itertuples():
            i += 1
            sys.stdout.write(f"\r  {i}/{total} {cond} run {run} {q.qid}   ")
            sys.stdout.flush()
            try:
                r = pipeline.answer_question(q.question, use_rag=use_rag)
                err = None
            except Exception as exc:  # keep going; record it
                r, err = {"answer": "", "refusal_reason": None}, f"{type(exc).__name__}: {exc}"
            rows.append({
                "qid": q.qid, "category": q.category, "expected": q.expected,
                "question": q.question, "condition": cond, "run": run,
                "answer": r.get("answer", ""), "outcome": classify(r) if not err else "error",
                "refusal_reason": r.get("refusal_reason"),
                "nearest_distance": (r.get("distances") or [None])[0],
                "sources": "; ".join(dict.fromkeys(r.get("sources") or [])),
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "finish_reason": r.get("finish_reason"),
                "total_s": r.get("total_s"), "error": err,
            })
            time.sleep(0.2)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "parity_responses.csv", index=False)

    on1 = df[(df.condition == "RAG on") & (df.run == 1)].set_index("qid")
    on2 = df[(df.condition == "RAG on") & (df.run == 2)].set_index("qid")
    off = df[df.condition == "RAG off"].set_index("qid")
    un = on1.index[on1.category != "answerable"]
    an = on1.index[on1.category == "answerable"]
    summary = {
        "unanswerable_answered_rag_off": int((off.loc[un].outcome != "refused").sum()),
        "unanswerable_answered_rag_on": int((on1.loc[un].outcome != "refused").sum()),
        "declined_by_gate": int((on1.loc[un].refusal_reason == "retrieval_distance").sum()),
        "declined_by_model": int(((on1.loc[un].refusal_reason != "retrieval_distance")
                                  & (on1.loc[un].outcome == "refused")).sum()),
        "answerable_answered_rag_on": int((on1.loc[an].outcome == "answered").sum()),
        "answerable_hedged_rag_on": int((on1.loc[an].outcome == "hedged").sum()),
        "answerable_refused_rag_on": int((on1.loc[an].outcome == "refused").sum()),
        "rag_on_identical_across_runs": int((on1.answer == on2.answer).sum()),
        "errors": int(df.error.notna().sum()),
        "prompt_tokens_total": int(pd.to_numeric(df.prompt_tokens, errors="coerce").fillna(0).sum()),
        "completion_tokens_total": int(pd.to_numeric(df.completion_tokens, errors="coerce").fillna(0).sum()),
        # Cloudflare reports finish_reason "stop" even when it cuts at max_tokens
        # (seen 12 Oct: 1024 tokens, "stop"), so count hits on the cap instead.
        "hit_max_tokens": int((pd.to_numeric(df.completion_tokens, errors="coerce") >= config.MAX_TOKENS).sum()),
    }
    pd.Series(summary).to_csv(ROOT / "results" / "parity_summary.csv", header=["value"])
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("-> results/parity_responses.csv, results/parity_summary.csv")
    print("Correctness on the 13 paired answerable questions is scored by hand into PARITY.md.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    qs = pd.read_csv(QUESTIONS)
    base = pd.read_csv(BASELINE)
    base_on = base[base.condition == "RAG on"].set_index("qid")
    assert len(qs) == 30 and len(base_on) == 30

    ok = retrieval_parity(qs, base_on)
    if args.retrieval_only:
        sys.exit(0 if ok else 1)
    if not ok:
        print("\nRetrieval parity failed; running the model anyway so the difference is measured.")
    model_run(qs)


if __name__ == "__main__":
    main()
