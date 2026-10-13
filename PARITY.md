# Parity: does the hosted demo reproduce the final-year project?

The demo swaps the FYP's local Llama 3.1 8B (Ollama, laptop GPU) for the same model family on
Cloudflare Workers AI (`@cf/meta/llama-3.1-8b-instruct-fp8`), and Streamlit for the React UI.
Everything else is carried over unchanged: corpus, chunking (500/50), `all-MiniLM-L6-v2` embeddings,
top-k 10, the 0.53 cosine-distance gate, the prompt, and temperature 0 / seed 42.

Run: `python scripts/parity_check.py`, Tue 13 Oct 2026, on the same 30 questions the FYP was
evaluated on (`eval/questions_rag.csv`): 15 answerable, 12 out-of-corpus, 3 related but
unanswerable. Raw output: `results/parity_retrieval.csv`, `results/parity_responses.csv`,
`results/parity_summary.csv`. FYP baseline: `results/fyp_baseline_rag_responses.csv`.

## Results

| | FYP (local) | Demo (hosted) |
|---|---|---|
| Index size | 1,551 chunks | 1,551 chunks |
| Nearest-chunk distance, 30 questions | baseline | **identical on all 30** (max \|Δ\| 0.0000) |
| Gate decisions (pass/decline at 0.53) | 18 / 12 | 18 / 12, **0 flips** |
| Unanswerable questions answered anyway, RAG off | 15 / 15 | 15 / 15 |
| Unanswerable questions answered anyway, RAG on | 0 / 15 | **0 / 15** (12 by the gate, 3 by the model) |
| Paired answerable questions correct, RAG on* | 12 / 13 | **12 / 13** |
| Answers hitting the 1,024-token cap | n/a | 0 |
| Errors | 0 | 0 |

\*The 13 answerable questions the FYP answered (A06 hedged and A10 was refused there, so they're
excluded from the pair, as in the FYP). Scored by hand against `ground_truth_check`. The one miss
is the same question in both: **A04** (Word2Vec's two architectures). Retrieval surfaces the
Doc2Vec passage and the model names PV-DM / PV-DBOW instead of CBOW / skip-gram.

**Conclusion:** the FYP's headline results hold on the hosted model. Grounding still takes
unsupported answers from 15/15 to 0, and answerable accuracy is unchanged.

## Differences worth knowing

- **Run-to-run determinism: 28 / 30 identical**, not 30 / 30. Temperature 0 and seed 42 are sent
  (checked 12 Oct), but A06 and A15 diverged late in the text between two RAG-on runs. The
  outcome stayed the same for A15. A06 flipped between "refused" and "answered" (next point).
  Hosted inference doesn't guarantee bit-identical output, so the demo doesn't claim it.
- **A06 (tf-idf) is mislabelled by the outcome classifier.** The answer is correct (tf × idf),
  but it ends with "The provided documents do not contain the specific formula for … double
  normalization K." The FYP classifier, carried over unchanged, reads a refusal sentence with
  nothing after it as a refusal. It also lists two tf normalisations that aren't in the source,
  so this is an answer-then-disclaimer case with some invented detail. Kept as a known limitation
  instead of tuning the classifier to one question.
- **A10** (why the Transformer needs positional encodings) is refused in both, though the answer is
  in `transformer.txt` (distance 0.3947, passes the gate). The model judges the context
  insufficient. Same behaviour as the FYP.
- **Retrieval nondeterminism (12 Oct):** one earlier retrieval-only run gave B08 0.7255 vs 0.7181,
  most likely HNSW's approximate search. Today all 30 match exactly. The gate margin for B08 is
  large (0.19), so it couldn't flip.

## Latency (hosted, Tue 13 Oct, laptop in India → Cloudflare)

| | n | median | p95 | max |
|---|---|---|---|---|
| RAG on, model called | 36 | 1.31 s | 6.63 s | 8.96 s (A06, 214 tokens) |
| RAG on, declined by the gate (no model call) | 24 | 0.01 s | | |
| RAG off (model alone, long answers) | 30 | 8.31 s | 31.7 s | |

Not comparable to the FYP's 1.90 s median (local GPU, end to end including the UI). It's here so
the demo's own numbers are on record. Excludes the one-off index build (~50 s on a cold start).

Tokens for the whole run: 42,171 prompt + 9,099 completion (90 calls, about 1,000 neurons on the
free plan).
