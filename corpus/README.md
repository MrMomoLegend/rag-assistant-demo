# The document corpus

This folder is the assistant's knowledge base: 22 plain-text files extracted from the FYP's
PDFs (`scripts/extract_corpus.py`). The app splits and embeds them into an in-memory index at
startup; nothing else needs to be run.

## What is in the repository

22 openly licensed documents on natural language processing:

- **Wikipedia articles (13)**: CYK algorithm, GPT-4, Hallucination (artificial
  intelligence), Hidden Markov model, Large language model, Named-entity recognition,
  Part-of-speech tagging, Retrieval-augmented generation, Speech recognition, Viterbi
  algorithm, Word2Vec, n-gram, tf–idf. Licensed CC BY-SA 4.0; © Wikipedia contributors.
  Source: https://en.wikipedia.org
- **Dive into Deep Learning (d2l.ai) sections**: approx-training, bert, glove,
  language-model, sentiment-analysis-and-dataset, similarity-analogy, subword-embedding,
  transformer, word2vec. Zhang, Lipton, Li & Smola, *Dive into Deep Learning*. Text
  licensed CC BY-SA 4.0. Source: https://d2l.ai

Both are redistributed unmodified under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).

## Relation to the final-year project

The FYP's Chapter 5 project-corpus results were measured against the PDFs these files were extracted from, with the
questions in `eval/questions_rag.csv`. The retrieval threshold (0.53) and k (10) were fixed
earlier on a non-redistributable course-reading corpus and were not re-tuned here.
