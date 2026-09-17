# Ask My Docs — Production RAG over Legal Contracts

![CI](https://github.com/bharathcherala193/Production-RAG-Application/actions/workflows/ci-eval.yml/badge.svg)

A domain-specific question-answering system over the [CUAD](https://www.atticusprojectai.org/cuad) legal contracts dataset (510 real commercial contracts), built to demonstrate the full shape of a production RAG pipeline — not just a basic chatbot demo.

Ask a question like *"Is there a non-compete restriction?"* and get an answer grounded in, and cited to, the actual retrieved contract text — with an automated evaluation pipeline that gates on answer quality.

## Architecture


                          User Question
                                │
                                ▼
                  ┌──────────────────────────┐
                  │   Hybrid Retrieval        │
                  │   BM25 (keyword) +        │
                  │   Vector Search (Chroma)  │
                  │   → top 20 candidates     │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   Cross-Encoder Rerank    │
                  │   ms-marco-MiniLM-L-6-v2  │
                  │   → top 5 candidates      │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Citation-Enforced        │
                  │  Generation               │
                  │  Llama 3.1 8B (Ollama)    │
                  │  → answer + [1][2] cites  │
                  └────────────┬─────────────┘
                               │
                               ▼
                      Answer + Sources
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
      ┌───────────────────┐         ┌───────────────────┐
      │  Ragas Evaluation  │         │   Streamlit UI     │
      │  Faithfulness +    │         │   (app.py)          │
      │  Answer Relevancy  │         └───────────────────┘
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │  CI Gate           │
      │  GitHub Actions —   │
      │  fails build if     │
      │  scores drop        │
      └───────────────────┘


      
## What it does

- **Hybrid retrieval** — combines BM25 keyword search and vector similarity search, so exact terms/numbers and paraphrased questions are both handled well.
- **Cross-encoder reranking** — re-scores retrieved candidates by reading the query and each candidate together, catching relevance that first-pass retrieval misses.
- **Citation-enforced generation** — every answer cites the excerpt it came from, and the model is instructed to say "I don't know" rather than guess when the context doesn't support an answer.
- **Automated evaluation** — [Ragas](https://github.com/explodinggpt/ragas) measures faithfulness (is the answer grounded in context?) and answer relevancy (does it actually answer the question?) across a test question set.
- **CI-gated** — a GitHub Actions workflow runs the evaluation pipeline on every push and fails the build if quality drops below threshold.

Runs **entirely locally and free** — no API keys, no usage limits. Embeddings and reranking use `sentence-transformers`; generation and judging use local LLMs via [Ollama](https://ollama.com).

## Results

| Metric | Before tuning | After tuning |
|---|---|---|
| Faithfulness (avg) | 0.77-0.82 | **1.00** |
| Answer relevancy (avg) | 0.56-0.72 (inconsistent) | **0.71** (stable) |

Improved by diagnosing two specific failure patterns: a weak generation model producing vague, uncited answers, and a prompt style that caused the model to make unverifiable "this excerpt doesn't apply" claims. Fixed by switching to a larger local model for generation and tightening the prompt to answer directly from only the relevant retrieved excerpts.

## Tech stack

- **Orchestration:** LangChain
- **Vector store:** ChromaDB
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **Generation & judging:** Llama 3.1 8B via Ollama (local)
- **Evaluation:** Ragas
- **CI:** GitHub Actions
- **Demo UI:** Streamlit

## Project structure

- Ingest.py # Chunk CUAD contracts, embed, store in ChromaDB
- retrieval.py # Hybrid (BM25 + vector) retrieval
- rerank.py # Cross-encoder reranking
- generate.py # Citation-enforced answer generation
- evaluate.py # Ragas evaluation over a local test set
- eval_ci.py # Same evaluation, run against a small fixture set — used in CI
- app.py # Streamlit demo UI
- .github/workflows/ci-eval.yml # CI evaluation gate

  ## Running it locally

```bash
pip install -r requirements.txt

# Pull local models (one-time)
ollama pull llama3.1:8b

# Build the index (downloads CUAD separately — see data/cuad/)
python Ingest.py

# Try the pipeline
python generate.py

# Run the evaluation gate
python evaluate.py

# Launch the demo UI
streamlit run app.py
```

