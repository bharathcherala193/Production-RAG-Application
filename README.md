# Ask My Docs — Production RAG over Legal Contracts
[▶ Demo](https://drive.google.com/file/d/1S834Bga_8R2A6KDMbkVFhmZ3A75v7Mlg/view?usp=drive_link)

![CI](https://github.com/bharathcherala193/Production-RAG-Application/actions/workflows/ci-eval.yml/badge.svg)

A domain-specific question-answering system over the [CUAD](https://www.atticusprojectai.org/cuad) legal contracts dataset (510 real commercial contracts), built to demonstrate the full shape of a production RAG pipeline — not just a basic chatbot demo.

Ask a question like *"Is there a non-compete restriction?"* and get an answer grounded in, and cited to, the actual retrieved contract text — with an automated evaluation pipeline that gates on answer quality.

Available in three ways: a **Streamlit UI**, a **one-command Docker deployment**, and as an **MCP server** you can call directly from Claude Desktop or any other MCP client.

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
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                      ▼
     ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
     │ Ragas Evaluation │ │ Streamlit UI │ │ MCP Server │
     │ Faithfulness + │ │ (app.py) │ │ (mcp_server.py) │
     │ Answer Relevancy │ └───────────────────┘ │ → Claude Desktop, │
     └─────────┬─────────┘ │ other MCP │
     │ │ clients │
     ▼ └───────────────────┘
      ┌───────────────────┐
      │ CI Gate │
      │ GitHub Actions — │
      │ fails build if │
      │ scores drop │
      └───────────────────┘


## What it does

- **Hybrid retrieval** — combines BM25 keyword search and vector similarity search, so exact terms/numbers and paraphrased questions are both handled well.
- **Cross-encoder reranking** — re-scores retrieved candidates by reading the query and each candidate together, catching relevance that first-pass retrieval misses.
- **Citation-enforced generation** — every answer cites the excerpt it came from, and the model is instructed to say "I don't know" rather than guess when the context doesn't support an answer.
- **Automated evaluation** — [Ragas](https://github.com/explodinggpt/ragas) measures faithfulness (is the answer grounded in context?) and answer relevancy (does it actually answer the question?) across a test question set.
- **CI-gated** — a GitHub Actions workflow runs the evaluation pipeline on every push and fails the build if quality drops below threshold.
- **MCP-callable** — the same pipeline is exposed as MCP tools (`ask_contract_question`, `search_contracts`), so Claude Desktop or any other MCP client can query the contract corpus directly inside a normal chat.

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
- **Generation & judging:** Llama 3.1 8B via Ollama (local, GPU-accelerated when available)
- **Evaluation:** Ragas
- **CI:** GitHub Actions
- **Demo UI:** Streamlit
- **Deployment:** Docker + Docker Compose
- **Agent integration:** MCP (Model Context Protocol)

## Project structure

- `Ingest.py` — chunk CUAD contracts, embed, store in ChromaDB
- `retrieval.py` — hybrid (BM25 + vector) retrieval
- `rerank.py` — cross-encoder reranking
- `generate.py` — citation-enforced answer generation
- `evaluate.py` — Ragas evaluation over a local test set
- `eval_ci.py` — same evaluation, run against a small fixture set — used in CI
- `app.py` — Streamlit demo UI
- `mcp_server.py` — MCP server exposing the pipeline as callable tools
- `export_demo_contracts.py` — samples a small subset of CUAD into `demo_contracts/` for the lightweight demo
- `Dockerfile` — image for the Streamlit app and the MCP server (shared base)
- `docker-compose.yml` — orchestrates `ollama`, `ollama-pull`, `app`, and the on-demand `mcp` service
- `.github/workflows/ci-eval.yml` — CI evaluation gate

## Running it locally (without Docker)

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

## Running it with Docker

Brings up Ollama (GPU-accelerated, if available) and the Streamlit app together — no local Python environment needed.

```bash
docker compose up --build -d
```

This will:
1. Build the app image (CPU-only PyTorch is used for the embedding/reranking models inside the image — see note below — while Ollama itself gets full GPU acceleration via its own container).
2. Start `ollama`, wait for it to be healthy.
3. Pull `llama3.1:8b` into a persisted volume (`ollama_data`) via the one-shot `ollama-pull` service — only downloads once; cached across restarts.
4. Start the Streamlit app at **http://localhost:8501**, with `chroma_db/` mounted from the host so your already-built index is used directly.

**GPU support:** requires an up-to-date NVIDIA driver on the host and Docker Desktop's WSL2 backend (Windows) — no separate container toolkit install needed. Verify with:
```bash
docker compose exec ollama nvidia-smi
```

**Why CPU-only torch in the app image:** the CUDA build of PyTorch pulls in several GB of NVIDIA toolkit wheels, which is unnecessary here — only the embedding and reranking models in `app.py`/`mcp_server.py` use `torch` directly, and those are small enough to run well on CPU. The LLM itself (the actual expensive part) still gets full GPU acceleration through the separate `ollama` container.

## MCP server — using this as a Claude Desktop tool

The same pipeline is available as an MCP server (`mcp_server.py`), exposing two tools:

- **`ask_contract_question(question, top_k=5)`** — runs the full pipeline and returns a grounded, cited answer.
- **`search_contracts(question, top_k=5)`** — retrieval + reranking only, returns the matching excerpts without generation (useful for inspecting what the retriever finds).

### Option A — run it directly (local Python environment)

```bash
pip install -r requirements.txt   # includes mcp[cli]
python mcp_server.py
```

To test interactively before wiring it into a client, use the MCP Inspector:
```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```
(Avoid `mcp dev mcp_server.py` for this project — it launches the server inside a fresh, isolated `uv`-managed environment that won't have `langchain`, `sentence-transformers`, etc. installed. Running against the system Python directly, as above, is the one that works here.)

### Option B — run it via Docker

A third service, `mcp`, is defined in `docker-compose.yml`, built from the same image as `app` but running `mcp_server.py` instead of Streamlit. It's tagged with `profiles: [tools]` so it's **excluded** from a normal `docker compose up` (an MCP stdio server has nothing to do without a client attached to it) and is only started on demand:

```bash
docker compose run --rm -T mcp
```

### Connecting to Claude Desktop

Add to `claude_desktop_config.json` (on Windows:
`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ask-my-docs": {
      "command": "docker",
      "args": ["compose", "-f", "/absolute/path/to/docker-compose.yml", "run", "--rm", "-T", "mcp"]
    }
  }
}
```

Restart Claude Desktop fully (quit from the system tray, not just close the window). `ask-my-docs` should then appear under Connectors, with `ask_contract_question` and `search_contracts` available as tools in any chat.

**Note:** `docker compose up -d ollama` (or the full stack) needs to be running separately for tool calls to succeed — the `mcp` container itself starts fine either way, but generation requires Ollama to be reachable.

