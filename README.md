# ESG Intelligence Platform

Automated ESG analyst that ingests sustainability reports, retrieves evidence, summarizes company posture, benchmarks it against peer scores, and reports evaluation metrics. Reachable via CLI, REST API, Python SDK, and a React dashboard.

## What It Does

- **Agentic workflow** – Retriever → Summarizer → Comparator → Evaluator agents run end-to-end; use `python -m agents.pipeline TICKER` or `GET /api/analyze` to trigger the flow.
- **Internal RAG infra** – sentence-transformer embeddings, FAISS index, ingestion/index scripts, and benchmark CSVs keep the knowledge base reproducible.
- **Developer touchpoints** – FastAPI service with `/api/analyze`, `/api/evaluate`, `/api/peers`, SDK (`ESGIntelClient`), and a Vite/React dashboard for quick demos.
- **Evaluation ready** – CLI & API compute RMSE, cosine similarity, precision/recall/F1, coverage, risk depth, evidence traceability, and factual consistency so model quality is measurable.

---
## Quick Start

### 1. Backend / Agents
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add HF_TOKEN / HF_MODEL
python -m rag.ingest   # build data/processed/passages.csv (only after changing raw CSVs)
python -m rag.index    # build FAISS index
python scripts/build_peer_data.py  # refresh dashboard dropdown
uvicorn api.main:app --reload
```
Key `.env` vars:
- `HF_TOKEN` – Hugging Face access token.
- `HF_MODEL` – e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- `HF_API_URL` – optional direct endpoint (`https://router.huggingface.co/hf-inference`) to avoid flaky providers.

### 2. Dashboard (optional)
```bash
cd dashboard
npm install
npm run dev  # serves http://localhost:5174 and proxies /api → 8000
```

---
## Agentic Pipeline

| Stage | Description |
| --- | --- |
| RetrieverAgent | pulls top passages per ticker from the FAISS index. |
| SummarizerAgent | uses Hugging Face Llama 3.1 via `InferenceClient` to emit structured E/S/G bullets + risks. |
| ComparatorAgent | benchmarks against peers using `eval/datasets/esg_benchmarks.csv` (augmented with public ESG score datasets). |
| EvaluatorAgent | scores summary coverage, balance, risk depth, evidence traceability, and factual consistency; returns recommendations. |

CLI example:
```bash
python -m agents.pipeline AMZN --peers AAPL,GOOG --query "renewable energy progress"
```

---
## REST API & SDK

| Endpoint | Purpose |
| --- | --- |
| `GET /api/analyze?company=MSFT` | Runs the full agent chain; accepts `query`, `peers`, `top_k`, `source_type`. |
| `POST /api/pipeline` | Same as above via JSON body. |
| `GET /api/evaluate?companies=MSFT,AAPL` | Executes the pipeline for multiple tickers and returns benchmark/eval metrics. |
| `POST /api/evaluate` | Score stored pipeline outputs (JSON/JSONL array). |
| `GET /api/peers?industry=Internet%20Retail&limit=5` | Returns peer rows from the benchmark CSV (NaNs sanitized). |
| `GET /health` | Model readiness info. |

Python SDK:
```python
from api.sdk import ESGIntelClient
client = ESGIntelClient(base_url="http://localhost:8000")
result = client.analyze("AMZN", top_k=3)
report = client.evaluate([result])
peers = client.peers(industry="Internet Retail", limit=5)
```

---
## React Dashboard

Vite + React UI for running analyses and viewing summary quality + peer tables. Update `dashboard/src/peerData.json` by running `python scripts/build_peer_data.py` whenever you change `data/raw/esg_news.csv`.

---
## Data & Ingestion

- `data/raw/esg_news.csv` – sample ESG passages (2014‑2021) extracted from public sustainability reports/filings; swap with your own CSVs (columns: `ticker`, `company_name`, `title`, `text`, `date`, etc.).
- `eval/datasets/esg_benchmarks.csv` – structured ESG scores (includes public datasets such as Kaggle’s 2023‑24 S&P 500 ESG data) powering the comparator/evaluator.

After adding new text (PDF extractions, news, etc.):
```bash
python -m rag.ingest
python -m rag.index
python scripts/build_peer_data.py
```

---
## Evaluation Pipeline

1. Collect outputs (JSON/JSONL) from `agents.pipeline` runs.
2. Score them:
```bash
python -m eval.evaluator --predictions runs/pipeline_outputs.jsonl --output runs/eval_report.json
```
Reports include RMSE, cosine similarity, precision/recall/F1, coverage, balance, risk depth, evidence traceability, and factual consistency plus aggregate summaries.

---
## Tech Stack

- **LLM / embeddings** – Hugging Face Llama 3.1 (InferenceClient) & sentence-transformers.
- **Vector store** – FAISS index for retrieval.
- **Backend** – FastAPI + pandas/NumPy + evaluation modules.
- **Frontend** – React + Vite + proxy to FastAPI.
- **SDK** – lightweight HTTP client (`api/sdk/ESGIntelClient`).

---
## Extending the Platform

- Add fresh PDFs or news feeds to `data/raw/`, then rerun ingest/index.
- Swap FAISS with Pinecone/Weaviate if you need hosted vector stores.
- Add CI (e.g., GitHub Actions) to run `python -m eval.evaluator` on fixtures for regression testing.
- Integrate Semantic Kernel/CrewAI if you want more agent planning or tool routing.

