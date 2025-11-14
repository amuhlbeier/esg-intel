# api/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from huggingface_hub import InferenceClient
from pydantic import BaseModel
import math
import pandas as pd

from config import HF_TOKEN, HF_MODEL
from agents.pipeline import AgentPipeline
from eval.evaluator import evaluate_predictions
from eval.evaluator import _load_benchmarks

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Did you create .env?")
    # Initialize the Hugging Face client
    app.state.hf = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    print(f"[startup] Hugging Face model loaded: {HF_MODEL}")
    yield
    print("[shutdown] App shutting down.")

app = FastAPI(lifespan=lifespan)


class PipelineRequest(BaseModel):
    company: str
    query: str | None = None
    peers: list[str] | None = None
    top_k: int | None = None
    source_type: str | None = None


class EvaluateRequest(BaseModel):
    records: list[dict]

@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL}

@app.get("/api/analyze")
def analyze(
    company: str,
    request: Request,
    query: str | None = None,
    peers: list[str] | None = Query(default=None),
    top_k: int | None = None,
    source_type: str | None = None,
):
    pipeline = AgentPipeline(
        company,
        query=query,
        peers=peers,
        top_k=top_k,
        source_type=source_type,
        llm_client=request.app.state.hf,
    )
    try:
        result = pipeline.run()
    except Exception as exc:  # pragma: no cover - runtime errors only
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload = _sanitize_payload(result.to_dict())
    return jsonable_encoder(payload)


@app.post("/api/pipeline")
def run_pipeline(payload: PipelineRequest, request: Request):
    peers = payload.peers or []
    pipeline = AgentPipeline(
        payload.company,
        query=payload.query,
        peers=peers,
        top_k=payload.top_k,
        source_type=payload.source_type,
        llm_client=request.app.state.hf,
    )
    try:
        result = pipeline.run()
    except Exception as exc:  # pragma: no cover - runtime errors only
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload = _sanitize_payload(result.to_dict())
    return jsonable_encoder(payload)


@app.post("/api/evaluate")
def evaluate(records: EvaluateRequest):
    try:
        report = evaluate_predictions(records.records)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return jsonable_encoder(_sanitize_payload(report))


@app.get("/api/evaluate")
def evaluate_companies(
    request: Request,
    companies: str = Query(..., description="Comma separated tickers"),
    top_k: int | None = None,
):
    tickers = [item.strip().upper() for item in companies.split(",") if item.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide at least one company")

    records = []
    for ticker in tickers:
        pipeline = AgentPipeline(ticker, top_k=top_k, llm_client=request.app.state.hf)
        result = pipeline.run()
        records.append(result.to_dict())

    return jsonable_encoder(_sanitize_payload(evaluate_predictions(records)))


@app.get("/api/peers")
def peers(industry: str | None = None, limit: int = 10):
    df = _load_benchmarks()
    frame = df
    if industry:
        frame = frame[frame["Industry"].str.contains(industry, case=False, na=False)]
    frame = frame.head(limit)
    columns = [
        "Symbol",
        "Name",
        "Sector",
        "Industry",
        "Total ESG Risk score",
        "Environment Risk Score",
        "Social Risk Score",
        "Governance Risk Score",
        "ESG Risk Level",
    ]
    subset = frame.loc[:, columns].astype(object)
    subset = subset.where(pd.notnull(subset), None)
    payload = subset.to_dict(orient="records")
    return jsonable_encoder({"industry": industry, "count": len(payload), "peers": payload})
def _sanitize_payload(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value
