from __future__ import annotations

import importlib
from pathlib import Path

from rag.ingest import run_ingest
from rag.index import run_index

RAW_PATH = Path("data/raw/esg_news.csv")
PASSAGES_PATH = Path("data/processed/passages.parquet")
FAISS_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.parquet")


def test_rag_pipeline():
    assert RAW_PATH.exists(), "Expected raw ESG CSV to exist"

    with RAW_PATH.open("r", encoding="utf-8") as source:
        rows = sum(1 for _ in source) - 1  # subtract header
    assert rows > 1, "Raw CSV must have more than one row"

    df = run_ingest(RAW_PATH, PASSAGES_PATH)
    assert PASSAGES_PATH.exists(), "Passages parquet not created"
    assert len(df) > 0, "No passages were ingested"

    run_index(PASSAGES_PATH)
    assert FAISS_PATH.exists()
    assert META_PATH.exists()

    retrieve_module = importlib.reload(importlib.import_module("rag.retrieve"))

    results = retrieve_module.retrieve("MSFT", "renewable energy", k=3)
    assert isinstance(results, list)
    assert len(results) >= 1

    expected_keys = {
        "score",
        "similarity",
        "recency_boost",
        "chunk_id",
        "company",
        "title",
        "source_type",
        "source_url",
        "published_at",
        "text",
    }
    for hit in results:
        assert expected_keys.issubset(hit.keys())
