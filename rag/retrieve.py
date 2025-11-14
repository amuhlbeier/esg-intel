from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd

from rag.embeddings import embed_texts

INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.csv")

_INDEX: faiss.Index | None = None
_META: pd.DataFrame | None = None


def _load_resources() -> tuple[faiss.Index, pd.DataFrame]:
    global _INDEX, _META
    if _INDEX is None or _META is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("Index files not found. Run `python -m rag.index` first.")
        _INDEX = faiss.read_index(str(INDEX_PATH))
        _META = pd.read_csv(META_PATH)
        if _INDEX.ntotal != len(_META):
            raise ValueError("Meta rows and FAISS index entries differ")
    return _INDEX, _META


def _normalize_company(value: str | None) -> str:
    return (value or "").strip().upper()


def _normalize_source_filter(source_type: str | list[str] | None) -> list[str] | None:
    if source_type is None:
        return None
    if isinstance(source_type, str):
        if not source_type:
            return None
        return [source_type.lower()]
    normalized = [item.lower() for item in source_type if item]
    return normalized or None


def _recency_boost(published_at: str | None) -> float:
    if not published_at:
        return 0.0
    try:
        if published_at.endswith("Z"):
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    except ValueError:
        return 0.0

    now = datetime.now(timezone.utc)
    delta = now - dt
    days = delta.total_seconds() / 86400
    if days < 0:
        days = 0.0
    return float(0.5 ** (days / 90.0))


def retrieve(
    company: str,
    query: str,
    k: int = 8,
    source_type: str | list[str] | None = None,
) -> List[dict]:
    if not query:
        raise ValueError("Query must not be empty")

    company_code = _normalize_company(company)
    if not company_code:
        raise ValueError("Company (ticker) must not be empty")

    index, meta = _load_resources()
    if index.ntotal == 0:
        return []

    query_vec = embed_texts([query])[0]
    query_vec = np.asarray(query_vec, dtype="float32")[None, :]

    candidate_count = index.ntotal
    similarities, indices = index.search(query_vec, candidate_count)

    source_filter = _normalize_source_filter(source_type)

    results: List[dict] = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx < 0:
            continue
        row = meta.iloc[int(idx)]
        if _normalize_company(row.get("company")) != company_code:
            continue
        if source_filter is not None and row.get("source_type", "").lower() not in source_filter:
            continue
        recency = _recency_boost(row.get("published_at"))
        score = 0.8 * float(sim) + 0.2 * recency
        results.append(
            {
                "score": score,
                "similarity": float(sim),
                "recency_boost": recency,
                "chunk_id": row.get("chunk_id"),
                "company": row.get("company"),
                "title": row.get("title"),
                "source_type": row.get("source_type"),
                "source_url": row.get("source_url"),
                "published_at": row.get("published_at"),
                "text": row.get("text"),
            }
        )
        if len(results) >= k:
            break

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:k]


# Load once at import as requested; ignore if assets are not ready yet.
try:  # pragma: no cover - executed during module import
    _load_resources()
except FileNotFoundError:
    pass
