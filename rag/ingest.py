from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

RAW_DATA_PATH = Path("data/raw/esg_news.csv")
PROCESSED_DIR = Path("data/processed")
PASSAGES_PATH = PROCESSED_DIR / "passages.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_LANGUAGE = "en"
SOURCE_TYPE = "report"
VERSION = "v1"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    company_name: str
    title: str
    source_type: str
    source_url: str
    published_at: str | None
    ingested_at: str
    text: str
    char_start: int
    char_end: int
    language: str
    ticker_confidence: float
    version: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[tuple[str, int, int]]:
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: List[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append((text[start:end], start, end))
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def _normalize_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_doc_id(ticker: str, date_value: str, title: str, text: str) -> str:
    seed = f"{ticker.upper()}|{date_value}|{title}|{text[:300]}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _to_chunks(row: dict, ingested_at: str) -> List[Chunk]:
    text = _clean_text(row.get("text"))
    if not text:
        return []

    ticker = (row.get("ticker") or row.get("company") or "").strip().upper()
    company_name = row.get("company_name") or row.get("company") or ticker
    title = row.get("title") or ""
    published_at = _normalize_timestamp(row.get("date"))
    source_url = row.get("url") or ""

    doc_id = _build_doc_id(ticker, row.get("date", ""), title, text)

    ticker_confidence = 1.0 if ticker else 0.0

    chunks: List[Chunk] = []
    for idx, (chunk_text, start, end) in enumerate(_chunk_text(text)):
        chunk_id = f"{doc_id}:::{idx}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                company=ticker,
                company_name=company_name,
                title=title,
                source_type=SOURCE_TYPE,
                source_url=source_url,
                published_at=published_at,
                ingested_at=ingested_at,
                text=chunk_text,
                char_start=start,
                char_end=end,
                language=DEFAULT_LANGUAGE,
                ticker_confidence=ticker_confidence,
                version=VERSION,
            )
        )
    return chunks


def run_ingest(
    raw_csv_path: Path | str = RAW_DATA_PATH,
    passages_path: Path | str = PASSAGES_PATH,
) -> pd.DataFrame:
    raw_path = Path(raw_csv_path)
    output_path = Path(passages_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing input CSV at {raw_path}")

    df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    ingested_at = _now_iso()
    all_chunks: list[Chunk] = []
    for row in df.to_dict(orient="records"):
        all_chunks.extend(_to_chunks(row, ingested_at=ingested_at))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = [chunk.__dict__ for chunk in all_chunks]
    if records:
        output_df = pd.DataFrame(records)
        rows_written = len(output_df)
    else:
        output_df = pd.DataFrame(columns=list(Chunk.__annotations__.keys()))
        rows_written = 0

    output_df.to_csv(output_path, index=False)

    manifest_record = {
        "input": str(raw_path),
        "output": str(output_path),
        "rows": rows_written,
        "ingested_at": ingested_at,
        "version": VERSION,
    }
    with MANIFEST_PATH.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(manifest_record) + "\n")

    return output_df


if __name__ == "__main__":
    run_ingest()
