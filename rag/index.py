from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import pandas as pd

from rag.embeddings import PASSAGES_PATH, embed_passages

INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.csv"

META_COLUMNS: Sequence[str] = (
    "chunk_id",
    "doc_id",
    "company",
    "company_name",
    "title",
    "source_type",
    "source_url",
    "published_at",
    "ingested_at",
    "text",
)


def run_index(passages_path: Path | str = PASSAGES_PATH) -> tuple[Path, Path]:
    df, embeddings = embed_passages(passages_path)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))

    meta_df = df.loc[:, META_COLUMNS]
    meta_df.to_csv(META_PATH, index=False)

    return FAISS_PATH, META_PATH


if __name__ == "__main__":
    run_index()
