from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PASSAGES_PATH = Path("data/processed/passages.csv")
MODEL_CANDIDATES = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]

_MODEL: SentenceTransformer | None = None
_MODEL_NAME: str | None = None


def _load_model() -> SentenceTransformer:
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL

    last_error: Exception | None = None
    for name in MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(name)
        except Exception as err:  # pragma: no cover - only hit if model missing
            last_error = err
            continue
        _MODEL = model
        _MODEL_NAME = name
        return model

    raise RuntimeError(f"Unable to load any embedding model from {MODEL_CANDIDATES}") from last_error


def embed_texts(texts: Iterable[str], normalize: bool = True) -> np.ndarray:
    model = _load_model()
    embeddings = model.encode(
        list(texts),
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_passages(file_path: Path | str = PASSAGES_PATH) -> Tuple[pd.DataFrame, np.ndarray]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing passages csv at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Passages csv has no rows to embed")

    texts = df["text"].fillna("").tolist()
    embeddings = embed_texts(texts, normalize=True)
    return df, embeddings
