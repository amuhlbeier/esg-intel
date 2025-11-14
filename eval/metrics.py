"""Utility metrics used by the evaluator agent and offline evaluation pipeline."""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def category_coverage(sections: Mapping[str, Sequence[str]]) -> float:
    if not sections:
        return 0.0
    filled = sum(1 for values in sections.values() if values)
    return filled / len(sections)


def balance_score(sections: Mapping[str, Sequence[str]]) -> float:
    lengths = [len(values) for values in sections.values()]
    if not lengths or sum(lengths) == 0:
        return 0.0
    imbalance = (max(lengths) - min(lengths)) / max(sum(lengths), 1)
    return 1.0 - imbalance


def risk_depth(risks: Sequence[str], *, max_items: int = 4) -> float:
    if not risks:
        return 0.0
    return _clamp(len(risks) / max_items)


def evidence_traceability(statements: int, sources: Iterable[str]) -> float:
    unique_sources = {source for source in sources if source}
    if statements <= 0:
        return 0.0
    return _clamp(len(unique_sources) / statements)


def compute_summary_metrics(summary: Mapping[str, object]) -> dict:
    sections = {
        key: [str(item) for item in summary.get(key, []) if str(item).strip()]
        for key in ("environment", "social", "governance")
    }
    key_risks = [str(item) for item in summary.get("key_risks", []) if str(item).strip()]
    sources = summary.get("sources") or summary.get("meta", {}).get("sources", [])
    if isinstance(sources, str):
        source_values = [sources]
    else:
        source_values = list(sources or [])

    total_statements = sum(len(items) for items in sections.values()) + len(key_risks)

    metrics = {
        "category_coverage": round(category_coverage(sections), 3),
        "balance": round(balance_score(sections), 3),
        "risk_depth": round(risk_depth(key_risks), 3),
        "evidence_traceability": round(evidence_traceability(total_statements, source_values), 3),
    }
    return metrics


def rmse(predictions: Sequence[float], truths: Sequence[float]) -> float | None:
    """Root mean squared error between two numeric vectors."""

    if not predictions or not truths:
        return None
    a = np.asarray(predictions, dtype="float64")
    b = np.asarray(truths, dtype="float64")
    length = min(len(a), len(b))
    if length == 0:
        return None
    diff = a[:length] - b[:length]
    return float(np.sqrt(np.mean(diff**2)))


def cosine_similarity(predictions: Sequence[float], truths: Sequence[float]) -> float | None:
    if not predictions or not truths:
        return None
    a = np.asarray(predictions, dtype="float64")
    b = np.asarray(truths, dtype="float64")
    length = min(len(a), len(b))
    if length == 0:
        return None
    a = a[:length]
    b = b[:length]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def precision_recall_f1(
    predicted_labels: Sequence[str],
    true_labels: Sequence[str],
    *,
    positive_label: str,
) -> dict:
    """Compute precision/recall/F1 for a binary condition on categorical labels."""

    if not predicted_labels or not true_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    length = min(len(predicted_labels), len(true_labels))
    preds = [predicted_labels[i] == positive_label for i in range(length)]
    refs = [true_labels[i] == positive_label for i in range(length)]

    tp = sum(1 for i in range(length) if preds[i] and refs[i])
    fp = sum(1 for i in range(length) if preds[i] and not refs[i])
    fn = sum(1 for i in range(length) if not preds[i] and refs[i])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def factual_consistency_score(
    statements: Sequence[str],
    evidence_passages: Sequence[str] | None = None,
) -> float:
    """Estimate factual consistency via citation presence and lexical overlap."""

    if not statements:
        return 0.0

    statements = [str(item) for item in statements if str(item).strip()]
    if not statements:
        return 0.0

    cite_hits = sum(1 for stmt in statements if _CITATION_PATTERN.search(stmt))
    cite_ratio = cite_hits / len(statements)

    overlap_score = 0.0
    if evidence_passages:
        tokenized_evidence = [
            {token for token in re.findall(r"[a-zA-Z]{3,}", passage.lower())}
            for passage in evidence_passages
            if passage
        ]
        tokenized_evidence = [tokens for tokens in tokenized_evidence if tokens]
        if tokenized_evidence:
            totals = []
            for stmt in statements:
                stmt_tokens = {token for token in re.findall(r"[a-zA-Z]{3,}", stmt.lower())}
                if not stmt_tokens:
                    totals.append(0.0)
                    continue
                best = max(
                    len(stmt_tokens & evidence_tokens) / len(stmt_tokens)
                    for evidence_tokens in tokenized_evidence
                )
                totals.append(best)
            overlap_score = sum(totals) / len(totals)

    return round(0.5 * cite_ratio + 0.5 * overlap_score, 3)
