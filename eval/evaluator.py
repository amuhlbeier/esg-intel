"""Evaluation helpers and CLI for ESG pipeline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from config import EVAL_DATA_PATH
from eval.metrics import (
    compute_summary_metrics,
    cosine_similarity,
    factual_consistency_score,
    precision_recall_f1,
    rmse,
)


SCORE_COLUMNS = {
    "environment": "Environment Risk Score",
    "social": "Social Risk Score",
    "governance": "Governance Risk Score",
    "overall": "Total ESG Risk score",
}
POSITIVE_LABEL = "outperforming peers"

_BENCHMARK_CACHE: pd.DataFrame | None = None


def _load_benchmarks(path: str | Path = EVAL_DATA_PATH) -> pd.DataFrame:
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is None:
        _BENCHMARK_CACHE = pd.read_csv(path)
        _BENCHMARK_CACHE["Symbol"] = _BENCHMARK_CACHE["Symbol"].astype(str).str.upper()
    return _BENCHMARK_CACHE


def _normalize_symbol(value: str | None) -> str:
    return (value or "").strip().upper()


def _score_to_category(score: float | None) -> str | None:
    if score is None:
        return None
    if score < 15:
        return "outperforming peers"
    if score < 30:
        return "in-line with peers"
    return "lagging peers"


def evaluate_summary(summary: Mapping[str, object], retrieval: Mapping[str, object] | None = None) -> dict:
    """Compute coverage + factuality metrics for a structured summary."""

    metrics = compute_summary_metrics(summary)
    statements: list[str] = []
    for key in ("environment", "social", "governance", "key_risks"):
        statements.extend(str(item) for item in summary.get(key, []) if str(item).strip())

    evidence = []
    if retrieval and isinstance(retrieval, Mapping):
        evidence = [
            str(doc.get("text", ""))
            for doc in retrieval.get("documents", [])
            if isinstance(doc, Mapping)
        ]

    factual_consistency = factual_consistency_score(statements, evidence)
    metrics["factual_consistency"] = factual_consistency

    overall = round(sum(metrics.values()) / len(metrics), 3) if metrics else 0.0
    weak_spots = [name for name, value in metrics.items() if value < 0.4]
    notes = (
        "gaps detected: " + ", ".join(sorted(set(weak_spots)))
        if weak_spots
        else "summary covers core ESG dimensions"
    )
    return {
        "overall_score": overall,
        "metrics": metrics,
        "notes": notes,
    }


def evaluate_against_benchmark(
    comparison: Mapping[str, object],
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    df = benchmark_df if benchmark_df is not None else _load_benchmarks()
    company = _normalize_symbol(comparison.get("company") or comparison.get("ticker"))
    if not company:
        return {"error": "missing company ticker"}

    row = df[df["Symbol"] == company]
    if row.empty:
        return {"company": company, "error": "benchmark row not found"}
    row = row.iloc[0]

    predicted_scores = comparison.get("scores", {}) if isinstance(comparison, Mapping) else {}
    numeric_pred: list[float] = []
    numeric_truth: list[float] = []
    label_preds: list[str] = []
    label_truth: list[str] = []

    for pillar, column in SCORE_COLUMNS.items():
        truth_value = row.get(column)
        if pd.isna(truth_value):
            continue
        pred_value = predicted_scores.get(pillar, {}).get("value")
        if pred_value is not None:
            numeric_pred.append(float(pred_value))
            numeric_truth.append(float(truth_value))

        pred_label = predicted_scores.get(pillar, {}).get("verdict")
        label_preds.append((pred_label or "").strip().lower())
        label_truth.append(_score_to_category(float(truth_value)) or "")

    rmse_value = rmse(numeric_pred, numeric_truth)
    cosine = cosine_similarity(numeric_pred, numeric_truth)

    valid_labels = [i for i in range(len(label_truth)) if label_truth[i]]
    label_preds = [label_preds[i] for i in valid_labels]
    label_truth = [label_truth[i] for i in valid_labels]
    accuracy = (
        sum(1 for pred, truth in zip(label_preds, label_truth) if pred == truth) / len(label_truth)
        if label_truth
        else None
    )
    prf = precision_recall_f1(label_preds, label_truth, positive_label=POSITIVE_LABEL) if label_truth else {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    return {
        "company": company,
        "rmse": round(rmse_value, 3) if rmse_value is not None else None,
        "cosine_similarity": round(cosine, 3) if cosine is not None else None,
        "category_accuracy": round(accuracy, 3) if accuracy is not None else None,
        **prf,
        "count": len(numeric_truth),
    }


def evaluate_predictions(records: Sequence[Mapping[str, object]]) -> dict:
    benchmark_df = _load_benchmarks()
    company_reports = []
    for record in records:
        comparison = record.get("comparison", {})
        retrieval = record.get("retrieval", {})
        summary = record.get("summary", {})
        benchmark_metrics = evaluate_against_benchmark(comparison, benchmark_df)
        summary_metrics = evaluate_summary(summary, retrieval)
        company_reports.append(
            {
                "company": benchmark_metrics.get("company") or record.get("company"),
                "benchmark_alignment": benchmark_metrics,
                "summary_quality": summary_metrics,
            }
        )

    aggregate = _aggregate_reports(company_reports)
    return {"companies": company_reports, "aggregate": aggregate}


def _aggregate_reports(reports: Sequence[Mapping[str, object]]) -> dict:
    def _avg(path: str) -> float | None:
        values = [
            report["benchmark_alignment"].get(path)
            for report in reports
            if report.get("benchmark_alignment", {}).get(path) is not None
        ]
        return round(sum(values) / len(values), 3) if values else None

    bench_summary = {
        "rmse": _avg("rmse"),
        "cosine_similarity": _avg("cosine_similarity"),
        "category_accuracy": _avg("category_accuracy"),
        "precision": _avg("precision"),
        "recall": _avg("recall"),
        "f1": _avg("f1"),
    }

    summary_metrics: dict[str, list[float]] = {}
    for report in reports:
        metrics = report.get("summary_quality", {}).get("metrics", {})
        for key, value in metrics.items():
            summary_metrics.setdefault(key, []).append(value)

    summary_aggregate = {
        key: round(sum(values) / len(values), 3)
        for key, values in summary_metrics.items()
        if values
    }

    return {
        "benchmark_alignment": bench_summary,
        "summary_metrics": summary_aggregate,
    }


def _load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file at {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    return json.loads(content)


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Evaluate ESG agent outputs against benchmarks")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a JSON or JSONL file containing pipeline outputs (one per company)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the full evaluation report as JSON",
    )
    args = parser.parse_args()

    predictions = _load_predictions(Path(args.predictions))
    report = evaluate_predictions(predictions)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
