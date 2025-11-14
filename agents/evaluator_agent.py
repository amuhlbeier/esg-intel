"""Evaluator agent that scores coverage/quality of ESG summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from eval.evaluator import evaluate_summary


@dataclass(slots=True)
class EvaluatorOutput:
    company: str
    evaluation: dict
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "evaluation": self.evaluation,
            "recommendations": self.recommendations,
        }


class EvaluatorAgent:
    """Generates factuality and coverage metrics given a structured summary."""

    def __init__(self, summary: Mapping[str, object], retrieval: Mapping[str, object] | None = None) -> None:
        company = str(summary.get("company") or "").upper()
        if not company:
            raise ValueError("summary must include a company identifier")
        self.company = company
        self.summary = summary
        self.retrieval = retrieval or {}

    def _build_recommendations(self, metrics: Mapping[str, float]) -> list[str]:
        recs: list[str] = []
        if metrics.get("evidence_traceability", 1.0) < 0.5:
            recs.append("Add explicit source references for more statements to improve traceability.")
        if metrics.get("category_coverage", 1.0) < 0.67:
            recs.append("Ensure Environment, Social, and Governance sections each include at least one insight.")
        if metrics.get("balance", 1.0) < 0.5:
            recs.append("Balance the number of bullets across ESG pillars to avoid overweighting a single dimension.")
        if metrics.get("risk_depth", 1.0) < 0.5:
            recs.append("Highlight more concrete risks or controversies with supporting evidence.")
        if metrics.get("factual_consistency", 1.0) < 0.6:
            recs.append("Tighten grounding by keeping statements aligned with the cited evidence snippets.")
        if not recs:
            recs.append("Summary meets baseline completeness and traceability thresholds.")
        return recs

    def run(self) -> EvaluatorOutput:
        evaluation = evaluate_summary(self.summary, self.retrieval)
        recommendations = self._build_recommendations(evaluation.get("metrics", {}))
        return EvaluatorOutput(self.company, evaluation, recommendations)
