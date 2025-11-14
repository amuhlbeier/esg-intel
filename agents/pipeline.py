"""Orchestrates the end-to-end ESG agent pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

from agents.comparator_agent import ComparatorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent


@dataclass(slots=True)
class PipelineResult:
    company: str
    retrieval: dict
    summary: dict
    comparison: dict
    evaluation: dict

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "retrieval": self.retrieval,
            "summary": self.summary,
            "comparison": self.comparison,
            "evaluation": self.evaluation,
        }


class AgentPipeline:
    """Runs Retriever → Summarizer → Comparator → Evaluator."""

    def __init__(
        self,
        company: str,
        *,
        query: str | None = None,
        peers: Iterable[str] | None = None,
        top_k: int | None = None,
        source_type: str | None = None,
        llm_client=None,
    ) -> None:
        self.company = company.upper().strip()
        if not self.company:
            raise ValueError("company ticker is required")
        self.query = query
        self.peers = [peer.upper().strip() for peer in (peers or []) if peer]
        self.top_k = top_k
        self.source_type = source_type
        self.llm_client = llm_client

    def run(self) -> PipelineResult:
        retriever = RetrieverAgent(
            self.company,
            query=self.query,
            k=self.top_k,
            source_type=self.source_type,
        )
        retrieval_output = retriever.run()

        summarizer = SummarizerAgent(
            retrieval_output.to_dict(),
            llm_client=self.llm_client,
        )
        summary_output = summarizer.run()

        comparator = ComparatorAgent(summary_output.to_dict(), peers=self.peers)
        comparison_output = comparator.run()

        evaluator = EvaluatorAgent(summary_output.to_dict(), retrieval_output.to_dict())
        evaluation_output = evaluator.run()

        return PipelineResult(
            company=self.company,
            retrieval=retrieval_output.to_dict(),
            summary=summary_output.to_dict(),
            comparison=comparison_output.to_dict(),
            evaluation=evaluation_output.to_dict(),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ESG agent pipeline")
    parser.add_argument("company", help="Ticker symbol to analyze")
    parser.add_argument("--query", help="Optional custom retrieval query")
    parser.add_argument("--peers", help="Comma separated peer tickers")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Override the number of retrieved passages")
    parser.add_argument("--source-type", dest="source_type", help="Filter retrieval to a source type")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    peers = [item.strip() for item in (args.peers or "").split(",") if item.strip()]
    pipeline = AgentPipeline(
        args.company,
        query=args.query,
        peers=peers,
        top_k=args.top_k,
        source_type=args.source_type,
    )
    result = pipeline.run()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
