"""Retriever agent for the ESG Intelligence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from config import TOP_K_RESULTS
from rag.retrieve import retrieve as rag_retrieve


DEFAULT_QUERY_TEMPLATE = "Provide the most recent ESG developments, risks, and opportunities for {company}."


@dataclass(slots=True)
class RetrievalOutput:
    """Structured response produced by the retriever agent."""

    company: str
    query: str
    documents: List[dict]

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "query": self.query,
            "documents": self.documents,
        }


class RetrieverAgent:
    """Lightweight wrapper around the FAISS-powered RAG retriever."""

    def __init__(
        self,
        company: str,
        query: str | None = None,
        *,
        k: int | None = None,
        source_type: str | Iterable[str] | None = None,
    ) -> None:
        if not company:
            raise ValueError("company (ticker) is required for retrieval")
        self.company = company.upper().strip()
        self._query = (query or "").strip()
        self.k = k or TOP_K_RESULTS
        self.source_type = source_type

    def _build_query(self) -> str:
        if self._query:
            return self._query
        return DEFAULT_QUERY_TEMPLATE.format(company=self.company)

    def run(self) -> RetrievalOutput:
        query = self._build_query()
        documents = rag_retrieve(self.company, query, k=self.k, source_type=self.source_type)
        return RetrievalOutput(company=self.company, query=query, documents=documents)
