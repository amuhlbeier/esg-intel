"""Auto-generated style SDK for interacting with the ESG Intel FastAPI server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import httpx


class ApiError(Exception):
    """Raised when the ESG Intel API returns an HTTP error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass
class ESGIntelClient:
    """Simple HTTP client for the ESG Intel API."""

    base_url: str = "http://localhost:8000"
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    def _handle(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise ApiError(response.status_code, response.text)
        if response.content:
            return response.json()
        return None

    # ------------------------------------------------------------------
    def health(self) -> dict:
        resp = self._client.get("/health")
        return self._handle(resp)

    def analyze(
        self,
        company: str,
        *,
        query: str | None = None,
        peers: Iterable[str] | None = None,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"company": company}
        if query:
            params["query"] = query
        if peers:
            params["peers"] = list(peers)
        if top_k is not None:
            params["top_k"] = top_k
        if source_type:
            params["source_type"] = source_type
        resp = self._client.get("/api/analyze", params=params)
        return self._handle(resp)

    def run_pipeline(self, payload: Mapping[str, Any]) -> dict:
        resp = self._client.post("/api/pipeline", json=payload)
        return self._handle(resp)

    def evaluate(self, records: list[Mapping[str, Any]]) -> dict:
        resp = self._client.post("/api/evaluate", json={"records": records})
        return self._handle(resp)

    def peers(self, *, industry: str | None = None, limit: int = 10) -> dict:
        params = {"limit": limit}
        if industry:
            params["industry"] = industry
        resp = self._client.get("/api/peers", params=params)
        return self._handle(resp)


def get_client(base_url: str = "http://localhost:8000", timeout: float = 60.0) -> ESGIntelClient:
    return ESGIntelClient(base_url=base_url, timeout=timeout)

