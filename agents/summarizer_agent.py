"""Summarizer agent that turns retrieved passages into a structured ESG brief."""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from config import HF_MODEL, HF_TOKEN, HF_API_URL

try:  # pragma: no cover - handled at runtime when dependency missing
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - CLI environments without HF SDK
    InferenceClient = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "environment": ("emission", "carbon", "renewable", "climate", "energy", "net-zero"),
    "social": ("community", "labor", "diversity", "employee", "safety", "equity"),
    "governance": ("board", "compliance", "audit", "governance", "ethics", "transparency"),
}


@dataclass(slots=True)
class SummaryOutput:
    company: str
    environment: List[str]
    social: List[str]
    governance: List[str]
    key_risks: List[str]
    sources: List[str]
    summary_text: str
    raw_model_output: str | None
    used_fallback: bool

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "environment": self.environment,
            "social": self.social,
            "governance": self.governance,
            "key_risks": self.key_risks,
            "sources": self.sources,
            "summary_text": self.summary_text,
            "raw_model_output": self.raw_model_output,
            "used_fallback": self.used_fallback,
        }


def _shorten(text: str, limit: int = 320) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class SummarizerAgent:
    """LLM-backed ESG summarizer with deterministic fallback."""

    def __init__(
        self,
        retrieval: Mapping[str, object] | Sequence[Mapping[str, object]],
        *,
        company: str | None = None,
        llm_client: InferenceClient | None = None,
        max_new_tokens: int = 700,
        temperature: float = 0.25,
    ) -> None:
        if isinstance(retrieval, Mapping) and "documents" in retrieval:
            documents = retrieval.get("documents") or []
            self.company = company or str(retrieval.get("company") or "").upper()
        else:
            documents = retrieval  # type: ignore[assignment]
            self.company = (company or "").upper()
        self.documents: List[Mapping[str, object]] = list(documents or [])
        if not self.company and self.documents:
            self.company = str(self.documents[0].get("company") or "").upper()
        if not self.company:
            raise ValueError("company ticker required for summarization")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = llm_client

    # ------------------------------------------------------------------
    # Prompt helpers
    def _context_snippets(self, limit: int = 5) -> str:
        lines: List[str] = []
        for idx, doc in enumerate(self.documents[:limit], start=1):
            snippet = _shorten(str(doc.get("text", "")))
            meta = {
                "chunk_id": doc.get("chunk_id"),
                "published_at": doc.get("published_at"),
                "source_type": doc.get("source_type"),
                "score": doc.get("score"),
            }
            lines.append(f"[{idx}] {meta}: {snippet}")
        return "\n".join(lines)

    def _build_prompt(self) -> str:
        context = self._context_snippets()
        template = f"""
        You are an independent ESG research analyst. Write a concise ESG summary for {self.company} using the
        supplied evidence. Capture concrete observations for Environment, Social, and Governance along with the
        most material risks. Paraphrase the evidence rather than copying text. Each bullet must cite which source
        index (e.g., [1]) informed it.

        CONTEXT\n{context}

        Respond with STRICT JSON in the following shape:
        {{
            "company": "{self.company}",
            "environment": ["bullet [index]"],
            "social": ["bullet [index]"],
            "governance": ["bullet [index]"],
            "key_risks": ["risk [index]"]
        }}
        """
        return textwrap.dedent(template).strip()

    # ------------------------------------------------------------------
    # LLM interaction
    def _ensure_client(self) -> InferenceClient | None:
        if self._client is not None:
            return self._client
        if InferenceClient is None or not HF_TOKEN:
            logger.warning("HF client unavailable or HF_TOKEN not set; using fallback summary")
            return None
        if HF_API_URL:
            self._client = InferenceClient(base_url=HF_API_URL, token=HF_TOKEN)
        else:
            self._client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
        return self._client

    def _call_llm(self, prompt: str) -> str | None:
        client = self._ensure_client()
        if client is None:
            return None

        # Build chat-style messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent ESG research analyst. "
                    "Respond ONLY with valid JSON matching the required schema."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Try chat-completion (the only task novita supports)
        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
            )
        except AttributeError:
            # Installed huggingface_hub is too old
            logger.error("Your huggingface_hub version does not support chat_completion; using fallback.")
            return None
        except Exception as exc:
            logger.error(f"HF chat_completion failed: {exc}")
            return None

        # Extract content
        try:
            return response.choices[0].message["content"]
        except Exception:
            logger.error(f"Unexpected chat_completion response format: {response}")
            return None

    # ------------------------------------------------------------------
    # Post-processing helpers
    def _try_parse_json(self, raw: str | None) -> MutableMapping[str, object] | None:
        if not raw:
            return None
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _keyword_summary(self, key: str) -> List[str]:
        keywords = CATEGORY_KEYWORDS.get(key, ())
        statements: List[str] = []
        for doc in self.documents:
            text = str(doc.get("text", ""))
            lower = text.lower()
            if any(term in lower for term in keywords):
                statements.append(_shorten(text))
            if len(statements) >= 3:
                break
        if not statements and self.documents:
            statements.append(_shorten(str(self.documents[0].get("text", ""))))
        return statements

    def _fallback_summary(self) -> SummaryOutput:
        environment = self._keyword_summary("environment")
        social = self._keyword_summary("social")
        governance = self._keyword_summary("governance")
        risks = environment[:1] + social[:1] + governance[:1]
        summary = " ".join(section[0] for section in (environment, social, governance) if section)
        sources = [str(doc.get("chunk_id")) for doc in self.documents[:5]]
        return SummaryOutput(
            company=self.company,
            environment=environment,
            social=social,
            governance=governance,
            key_risks=risks[:3],
            sources=sources,
            summary_text=_shorten(summary, limit=500),
            raw_model_output=None,
            used_fallback=True,
        )

    def _assemble_output(
        self,
        payload: Mapping[str, object] | None,
        raw_output: str | None,
    ) -> SummaryOutput:
        if not payload:
            return self._fallback_summary()

        environment = [str(item) for item in payload.get("environment", []) if str(item).strip()]
        social = [str(item) for item in payload.get("social", []) if str(item).strip()]
        governance = [str(item) for item in payload.get("governance", []) if str(item).strip()]
        risks = [str(item) for item in payload.get("key_risks", []) if str(item).strip()]
        if not any((environment, social, governance)):
            return self._fallback_summary()

        sections = environment + social + governance
        summary = " ".join(_shorten(text, 200) for text in sections[:6])
        sources = [str(doc.get("chunk_id")) for doc in self.documents[:5] if doc.get("chunk_id")]

        return SummaryOutput(
            company=self.company,
            environment=environment,
            social=social,
            governance=governance,
            key_risks=risks[:5],
            sources=sources,
            summary_text=_shorten(summary, limit=600),
            raw_model_output=raw_output,
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    def run(self) -> SummaryOutput:
        prompt = self._build_prompt()
        raw_output = self._call_llm(prompt)
        parsed = self._try_parse_json(raw_output)
        return self._assemble_output(parsed, raw_output)
