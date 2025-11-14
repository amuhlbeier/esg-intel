"""Comparator agent that benchmarks ESG risk scores versus peers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

import pandas as pd

from config import EVAL_DATA_PATH


ESG_COLUMNS = {
    "environment": "Environment Risk Score",
    "social": "Social Risk Score",
    "governance": "Governance Risk Score",
    "overall": "Total ESG Risk score",
}


_BENCHMARK_CACHE: pd.DataFrame | None = None
_BENCHMARK_PATH: Path | None = None


def _load_benchmarks(path: Path | str = EVAL_DATA_PATH) -> pd.DataFrame:
    global _BENCHMARK_CACHE, _BENCHMARK_PATH
    resolved = Path(path)
    if _BENCHMARK_CACHE is not None and resolved == _BENCHMARK_PATH:
        return _BENCHMARK_CACHE
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark csv missing at {resolved}")
    df = pd.read_csv(resolved)
    for column in ESG_COLUMNS.values():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    _BENCHMARK_PATH = resolved
    _BENCHMARK_CACHE = df
    return df


def _calc_percentile(series: pd.Series, symbol: str) -> float | None:
    series = series.dropna()
    if series.empty or symbol not in series.index:
        return None
    ranks = series.rank(method="average", ascending=True)
    denom = max(len(ranks) - 1, 1)
    percentile = 1.0 - ((ranks[symbol] - 1.0) / denom)
    return float(round(percentile * 100.0, 2))


def _verdict(percentile: float | None) -> str:
    if percentile is None:
        return "insufficient data"
    if percentile >= 67:
        return "outperforming peers"
    if percentile <= 33:
        return "lagging peers"
    return "in-line with peers"


@dataclass(slots=True)
class ComparatorOutput:
    company: str
    peers: list[str]
    scores: dict
    notes: str

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "peers": self.peers,
            "scores": self.scores,
            "notes": self.notes,
        }


class ComparatorAgent:
    """Benchmarks a company's ESG risk metrics against peer averages."""

    def __init__(
        self,
        summary: Mapping[str, object],
        peers: Iterable[str] | None = None,
        *,
        benchmark_path: Path | str = EVAL_DATA_PATH,
    ) -> None:
        company = str(summary.get("company") or "").upper()
        if not company:
            raise ValueError("summary must contain a company ticker")
        self.company = company
        self.peers = [symbol.upper() for symbol in (peers or []) if symbol]
        self.benchmark_path = benchmark_path

    def _prepare_frame(self) -> pd.DataFrame:
        df = _load_benchmarks(self.benchmark_path)
        symbols = {self.company, *self.peers}
        filtered = df[df["Symbol"].isin(symbols)].copy()
        filtered = filtered.dropna(subset=[ESG_COLUMNS["overall"]], how="all")
        filtered.set_index("Symbol", inplace=True, drop=False)
        return filtered

    def _score_dimension(self, df: pd.DataFrame, column: str) -> dict:
        if column not in df.columns or df.empty or self.company not in df.index:
            return {
                "value": None,
                "direction": "lower_is_better",
                "percentile": None,
                "peer_average": None,
                "peer_min": None,
                "peer_max": None,
                "verdict": "insufficient data",
            }

        scoped = df.loc[:, column].dropna()
        raw_value = scoped.get(self.company) if self.company in scoped else None
        value = None if raw_value is None or pd.isna(raw_value) else float(round(raw_value, 2))
        percentile = _calc_percentile(scoped, self.company)
        return {
            "value": value,
            "direction": "lower_is_better",
            "percentile": percentile,
            "peer_average": float(round(scoped.mean(), 2)) if not scoped.empty else None,
            "peer_min": float(round(scoped.min(), 2)) if not scoped.empty else None,
            "peer_max": float(round(scoped.max(), 2)) if not scoped.empty else None,
            "verdict": _verdict(percentile),
        }

    def run(self) -> ComparatorOutput:
        df = self._prepare_frame()
        if df.empty or self.company not in df.index:
            note = "No benchmark data available for the requested ticker"
            return ComparatorOutput(self.company, self.peers, scores={}, notes=note)

        scores = {
            name: self._score_dimension(df, column)
            for name, column in ESG_COLUMNS.items()
        }

        available_percentiles = [s["percentile"] for s in scores.values() if s["percentile"] is not None]
        overall_percentile = round(mean(available_percentiles), 2) if available_percentiles else None
        scores["overall"]["composite_percentile"] = overall_percentile

        missing_peer_symbols = [symbol for symbol in self.peers if symbol not in df.index]
        note_parts = []
        if missing_peer_symbols:
            note_parts.append(f"missing peers: {', '.join(missing_peer_symbols)}")
        note = "; ".join(note_parts) if note_parts else "peers evaluated"

        return ComparatorOutput(self.company, self.peers, scores=scores, notes=note)
