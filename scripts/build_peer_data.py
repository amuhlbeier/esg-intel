from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/esg_news.csv")
BENCH_PATH = Path("eval/datasets/esg_benchmarks.csv")
OUTPUT_PATH = Path("dashboard/src/peerData.json")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw ESG CSV at {RAW_PATH}")

    raw_df = pd.read_csv(RAW_PATH, usecols=["ticker"], dtype=str)
    tickers = (
        raw_df["ticker"].dropna().astype(str).str.upper().sort_values().unique().tolist()
    )

    bench_df = pd.read_csv(BENCH_PATH, dtype=str)
    bench_df["Symbol"] = bench_df["Symbol"].astype(str).str.upper()
    industries = (
        bench_df[bench_df["Symbol"].isin(tickers)]["Industry"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    payload = {
        "tickers": tickers,
        "industries": industries,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(tickers)} tickers and {len(industries)} industries to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
