from datasets import load_dataset
from pathlib import Path
import pandas as pd

DATASET_ID = "lemousehunter/SnP500-annual-and-sustainability-reports"

print(f"▶ Loading {DATASET_ID} …")
ds = load_dataset(DATASET_ID)
df = ds["train"].to_pandas()
print("Columns found:", list(df.columns))  # ['ticker','year','report_type','chunk_list']
print("Row count:", len(df))

def join_chunks(x):
    if isinstance(x, list):
        return " ".join(s for s in x if isinstance(s, str))
    return str(x) if x is not None else ""

text_series = df["chunk_list"].apply(join_chunks)

out = pd.DataFrame({
    "company":      df["ticker"].astype(str).str.upper() if "ticker" in df else "",
    "ticker":       df["ticker"].astype(str).str.upper() if "ticker" in df else "",
    "title":        df["report_type"].astype(str) if "report_type" in df else "Sustainability Report",
    "url":          "",  # not provided in this dataset
    "date":         df["year"].astype(str) if "year" in df else "",
    "text":         text_series,
    "company_name": df["ticker"].astype(str).str.upper() if "ticker" in df else "",
})

# keep rows with non-empty text
before = len(out)
out = out[out["text"].str.strip().str.len() > 0].reset_index(drop=True)
after = len(out)
print(f"Kept {after} / {before} rows with non-empty text.")

# (Optional) cap for a quick test while developing
# out = out.head(500)

Path("data/raw").mkdir(parents=True, exist_ok=True)
dest = Path("data/raw/esg_news.csv")
out.to_csv(dest, index=False)
print(f"✅ Saved {len(out):,} rows to {dest}")
