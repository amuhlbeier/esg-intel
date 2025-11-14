from dotenv import load_dotenv
import os

load_dotenv()  # load values from .env

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_API_URL = os.getenv("HF_API_URL")  # optional direct endpoint override
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

PROJECT_NAME = "AI ESG Intelligence Platform"
API_VERSION = "v1"

# --- RAG ---
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

# --- Eval ---
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "eval/datasets/esg_benchmarks.csv")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
