"""
config.py — Centralized configuration for the RAG Knowledge Base.

All tuneable parameters live here. To switch from free to paid models,
change FREE_MODEL to a premium model string — that's it.
"""

import os
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ── OpenRouter settings ────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# 🔄 CLOUDROUTER MODELS — Categorised by Specialisation (2026 Fleet)
CLOUDROUTER_MODELS: dict[str, dict[str, str]] = {
    "Coding & Logic": {
        "Qwen 3 Coder 480B (SOTA)": "qwen/qwen3-coder:free",
        "Gemini 2.0 Flash (Stable)": "google/gemini-2.0-flash-001",
    },
    "Deep Reasoning": {
        "DeepSeek R1 (Full)": "deepseek/deepseek-r1:free",
        "Qwen 3.6 Plus (Balanced)": "qwen/qwen3.6-plus:free",
    },
    "Speed & Efficiency": {
        "Nvidia Nemotron Nano 9B v2": "nvidia/nemotron-nano-9b-v2:free",
        "Mistral Small 3.1 (24B)": "mistralai/mistral-small-3.1-24b-instruct:free",
    },
    "Vision & Images": {
        "Qwen 2.5 VL (Multimodal)": "qwen/qwen-2.5-vl-7b-instruct:free",
        "Kimi VL (High Reasoning)": "moonshotai/kimi-vl-a3b-thinking:free",
    }
}

# The default model to use on startup
DEFAULT_MODEL: str = "google/gemini-2.0-flash-001"

# 🚀 PROMPT CACHING — Harnessing from Claude Code
# The required beta header for Anthropic's prompt caching feature
ANTHROPIC_CACHE_BETA_HEADER: str = "prompt-caching-2024-07-31"
ENABLE_PROMPT_CACHING: bool = True

# 🎯 CACHE TUNING — Gemini & Anthropic Optimisation
# Gemini 2.0 Flash needs ~1028 tokens to trigger a cache write. 
CACHE_THRESHOLD_TOKENS: int = 1028
# Anthropic supports up to 4 breakpoints.
MAX_CACHE_CHECKPOINTS: int = 4

# ── Ollama (100 % local) ──────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS: list[str] = ["llama3.1", "llama3.2:1b", "qwen2.5:3b"]

LLM_TEMPERATURE: float = 0.0           # deterministic for code analysis
MAX_TOKENS: int = 4096                 # enough room for code generation

# ── Embedding model (runs 100 % locally) ────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"     # ~130 MB download

# ── Chunking parameters ────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000          # characters per chunk (default fallback)
CHUNK_OVERLAP: int = 200        # overlap between chunks
CODE_CHUNK_SIZE: int = 500      # smaller chunks for dense code
PDF_CHUNK_SIZE: int = 1500      # larger chunks for prose documents
ZERO_CHUNK_THRESHOLD: int = 200000 # files smaller than this (chars) aren't chunked

# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_DB_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
RETRIEVER_K: int = 4            # top-k chunks for similarity search
RETRIEVER_FETCH_K: int = 20     # MMR fetches more, then picks diverse top-k

# ── Re-ranker (runs locally, ~80 MB) ──────────────────────────────────────
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N: int = 4         # documents surviving re-ranking

# ── Search signal thresholds ───────────────────────────────────────────────
MIN_PREV_QUERY_LENGTH: int = 15   # min length of previous query to include in search signal
MIN_CURRENT_QUERY_LENGTH: int = 10  # below this, always augment with previous query

# ── Semantic cache ─────────────────────────────────────────────────────────
SEMANTIC_CACHE_THRESHOLD: float = 0.85    # cosine similarity to reuse cached docs
PINNED_RELEVANCE_THRESHOLD: float = 0.40  # min similarity to inject pinned file

# ── Ghost history ──────────────────────────────────────────────────────────
GHOST_HISTORY_WINDOW: int = 8             # recent messages to keep in full
GHOST_HISTORY_MAX: int = 10               # beyond this, truncate with ghost logic
AI_RESPONSE_MAX_CHARS: int = 500          # max chars per AI message in ghost history

# ── Sentinel History Cache ─────────────────────────────────────────────────
SENTINEL_INTERVAL: int = 5                # summarize every N conversation turns
SENTINEL_MAX_TOKENS: int = 500            # max tokens for the state-block summary

# ── Cross-Provider Cache Config ────────────────────────────────────────────
# Provider-specific cache checkpoint limits and minimum token thresholds.
# {provider_pattern: (max_checkpoints, min_tokens_for_cache_write)}
PROVIDER_CACHE_PROFILES: dict[str, tuple[int, int]] = {
    "claude-3":    (4, 1024),    # Anthropic: 4 breakpoints, 1024 min
    "gemini-2.0":  (8, 1028),    # Gemini: more checkpoints, 1028 min
    "deepseek":    (4, 1024),    # DeepSeek: similar to Claude
    "qwen-3":      (4, 1024),    # Qwen: similar to Claude
}

# ── Hybrid Search ──────────────────────────────────────────────────────────
ENABLE_HYBRID_SEARCH: bool = True         # BM25 + vector combined
BM25_WEIGHT: float = 0.3                  # weight for keyword search
VECTOR_WEIGHT: float = 0.7               # weight for semantic search

# ── Supported code extensions ───────────────────────────────────────────────
CODE_EXTENSIONS: list[str] = [
    # Languages
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".go", ".rs", ".cs",
    # Hardware Design (Verilog & SystemVerilog)
    ".v", ".sv", 
    # Web / UI frameworks
    ".vue", ".svelte", ".html", ".css",
    # Config / markup
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    # Scripts & DevOps
    ".sh", ".bash", ".bat", ".ps1",
    # Database
    ".sql",
    # Docker (handled separately for extensionless files like "Dockerfile")
]

# ── Exclusion patterns — files that MUST NOT be ingested ────────────────────
EXCLUDED_FILE_PATTERNS: list[str] = [
    "*-lock.json",          # package-lock.json
    "*.lock",               # yarn.lock, Pipfile.lock, poetry.lock …
    "*.csv",                # data dumps
    "*.log",                # log files
    "*.min.js",             # minified JS
    "*.min.css",            # minified CSS
    "*.map",                # source maps
    "*.svg",                # binary-ish markup
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.webp",
    "*node_modules*",       # never ingest node_modules
    "*venv*",               # virtual-env artefacts
    "*__pycache__*",
    "*.pyc",
    "*chroma_db*",          # our own DB directory
    "*.env",                # secrets
]
