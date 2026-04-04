"""
backend.py — Data Ingestion Engine.

Handles:
  • PDF loading and chunking
  • Codebase / directory loading and chunking (with exclusions)
  • Embedding via local HuggingFace BGE model
  • Persistent storage in ChromaDB
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Callable

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    PDF_CHUNK_SIZE,
    CHROMA_DB_DIR,
    CODE_EXTENSIONS,
    EXCLUDED_FILE_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS  (cached at module level so we only load the model once)
# ═══════════════════════════════════════════════════════════════════════════

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the singleton embedding model, downloading on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT SPLITTERS
# ═══════════════════════════════════════════════════════════════════════════

# Extension → LangChain Language enum mapping for syntax-aware splitting.
_LANGUAGE_MAP: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".ts":    Language.TS,
    ".jsx":   Language.JS,
    ".tsx":   Language.TS,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".c":     Language.C,
    ".h":     Language.CPP,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".cs":    Language.CSHARP,
    ".html":  Language.HTML,
    ".md":    Language.MARKDOWN,
    # 💎 Hardware Design (Verilog / SystemVerilog / VHDL)
    ".v":     Language.CPP,  # Verilog syntax-aware splitting via C++ logic
    ".sv":    Language.CPP,  # SystemVerilog
    ".vhd":   Language.CPP,  # VHDL
}


def _get_splitter(
    extension: str = "",
    chunk_size_override: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Return a RecursiveCharacterTextSplitter, optionally language-aware.

    If the file extension maps to a known language, we use
    ``from_language()`` so chunk boundaries respect syntax (functions,
    classes, blocks) instead of cutting mid-statement.
    """
    size = chunk_size_override or CHUNK_SIZE
    lang = _LANGUAGE_MAP.get(extension.lower())
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=size,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  FILE-LEVEL EXCLUSION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _is_excluded(filepath: str) -> bool:
    """Return True if the file matches any exclusion pattern."""
    name = os.path.basename(filepath)
    full = filepath.replace("\\", "/")
    for pattern in EXCLUDED_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full, pattern):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  PDF LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_chunk_pdf(file_path: str) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    splitter = _get_splitter(chunk_size_override=PDF_CHUNK_SIZE)
    return splitter.split_documents(raw_docs)


def load_and_chunk_pdf_upload(uploaded_file: BinaryIO, filename: str) -> list[Document]:
    """
    Accept a Streamlit UploadedFile, write it to a temp file,
    ingest it, then clean up.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return load_and_chunk_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
#  CODEBASE LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _collect_code_files(directory: str) -> list[str]:
    """
    Walk *directory* and return absolute paths of code files
    whose extension is in CODE_EXTENSIONS and that are NOT excluded.
    Also picks up extensionless files named 'Dockerfile'.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Prune heavy directories early
        dirs[:] = [
            d for d in dirs
            if d not in {"node_modules", "venv", ".venv", "__pycache__",
                         ".git", "chroma_db", ".tox", ".mypy_cache"}
        ]
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()

            # Extensionless special files
            if fname in ("Dockerfile", "Makefile", "Jenkinsfile", ".dockerignore"):
                if not _is_excluded(fpath):
                    paths.append(fpath)
                continue

            if ext in CODE_EXTENSIONS and not _is_excluded(fpath):
                paths.append(fpath)
    return paths


def load_and_chunk_codebase(
    directory: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    Recursively load a code directory, applying language-aware
    chunking per file and attaching source metadata.

    *on_progress(current, total, filename)* is called after each file.
    """
    all_chunks: list[Document] = []
    file_paths = _collect_code_files(directory)
    total = len(file_paths)

    for idx, fpath in enumerate(file_paths):
        if on_progress:
            on_progress(idx + 1, total, os.path.basename(fpath))
        ext = Path(fpath).suffix.lower()
        try:
            loader = TextLoader(fpath, autodetect_encoding=True)
            raw_docs = loader.load()
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        splitter = _get_splitter(ext, chunk_size_override=CODE_CHUNK_SIZE)
        chunks = splitter.split_documents(raw_docs)

        # Enrich metadata so retriever can cite sources
        for chunk in chunks:
            chunk.metadata["source_type"] = "code"
            chunk.metadata["file_extension"] = ext
        all_chunks.extend(chunks)

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
#  CHROMADB INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def _content_hash(doc: Document) -> str:
    """Return a SHA-256 hex digest of a document's page_content."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def ingest_into_chroma(
    documents: list[Document],
    collection_name: str = "default",
) -> tuple[Chroma, int]:
    """
    Embed *documents* and upsert them into a persistent ChromaDB collection,
    skipping duplicates based on content hash.

    Returns (Chroma instance, number of new documents added).
    """
    if not documents:
        raise ValueError("No documents to ingest — check your file/folder path.")

    # Stamp each document with a content hash
    for doc in documents:
        doc.metadata["content_hash"] = _content_hash(doc)

    embedding = get_embedding_model()

    # Try loading existing collection to deduplicate
    existing_db = load_existing_chroma(collection_name)
    if existing_db is not None:
        existing_data = existing_db.get()
        existing_hashes = {
            meta["content_hash"]
            for meta in existing_data.get("metadatas", [])
            if meta and "content_hash" in meta
        }
        new_docs = [d for d in documents if d.metadata["content_hash"] not in existing_hashes]
        if not new_docs:
            return existing_db, 0
        existing_db.add_documents(new_docs)
        return existing_db, len(new_docs)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )
    return db, len(documents)


def load_existing_chroma(collection_name: str = "default") -> Chroma | None:
    """
    Load a previously-persisted ChromaDB collection.
    Returns None if the directory doesn't exist yet.
    """
    if not os.path.isdir(CHROMA_DB_DIR):
        return None

    embedding = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding,
        collection_name=collection_name,
    )
    # Quick sanity check — empty collection means nothing was ingested
    try:
        data = db.get(limit=1)
        count = len(data["ids"])
        if count == 0:
            # Double-check with full count via get()
            count = len(db.get()["ids"])
    except Exception:
        count = 0
    if count == 0:
        return None
    return db


def list_collections() -> list[str]:
    """Return the names of all ChromaDB collections on disk."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return []
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return [c.name for c in client.list_collections()]


def get_collection_info(collection_name: str) -> dict:
    """Return chunk count and list of unique source files for a collection."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return {"count": 0, "sources": []}
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        return {"count": 0, "sources": []}
    count = coll.count()
    data = coll.get(include=["metadatas"])
    sources = set()
    for meta in data.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return {"count": count, "sources": sorted(sources)}


def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection. Returns True if deleted."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return False
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  ASYNC INGESTION (Phase 1a)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncIngestionTask:
    """
    Run ingestion (chunking + embedding) in a background thread
    so the Streamlit UI stays responsive.

    Usage:
        task = AsyncIngestionTask(chunks, collection_name)
        task.start()
        # Poll task.progress, task.is_done, task.result from the UI
    """

    def __init__(self, documents: list[Document], collection_name: str = "default"):
        self.documents = documents
        self.collection_name = collection_name
        self.progress: float = 0.0          # 0.0 to 1.0
        self.status: str = "pending"        # pending | running | done | error
        self.result: tuple | None = None    # (Chroma, added_count) on success
        self.error: str = ""
        self._thread: threading.Thread | None = None

    def start(self):
        """Launch the ingestion in a background thread."""
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.progress = 0.1
            db, added = ingest_into_chroma(self.documents, self.collection_name)
            self.progress = 1.0
            self.result = (db, added)
            self.status = "done"
        except Exception as e:
            self.error = str(e)
            self.status = "error"

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")


def summarize_document_for_pin(file_path: str, max_chars: int = 3000) -> str:
    """
    Context Summarization (Phase 2b): Create a compact summary of
    a large file for pinning instead of the full content.

    Uses extractive summarization (no LLM call):
      - For code: extracts function/class signatures + docstrings
      - For text: extracts first N characters with paragraph boundaries
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""

    if not content:
        return ""

    ext = Path(file_path).suffix.lower()

    # Code files: extract signatures
    if ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"):
        return _extract_code_signatures(content, max_chars)

    # Text/PDF/markup: extract leading paragraphs
    paragraphs = content.split("\n\n")
    summary = ""
    for para in paragraphs:
        if len(summary) + len(para) > max_chars:
            break
        summary += para + "\n\n"
    return summary.strip() if summary else content[:max_chars]


def _extract_code_signatures(content: str, max_chars: int) -> str:
    """Extract function/class definitions and docstrings from code."""
    import re
    lines = content.split("\n")
    signatures = []
    total_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match common definition patterns
        if (stripped.startswith(("def ", "class ", "function ", "func ",
                                 "export ", "public ", "private ", "async def "))
                or re.match(r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", stripped)):
            # Include the signature line
            signatures.append(line)
            total_len += len(line)
            # Include docstring/comment on next line if present
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(('"""', "'''", "//", "/*", "#", "*")):
                    signatures.append(lines[i + 1])
                    total_len += len(lines[i + 1])
            if total_len > max_chars:
                break

    if not signatures:
        return content[:max_chars]

    return "\n".join(signatures)
