import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
test_e2e_final.py — End-to-End Query Tests (T-E2E) from Section 9 of master_plan_test.md

Runs ALL 5 T-E2E tests against the REAL RAG pipeline using REAL LLMs
(OpenRouter — no mocks, no mock LLM).

Tests:
  T-E2E-1  Single-Turn Query Returns Answer
  T-E2E-2  Follow-Up Intent Detected and Context Reused
  T-E2E-3  Model Switching Clears Cache State
  T-E2E-4  Semantic Cache Hit on Near-Identical Query
  T-E2E-5  Long Conversation Triggers Sentinel at Turn 3

Usage:
    python test_e2e_final.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import traceback

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime
from dataclasses import dataclass, field

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Project imports
import config
import backend
import rag_chain


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    test_id: str
    name: str
    status: str = "NOT_RUN"  # PASS / FAIL / ERROR / SKIP
    duration_s: float = 0.0
    details: str = ""
    error: str = ""
    checks: list[dict] = field(default_factory=list)

    def add_check(self, label: str, passed: bool, info: str = ""):
        self.checks.append({"label": label, "passed": passed, "info": info})

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks) if self.checks else False


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

COLL_NAME = "e2e_test_collection_final"
RATE_LIMIT_DELAY = 15  # seconds between LLM-calling tests to avoid 429s


def _safe_stream_chain(chain, inputs: dict, max_retries: int = 3) -> tuple[dict | None, str]:
    """Stream the chain and return (metadata_dict, full_answer_text).
    
    Retries on rate-limit errors (429) with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            metadata = None
            answer_parts = []
            for chunk in chain.stream(inputs):
                if metadata is None and ("context" in chunk or "intent" in chunk):
                    # This is the metadata chunk (yielded first in the chain)
                    metadata = chunk
                    # If it also has an answer (CACHE_HIT case), capture it
                    if "answer" in chunk:
                        answer_parts.append(chunk["answer"])
                    continue
                if "answer" in chunk:
                    answer_parts.append(chunk["answer"])
            full_answer = "".join(answer_parts)
            return metadata, full_answer
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = (attempt + 1) * 10
                print(f"      ⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    # If all retries exhausted, raise
    raise RuntimeError(f"Rate limited after {max_retries} retries")


def _ingest_test_data() -> object:
    """Ingest realistic test documents into a fresh test collection."""
    # Delete previous test collection
    backend.delete_collection(COLL_NAME)
    time.sleep(0.5)

    # Create realistic test documents from actual codebase content
    docs = [
        Document(
            page_content=(
                "def hybrid_search(db, query, collection_name='default', k=10, "
                "exclude_file=None, filter_extensions=None, query_embedding=None):\n"
                '    """\n'
                "    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF).\n"
                "    If query_embedding is provided, it is reused for the vector search\n"
                "    via similarity_search_by_vector, avoiding a redundant embedding\n"
                "    inference that ChromaDB would otherwise perform internally.\n"
                '    """\n'
                "    if not ENABLE_HYBRID_SEARCH:\n"
                "        return db.similarity_search(query, k=k)\n"
                "    # 1. Vector Search (Semantic)\n"
                "    vector_docs = db.similarity_search(query, k=k*3)\n"
                "    # 2. BM25 Keyword Search (SQLite FTS5)\n"
                "    fts = SQLiteFTS5BM25(collection_name)\n"
                "    bm25_docs = fts.search(query, k=k*3)\n"
                "    # 3. Reciprocal Rank Fusion (RRF)\n"
                "    RRF_K = 60\n"
                "    scores = {}\n"
                "    return rrf_results\n"
            ),
            metadata={"source": "rag_chain.py", "content_hash": "h_hybrid",
                       "file_extension": ".py", "chunk_index": 0}
        ),
        Document(
            page_content=(
                "class SemanticCache:\n"
                '    """Persistent Query/Response cache using ChromaDB.\n'
                '    Bypasses RAG and LLM for repeat queries."""\n'
                "    def __init__(self, collection_name='semantic_cache'):\n"
                "        self.db = Chroma(...)\n"
                "    def lookup(self, query, threshold=0.95):\n"
                "        results = self.db.similarity_search_with_relevance_scores(query, k=1)\n"
                "        if results:\n"
                "            doc, score = results[0]\n"
                "            if score >= threshold:\n"
                "                return doc.metadata.get('answer')\n"
                "        return None\n"
                "    def upsert(self, query, answer):\n"
                "        if self.lookup(query, threshold=0.99): return\n"
                "        self.db.add_texts(texts=[query], metadatas=[{'answer': answer}])\n"
            ),
            metadata={"source": "rag_chain.py", "content_hash": "h_semantic_cache",
                       "file_extension": ".py", "chunk_index": 1}
        ),
        Document(
            page_content=(
                "class LocalReRanker:\n"
                '    """Local Cross-Encoder that re-scores retrieved chunks\n'
                '    to ensure surgical precision before the context is passed to the LLM."""\n'
                "    def __init__(self):\n"
                "        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\n"
                "    def rerank(self, query, documents, top_k):\n"
                "        pairs = [[query, doc.page_content] for doc in documents]\n"
                "        scores = self.model.predict(pairs)\n"
                "        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)\n"
                "        return [doc for score, doc in scored_docs[:top_k]]\n"
            ),
            metadata={"source": "rag_chain.py", "content_hash": "h_reranker",
                       "file_extension": ".py", "chunk_index": 2}
        ),
        Document(
            page_content=(
                "class VectorRouter:\n"
                '    """Zero-latency decision engine using vector similarity to handle\n'
                '    classification and state management without LLM overhead."""\n'
                "    def classify_intent(self, query, history):\n"
                "        '''Classify as NEW topic or FOLLOW-UP using the local 1B model.'''\n"
                "        if not history: return 'NEW'\n"
                "        return 'NEW'\n"
                "    def detect_specialty(self, query):\n"
                "        '''Detect the best specialist for the query: CODE, REASONING, VISION, GENERAL.'''\n"
                "        return 'GENERAL'\n"
            ),
            metadata={"source": "rag_chain.py", "content_hash": "h_router",
                       "file_extension": ".py", "chunk_index": 3}
        ),
        Document(
            page_content=(
                "class SQLiteFTS5BM25:\n"
                '    """On-disk Full Text Search engine replacing RAM-heavy rank_bm25.\n'
                '    Uses SQLite\'s FTS5 extension which is built into standard Python."""\n'
                "    def search(self, query, k=10):\n"
                "        '''Fast keyword search via SQLite FTS5.'''\n"
                "        rows = conn.execute('SELECT ... FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?', (query, k))\n"
                "        return [Document(page_content=content, metadata=json.loads(meta)) for content, meta in rows]\n"
            ),
            metadata={"source": "backend.py", "content_hash": "h_fts5",
                       "file_extension": ".py", "chunk_index": 0}
        ),
        Document(
            page_content=(
                "def compress_chat_history(history, sentinel_state):\n"
                "    '''Intelligently trim the chat history based on Sentinel Summaries\n"
                "    or Ghost History logic for token budget preservation.'''\n"
                "    if sentinel_state:\n"
                "        truncated_history = history[-4:]\n"
                "    elif len(history) <= GHOST_HISTORY_MAX:\n"
                "        truncated_history = history\n"
                "    else:\n"
                "        # ghost logic...\n"
                "        truncated_history = anchor + ghosts + window\n"
                "    while _est_tokens(truncated_history) > MAX_HISTORY_TOKENS:\n"
                "        truncated_history.pop(2)\n"
                "    return truncated_history\n"
            ),
            metadata={"source": "rag_chain.py", "content_hash": "h_compress",
                       "file_extension": ".py", "chunk_index": 4}
        ),
    ]

    db, added = backend.ingest_into_chroma(docs, COLL_NAME)
    print(f"  Ingested {added} chunks into collection '{COLL_NAME}'")
    return db


# ═══════════════════════════════════════════════════════════════════════════
#  T-E2E-1: Single-Turn Query Returns Answer
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_1_single_turn(db) -> TestResult:
    """T-E2E-1: A single query with no history returns a substantive answer."""
    r = TestResult("T-E2E-1", "Single-Turn Query Returns Answer")
    t0 = time.time()
    try:
        chain = rag_chain.build_rag_chain(db, model=config.DEFAULT_MODEL)

        inputs = {
            "input": "What does the hybrid_search function do?",
            "chat_history": [],
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": [],
            "last_query": "",
            "last_query_embedding": None,
            "force_retrieval": True,   # Force fresh retrieval
            "collection_name": COLL_NAME,
            "auto_specialist": False,
        }

        metadata, full_answer = _safe_stream_chain(chain, inputs)

        # Check 1: Got an answer
        r.add_check("Answer non-empty", len(full_answer) > 0,
                     f"Answer length: {len(full_answer)} chars")

        # Check 2: Answer is substantive (>50 chars)
        r.add_check("Answer > 50 chars", len(full_answer) > 50,
                     f"Full answer preview: {full_answer[:200]}")

        # Check 3: Answer mentions retrieval/search concepts
        answer_lower = full_answer.lower()
        relevant_keywords = ['retrieval', 'search', 'bm25', 'vector', 'hybrid', 'rrf', 'rank', 'fusion']
        found_keywords = [w for w in relevant_keywords if w in answer_lower]
        r.add_check("Answer mentions retrieval/search concepts", len(found_keywords) > 0,
                     f"Keywords found: {found_keywords}")

        # Check 4: Metadata has intent = NEW (no history)
        if metadata:
            r.add_check("Intent is NEW", metadata.get("intent") == "NEW",
                         f"Intent: {metadata.get('intent')}")
            r.add_check("Context docs retrieved", len(metadata.get("context", [])) > 0,
                         f"Docs retrieved: {len(metadata.get('context', []))}")
        else:
            r.add_check("Metadata received", False, "No metadata chunk received")

        r.status = "PASS" if r.all_passed else "FAIL"
        r.details = f"Answer ({len(full_answer)} chars): {full_answer[:300]}"

    except Exception as e:
        r.status = "ERROR"
        r.error = f"{type(e).__name__}: {e}"
    r.duration_s = time.time() - t0
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  T-E2E-2: Follow-Up Intent Detected and Context Reused
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_2_followup(db) -> TestResult:
    """T-E2E-2: A follow-up question detects FOLLOW-UP intent and reuses context."""
    r = TestResult("T-E2E-2", "Follow-Up Intent Detected and Context Reused")
    t0 = time.time()
    try:
        chain = rag_chain.build_rag_chain(db, model=config.DEFAULT_MODEL)

        # Turn 1: Ask about hybrid search
        inputs_t1 = {
            "input": "What does the hybrid_search function do?",
            "chat_history": [],
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": [],
            "last_query": "",
            "last_query_embedding": None,
            "force_retrieval": True,
            "collection_name": COLL_NAME,
            "auto_specialist": False,
        }
        meta1, answer1 = _safe_stream_chain(chain, inputs_t1)
        turn1_docs = meta1.get("context", []) if meta1 else []
        turn1_embedding = meta1.get("query_embedding") if meta1 else None

        r.add_check("Turn 1: Got answer", len(answer1) > 0, f"Turn 1 length: {len(answer1)}")

        # Delay before Turn 2 to avoid rate limits
        time.sleep(RATE_LIMIT_DELAY)

        # Turn 2: Follow-up question
        history = [
            HumanMessage(content="What does the hybrid_search function do?"),
            AIMessage(content=answer1[:500] if len(answer1) > 500 else answer1),
        ]
        inputs_t2 = {
            "input": "How is it different from vector-only search?",
            "chat_history": history,
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": turn1_docs,
            "last_query": "What does the hybrid_search function do?",
            "last_query_embedding": turn1_embedding,
            "force_retrieval": True,
            "collection_name": COLL_NAME,
            "auto_specialist": False,
        }
        meta2, answer2 = _safe_stream_chain(chain, inputs_t2)

        r.add_check("Turn 2: Got answer", len(answer2) > 0, f"Turn 2 length: {len(answer2)}")

        if meta2:
            intent = meta2.get("intent", "")
            # Note: classify_intent uses local Ollama. If unavailable, defaults to "NEW".
            # We check for FOLLOW-UP but report gracefully if Ollama is down.
            is_followup = intent == "FOLLOW-UP"
            r.add_check(
                "Turn 2: Intent is FOLLOW-UP",
                is_followup,
                f"Intent: {intent} (Ollama required for FOLLOW-UP detection)"
            )
            # Even if intent is NEW due to Ollama being down, context docs should exist
            r.add_check("Turn 2: Context docs present",
                         len(meta2.get("context", [])) > 0,
                         f"Turn 2 docs: {len(meta2.get('context', []))}")
        else:
            r.add_check("Turn 2: Metadata received", False, "No metadata chunk received")

        r.status = "PASS" if r.all_passed else "FAIL"
        r.details = f"Turn 1: {len(answer1)} chars, Turn 2: {len(answer2)} chars"

    except Exception as e:
        r.status = "ERROR"
        r.error = f"{type(e).__name__}: {e}"
    r.duration_s = time.time() - t0
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  T-E2E-3: Model Switching Clears Cache State
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_3_model_switch(db) -> TestResult:
    """T-E2E-3: Different model selections produce distinct RAG chain instances."""
    r = TestResult("T-E2E-3", "Model Switching Clears Cache State")
    t0 = time.time()
    try:
        model_gemma = "google/gemma-4-31b-it:free"
        model_qwen = "qwen/qwen3-coder:free"

        chain_gemma = rag_chain.build_rag_chain(db, model=model_gemma)
        chain_qwen = rag_chain.build_rag_chain(db, model=model_qwen)

        # Check 1: Chains are distinct objects
        r.add_check("Chains are distinct objects",
                     chain_gemma is not chain_qwen,
                     f"gemma id={id(chain_gemma)}, qwen id={id(chain_qwen)}")

        # Check 2: Cache capability differs correctly
        gemma_capable = rag_chain.is_cache_capable(model_gemma)
        qwen_capable = rag_chain.is_cache_capable(model_qwen)
        r.add_check("Gemma is cache-capable",
                     gemma_capable is True,
                     f"is_cache_capable(gemma)={gemma_capable}")
        r.add_check("Qwen is NOT cache-capable",
                     qwen_capable is False,
                     f"is_cache_capable(qwen)={qwen_capable}")

        # Check 3: Cache profile difference
        gemma_profile = rag_chain.get_cache_profile(model_gemma)
        qwen_profile = rag_chain.get_cache_profile(model_qwen)
        r.add_check("Cache profiles match expected values",
                     gemma_profile == (8, 1028) and qwen_profile == (4, 1024),
                     f"Gemma: {gemma_profile}, Qwen: {qwen_profile}")

        # Check 4: Gemma chain produces answer via real LLM
        test_input = {
            "input": "What is BM25?",
            "chat_history": [],
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": [],
            "last_query": "",
            "last_query_embedding": None,
            "force_retrieval": True,
            "collection_name": COLL_NAME,
            "auto_specialist": False,
        }
        meta_g, answer_g = _safe_stream_chain(chain_gemma, test_input)
        r.add_check("Gemma chain produces answer",
                     len(answer_g) > 0,
                     f"Gemma answer: {len(answer_g)} chars")

        # Delay before second model call
        time.sleep(RATE_LIMIT_DELAY)

        # Check 5: Qwen chain produces answer via real LLM
        meta_q, answer_q = _safe_stream_chain(chain_qwen, test_input)
        r.add_check("Qwen chain produces answer",
                     len(answer_q) > 0,
                     f"Qwen answer: {len(answer_q)} chars")

        r.status = "PASS" if r.all_passed else "FAIL"
        r.details = f"Gemma: {len(answer_g)} chars, Qwen: {len(answer_q)} chars"

    except Exception as e:
        r.status = "ERROR"
        r.error = f"{type(e).__name__}: {e}"
    r.duration_s = time.time() - t0
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  T-E2E-4: Semantic Cache Hit on Near-Identical Query
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_4_semantic_cache(db) -> TestResult:
    """T-E2E-4: A near-identical query returns CACHE_HIT."""
    r = TestResult("T-E2E-4", "Semantic Cache Hit on Near-Identical Query")
    t0 = time.time()
    try:
        chain = rag_chain.build_rag_chain(db, model=config.DEFAULT_MODEL)

        # Seed the semantic cache with a known answer
        sem_cache = rag_chain.get_semantic_cache()
        seed_query = "What is BM25 search?"
        seed_answer = (
            "BM25 is a probabilistic ranking function used in information retrieval. "
            "It scores documents based on term frequency and inverse document frequency, "
            "adjusted for document length normalization. In this system, it is implemented "
            "via SQLite FTS5 for efficient on-disk keyword matching."
        )
        sem_cache.upsert(seed_query, seed_answer)
        time.sleep(1.0)  # Allow ChromaDB to persist

        # Now query with a near-identical phrasing
        inputs = {
            "input": "What is BM25 search?",
            "chat_history": [],
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": [],
            "last_query": "",
            "last_query_embedding": None,
            "force_retrieval": False,   # Allow cache hit
            "collection_name": COLL_NAME,
            "auto_specialist": False,
        }

        # For CACHE_HIT, the chain yields {"answer": cached_ans, "intent": "CACHE_HIT"}
        # as a single chunk then returns immediately (no LLM call).
        all_chunks = list(chain.stream(inputs))

        intents = [c.get("intent") for c in all_chunks if c.get("intent")]
        answers = [c.get("answer", "") for c in all_chunks if c.get("answer")]
        full_answer = "".join(answers)

        r.add_check("Intent is CACHE_HIT",
                     "CACHE_HIT" in intents,
                     f"Intents found: {intents}")

        r.add_check("Cached answer returned",
                     len(full_answer) > 0,
                     f"Answer length: {len(full_answer)}")

        r.add_check("Cached answer matches seed",
                     seed_answer in full_answer or full_answer in seed_answer,
                     f"Match: seed_len={len(seed_answer)}, got_len={len(full_answer)}")

        r.status = "PASS" if r.all_passed else "FAIL"
        r.details = f"Cache hit with answer: {full_answer[:200]}"

    except Exception as e:
        r.status = "ERROR"
        r.error = f"{type(e).__name__}: {e}"
    r.duration_s = time.time() - t0
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  T-E2E-5: Long Conversation Triggers Sentinel at Turn 3
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_5_sentinel(db) -> TestResult:
    """T-E2E-5: A long history (>1500 tokens) triggers sentinel at >= SENTINEL_INTERVAL turns."""
    r = TestResult("T-E2E-5", "Long Conversation Triggers Sentinel at Turn 3")
    t0 = time.time()
    try:
        chain = rag_chain.build_rag_chain(db, model=config.DEFAULT_MODEL)

        # Build a history that exceeds SENTINEL_TOKEN_THRESHOLD (1500 tokens)
        # 1500 tokens ~ 4500 chars (at 3 chars/token)
        heavy_content = "word " * 2300  # ~11500 chars ~ 3833 tokens

        # Build 3+ human turns to meet SENTINEL_INTERVAL=3
        history = [
            HumanMessage(content=heavy_content), AIMessage(content="OK, understood."),
            HumanMessage(content="Tell me more."),   AIMessage(content="Sure, here you go."),
            HumanMessage(content="And another topic."), AIMessage(content="Of course."),
        ]

        # Verify the history exceeds the token threshold
        est_tokens = rag_chain._est_tokens(history)
        r.add_check(
            f"History tokens ({est_tokens}) >= threshold ({config.SENTINEL_TOKEN_THRESHOLD})",
            est_tokens >= config.SENTINEL_TOKEN_THRESHOLD,
            f"Estimated tokens: {est_tokens}"
        )

        # Verify turn count >= SENTINEL_INTERVAL
        turn_count = sum(1 for m in history if isinstance(m, HumanMessage))
        r.add_check(
            f"Turn count ({turn_count}) >= SENTINEL_INTERVAL ({config.SENTINEL_INTERVAL})",
            turn_count >= config.SENTINEL_INTERVAL,
            f"Turns: {turn_count}"
        )

        inputs = {
            "input": "Continue explaining the architecture",
            "chat_history": history,
            "full_source_context": "None pinned.",
            "sentinel_state": "",
            "cached_docs": [],
            "last_query": "",
            "last_query_embedding": None,
            "force_retrieval": True,
            "collection_name": COLL_NAME,
            "auto_specialist": False,
            "sentinel_future_active": False,
        }

        all_chunks = list(chain.stream(inputs))

        # Find the metadata chunk (first one with 'context' key)
        meta = None
        for c in all_chunks:
            if "context" in c and "intent" in c:
                meta = c
                break

        if meta:
            sentinel_future = meta.get("sentinel_future")
            r.add_check(
                "Sentinel future triggered",
                sentinel_future is not None,
                f"sentinel_future type={type(sentinel_future).__name__}"
            )

            # If sentinel fires, wait for it to complete and verify it returns a string
            if sentinel_future is not None:
                try:
                    result = sentinel_future.result(timeout=15)
                    r.add_check(
                        "Sentinel produced summary",
                        result is not None and len(str(result)) > 0,
                        f"Summary preview: {str(result)[:200]}"
                    )
                except Exception as e:
                    r.add_check("Sentinel result retrieved", False, f"Error: {e}")
        else:
            r.add_check("Metadata received", False, "No metadata chunk received")

        r.status = "PASS" if r.all_passed else "FAIL"

    except Exception as e:
        r.status = "ERROR"
        r.error = f"{type(e).__name__}: {e}"
    r.duration_s = time.time() - t0
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results: list[TestResult], total_time: float) -> str:
    """Generate a Markdown report of the E2E test results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# End-to-End Query Test Report (T-E2E)")
    lines.append(f"**Date:** {timestamp}")
    lines.append(f"**Model:** `{config.DEFAULT_MODEL}`")
    lines.append(f"**Collection:** `{COLL_NAME}`")
    lines.append(f"**Total Runtime:** {total_time:.1f}s")
    lines.append("")

    # Summary table
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")
    skipped = sum(1 for r in results if r.status == "SKIP")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✅ Passed | {passed} |")
    lines.append(f"| ❌ Failed | {failed} |")
    lines.append(f"| 💥 Errors | {errors} |")
    lines.append(f"| ⏭️ Skipped | {skipped} |")
    lines.append(f"| **Total** | **{len(results)}** |")
    lines.append("")

    # Results table
    lines.append("## Results Overview")
    lines.append("")
    lines.append("| Test ID | Name | Status | Duration |")
    lines.append("|---------|------|--------|----------|")
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️", "NOT_RUN": "⬜"}.get(r.status, "⬜")
        lines.append(f"| {r.test_id} | {r.name} | {icon} {r.status} | {r.duration_s:.1f}s |")
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️"}.get(r.status, "⬜")
        lines.append(f"### {icon} {r.test_id} — {r.name}")
        lines.append(f"**Status:** {r.status} | **Duration:** {r.duration_s:.1f}s")
        lines.append("")

        if r.checks:
            lines.append("| # | Check | Result | Info |")
            lines.append("|---|-------|--------|------|")
            for i, c in enumerate(r.checks, 1):
                check_icon = "✅" if c["passed"] else "❌"
                info = c["info"].replace("|", "\\|").replace("\n", " ")[:150]
                lines.append(f"| {i} | {c['label']} | {check_icon} | {info} |")
            lines.append("")

        if r.details:
            lines.append("> **Details:** " + r.details[:500].replace("\n", " "))
            lines.append("")

        if r.error:
            lines.append("> [!CAUTION]")
            lines.append(f"> **Error:** `{r.error[:500]}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Config snapshot
    lines.append("## Configuration Snapshot")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| DEFAULT_MODEL | `{config.DEFAULT_MODEL}` |")
    lines.append(f"| SENTINEL_INTERVAL | `{config.SENTINEL_INTERVAL}` |")
    lines.append(f"| SENTINEL_TOKEN_THRESHOLD | `{config.SENTINEL_TOKEN_THRESHOLD}` |")
    lines.append(f"| RERANK_CANDIDATES | `{config.RERANK_CANDIDATES}` |")
    lines.append(f"| MAX_ZERO_CHUNK_CHARS | `{config.MAX_ZERO_CHUNK_CHARS}` |")
    lines.append(f"| SEMANTIC_CACHE_THRESHOLD | `{config.SEMANTIC_CACHE_THRESHOLD}` |")
    lines.append(f"| ENABLE_HYBRID_SEARCH | `{config.ENABLE_HYBRID_SEARCH}` |")
    lines.append(f"| USE_RERANKER | `{config.USE_RERANKER}` |")
    lines.append(f"| MAX_TOKENS | `{config.MAX_TOKENS}` |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  END-TO-END QUERY TESTS (T-E2E) - Section 9")
    print("  Using REAL LLM (no mocks) via OpenRouter")
    print(f"  Model: {config.DEFAULT_MODEL}")
    print(f"  Rate limit delay: {RATE_LIMIT_DELAY}s between LLM calls")
    print("=" * 70)

    total_start = time.time()

    # ── Setup: Ingest test data ──────────────────────────────────
    print("\n[SETUP] Ingesting test collection...")
    try:
        db = _ingest_test_data()
    except Exception as e:
        print(f"FATAL: Could not ingest test data: {e}")
        traceback.print_exc()
        return

    results: list[TestResult] = []

    # ── Run T-E2E-4 first (no LLM call needed — semantic cache only) ──
    print("\n[T-E2E-4] Semantic Cache Hit...")
    r4 = test_e2e_4_semantic_cache(db)
    results.append(r4)
    _print_result(r4)

    # ── Run T-E2E-1 ──────────────────────────────────────────────
    print(f"\n[T-E2E-1] Single-Turn Query Returns Answer...")
    r1 = test_e2e_1_single_turn(db)
    results.append(r1)
    _print_result(r1)
    time.sleep(RATE_LIMIT_DELAY)

    # ── Run T-E2E-2 ──────────────────────────────────────────────
    print(f"\n[T-E2E-2] Follow-Up Intent Detection...")
    r2 = test_e2e_2_followup(db)
    results.append(r2)
    _print_result(r2)
    time.sleep(RATE_LIMIT_DELAY)

    # ── Run T-E2E-3 ──────────────────────────────────────────────
    print(f"\n[T-E2E-3] Model Switching / Cache State...")
    r3 = test_e2e_3_model_switch(db)
    results.append(r3)
    _print_result(r3)
    time.sleep(RATE_LIMIT_DELAY)

    # ── Run T-E2E-5 ──────────────────────────────────────────────
    print(f"\n[T-E2E-5] Sentinel Trigger...")
    r5 = test_e2e_5_sentinel(db)
    results.append(r5)
    _print_result(r5)

    # ── Cleanup ──────────────────────────────────────────────────
    print("\n[CLEANUP] Deleting test collection...")
    backend.delete_collection(COLL_NAME)

    total_time = time.time() - total_start

    # Sort results by test ID for the report
    results.sort(key=lambda r: r.test_id)

    # ── Generate Report ──────────────────────────────────────────
    report = generate_report(results, total_time)
    report_path = os.path.join(PROJECT_ROOT, "e2e_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Summary ──────────────────────────────────────────────────
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed, {errored} errors / {len(results)} total")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Report saved to: {report_path}")
    print("=" * 70)


def _print_result(r: TestResult):
    """Print a single test result to console."""
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "ERROR": "[ERR ]"}.get(r.status, "[????]")
    print(f"   {icon} ({r.duration_s:.1f}s)")
    for c in r.checks:
        mark = "  OK " if c["passed"] else "  FAIL"
        print(f"   {mark}: {c['label']}")
    if r.error:
        print(f"   ERROR: {r.error[:300]}")


if __name__ == "__main__":
    main()
