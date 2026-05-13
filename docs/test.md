# IMC Test Plan — Private AI RAG Knowledge Base

> **Integration, Module, Component (IMC) Test Plan**
> Validated against codebase on 2026-04-08 (Updated 19:33)
> Files: `config.py`, `rag_chain.py`, `backend.py`, `app.py`

---

## 1. Unit Tests — Configuration (`config.py`)

### T-CFG-1: Config constants are correct types and ranges
| Check | Expected |
|-------|----------|
| `ZERO_CHUNK_THRESHOLD` | `100000` (int, > 0) |
| `RETRIEVER_K` | `6` (int, > 0) |
| `RERANK_TOP_K` | `6` (int, > 0) |
| `RERANK_CANDIDATES` | `25` (int, >= RERANK_TOP_K) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.85` (float, 0 < x <= 1.0) |
| `BM25_WEIGHT` | `0.5` (float, 0 <= x <= 1.0) |
| `VECTOR_WEIGHT` | `0.5` (float, 0 <= x <= 1.0) |
| `GHOST_HISTORY_WINDOW` | `8` (int, > 0) |
| `GHOST_HISTORY_MAX` | `10` (int, >= GHOST_HISTORY_WINDOW) |
| `SENTINEL_INTERVAL` | `5` (int, > 0) |
| `SENTINEL_TOKEN_THRESHOLD` | `2000` (int, > 0) |
| `SENTINEL_MAX_TOKENS` | `500` (int, > 0) |
| `MAX_HISTORY_TOKENS` | `2000` (int, > 0) |
| `AI_RESPONSE_MAX_CHARS` | `800` (int, > 0) |
| `GHOST_AI_CHARS` | `200` (int, > 0) |
| `DEFAULT_MODEL` | `"qwen/qwen-turbo"` (non-empty str) |
| `PINNED_RELEVANCE_THRESHOLD` | `0.40` (float, 0 < x <= 1.0) |
| `STICKY_PINNED_CONTEXT` | `True` (bool) |
| `MIN_PREV_QUERY_LENGTH` | `15` (int, > 0) |
| `MIN_CURRENT_QUERY_LENGTH` | `10` (int, > 0) |

### T-CFG-2: PROVIDER_CACHE_PROFILES structure
- All keys are lowercase provider prefixes: `"claude"`, `"gemini"`, `"gemma"`, `"deepseek"`, `"qwen"`, `"nemotron"`, `"glm"`, `"gpt-5"`, `"reka"`, `"mistral"`
- All values are `(int, int)` tuples — `(max_checkpoints, min_tokens)`
- Claude: `(4, 1024)`, Gemini: `(8, 1028)`, Gemma: `(8, 1028)`

### T-CFG-3: SPECIALIST_MAPPING completeness
- Must contain all four keys: `"CODE"`, `"REASONING"`, `"VISION"`, `"GENERAL"`
- `CODE` & `REASONING` map to `google/gemma-4-...`
- `VISION` maps to `rekaai/reka-edge`
- `GENERAL` maps to `nvidia/nemotron-3...`

### T-CFG-4: CODE_EXTENSIONS and EXCLUDED_FILE_PATTERNS
- `CODE_EXTENSIONS` includes `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.v`, `.sv`, `.html`, `.sql`
- `EXCLUDED_FILE_PATTERNS` includes `"*.env"`, `"*node_modules*"`, `"*chroma_db*"`, `"*.lock"`
- No overlap: no extension in `CODE_EXTENSIONS` matches any pattern in `EXCLUDED_FILE_PATTERNS`

---

## 2. Unit Tests — Backend (`backend.py`)

### T-BE-1: `get_embedding_model()` thread safety
- **Setup:** Call `get_embedding_model()` from 4 concurrent threads
- **Assert:** Only ONE `HuggingFaceEmbeddings` instance is created (check `_embed_lock` serialization)
- **Assert:** All 4 threads receive the same object (identity check with `is`)

### T-BE-2: `_content_hash` deduplication
- **Input:** Two `Document` objects with identical `page_content` but different `source` metadata
- **Assert:** Both produce the SAME `_content_hash` (hash is content-only, SHA-256 of `page_content`)
- **Input:** Two `Document` objects with different `page_content`
- **Assert:** Different `_content_hash` values

### T-BE-3: `SQLiteFTS5BM25` incremental indexing
- **Setup:** Create `SQLiteFTS5BM25` instance for "test_col"
- **Action:** Add 2 documents
- **Assert:** `hashes` table contains both content hashes
- **Assert:** `docs_fts` table contains 2 rows
- **Action:** Add same 2 documents again
- **Assert:** Row count in `docs_fts` remains 2 (duplicate detection)

### T-BE-4: `SQLiteFTS5BM25` search precision
- **Setup:** Index "The quick brown fox" and "The lazy dog"
- **Action:** Search for "fox"
- **Assert:** Match found; rank indicates relevance
- **Action:** Search for "cat"
- **Assert:** Returns empty list (no crash)

### T-BE-5: Content-hash deduplication on re-ingest
- **Setup:** Ingest a file with 3 chunks into a collection
- **Action:** Re-ingest the same file (unchanged)
- **Assert:** `ingest_into_chroma` returns `0` added documents
- **Assert:** Collection count unchanged

### T-BE-6: Incremental ingestion appends correctly
- **Setup:** Ingest file_A.py (5 chunks) into collection
- **Action:** Ingest file_B.py (3 chunks) into same collection
- **Assert:** Collection count = 8
- **Assert:** Both `source` values appear in metadata
- **Assert:** BM25 index updated (new pickle written to disk)

### T-BE-7: Exclusion patterns respected during ingestion
- **Setup:** Directory with `app.py`, `package-lock.json`, `.env`, `archive/old.md`
- **Assert:** Only `app.py` is ingested; `.env`, `archive`, and `lock` are skipped

### T-BE-8: `CodeASTChunker` — Python logic
- **Input:** Python file with 2 classes and 3 functions
- **Assert:** AST chunker identifies logical boundaries
- **Assert:** Each chunk includes `// File: ... | Class: ...` context header

### T-BE-9: `CodeASTChunker` — HDL Regex Fallback
- **Input:** Verilog file with `module top; ... endmodule`
- **Assert:** `_regex_chunk_hdl` correctly extracts the module block
- **Assert:** Metadata includes `hdl_type = "regex_block"`

### T-BE-10: Large file chunking
- **Input:** File with content > `ZERO_CHUNK_THRESHOLD` (100,000 chars)
- **Assert:** File is split into multiple chunks
- **Assert:** Each chunk has `chunk_index` metadata starting from 0

### T-BE-11: Collection deletion cleanup
- **Setup:** Ingest documents into collection "test_col"
- **Action:** Delete collection "test_col"
- **Assert:** ChromaDB collection no longer exists
- **Assert:** `test_col_fts.db` (SQLite) removed from disk
- **Setup:** Ingest documents into collection "test_col"
- **Action:** Delete collection "test_col"
- **Assert:** ChromaDB collection no longer exists
- **Assert:** `test_col_fts.db` (SQLite) removed from disk

---

## 3. Unit Tests — RAG Chain (`rag_chain.py`)

### T-RC-1: `is_cache_capable` & `get_cache_profile`
| Input | Expected (Capable, Profile) |
|-------|----------|
| `"anthropic/claude-3.5-sonnet"` | `True`, `(4, 1024)` |
| `"google/gemini-2.0-flash"` | `True`, `(8, 1028)` |
| `"google/gemma-4-26b"` | `True`, `(8, 1028)` |
| `"deepseek/deepseek-chat"` | `False` (Implicit), `(4, 1024)` |
| `"qwen/qwen-turbo"` | `False` (Implicit), `(4, 1024)` |
| `"ollama/llama3.1"` | `False`, `(4, 1024)` |

### T-RC-2: `_normalize_query` consistency
| Input | Expected |
|-------|----------|
| `"What is Python?"` | `"what is python"` |
| `"  Hello, World!  "` | `"hello world"` |
| `"test"` | `"test"` |
| `"C++ vs Rust"` | `"c vs rust"` |

### T-RC-3: `_get_pinned_embedding` caching via `lru_cache`
- **Setup:** Call `_get_pinned_embedding("test content prefix")` twice
- **Assert:** `embed_query` is called only ONCE (second call returns cached result)
- **Assert:** `_get_pinned_embedding.cache_info().hits == 1` after second call

### T-RC-4: `calculate_cosine_similarity` correctness
| Input | Expected |
|-------|----------|
| Identical vectors `[1,0,0]`, `[1,0,0]` | `1.0` |
| Orthogonal vectors `[1,0,0]`, `[0,1,0]` | `0.0` |
| Opposite vectors `[1,0,0]`, `[-1,0,0]` | `-1.0` |
| Zero vector `[0,0,0]`, `[1,0,0]` | `0.0` (no division error) |

### T-RC-5: `_sort_docs_deterministically` ordering
- **Setup:** 4 docs: A (source="a.py", chunk=0), B (source="b.py", chunk=0), C (source="a.py", chunk=1), D (new doc, source="c.py")
- **With stable_hashes:** Hashes matching A, B, C
- **Assert:** Order is [A, C, B, D] — established docs sorted by (source, chunk_index), then new docs last
- **Assert:** `_is_new` metadata: A=False, B=False, C=False, D=True

### T-RC-6: Singletons via Module Init
- **Assert:** `get_reranker()` returns same instance on repeated calls
- **Assert:** `get_router()` returns same instance on repeated calls
- **Assert:** `get_semantic_cache()` returns same instance on repeated calls

### T-RC-7: `VectorRouter` classification & detection
| Input | Expected |
|-------|----------|
| `"Write a Python function"` | Specialty: `"CODE"` |
| `"Deep reasoning logic"` | Specialty: `"REASONING"` |
| `"Look at this chart"` | Specialty: `"VISION"` |
| `"Previous code refactor"` | Intent: `"FOLLOW-UP"` (via local 1B model) |

### T-RC-8: `LocalReRanker` Cross-Encoder scoring
- **Setup:** Query "Python" and 3 docs (one about Python, two about Java)
- **Assert:** Python doc ranked #1 with high relevance score (> 0.7)
- **Assert:** Output limited to `top_k` documents

### T-RC-9: `SemanticCache` lookup/upsert
- **Setup:** Upsert "What is RAG?" with answer "Retrieval Augmented Gen"
- **Action:** Lookup "What is RAG?" (identical)
- **Assert:** Returns cached answer
- **Action:** Lookup "Tell me about RAG" (semantically similar)
- **Assert:** Returns cached answer if score >= 0.95

### T-RC-10: Hybrid search RRF fusion (SQLite Backend)
- **Setup:** Vector docs from Chroma + Keyword docs from SQLite
- **Assert:** RRF (k=60) merges results; documents appearing in both rank highest
- **Assert:** `doc_id` collision prevention (uses `content_hash` or content excerpt)

### T-RC-11: Exact-match embedding cache
- **Setup:** `last_query = "test query"`, `last_query_emb = [0.1, 0.2, ...]`
- **Action:** Send `user_input = "test query"` (identical after normalization)
- **Assert:** `current_emb` reuses `last_query_emb` without calling `embed_query`

---

## 4. Unit Tests — Conversation History

### T-HIST-1: `compress_chat_history` with sentinel state
- **Setup:** 20 messages in history, `sentinel_state = "Summary: User asked about X"`
- **Assert:** Returns only last 4 messages (2 user + 2 AI)
- **Assert:** AI messages truncated but code blocks preserved via `_truncate_ai_in_history`
- **Assert:** Total estimated tokens reduced significantly

### T-HIST-2: `_truncate_ai_in_history` with code blocks
- **Setup:** AI message with 2KB of prose and a 500-char code block
- **Assert:** Resulting content contains the code block IN FULL
- **Assert:** Prose is truncated to ~400 chars + "... [prose truncated]"
- **Assert:** If code block is massive, it is also capped with a sentinel comment

### T-HIST-3: `compress_chat_history` short history passthrough
- **Setup:** 6 messages (below `GHOST_HISTORY_MAX=10`), `sentinel_state = None`
- **Assert:** All 6 messages returned (no ghost compression)
- **Assert:** AI messages still truncated to `AI_RESPONSE_MAX_CHARS` (800 chars)

### T-HIST-4: Sentinel cooldown prevents over-summarization
- **Setup:** `_sentinel_cooldown["last_turn"] = 5`, `turn_count = 8`, `SENTINEL_INTERVAL = 5`
- **Assert:** `should_summarize = False` (8 - 5 = 3 < 5)
- **Setup:** `turn_count = 10`
- **Assert:** `should_summarize = True` (10 - 5 = 5 >= 5, assuming token threshold met)

### T-HIST-5: Background sentinel update via `ThreadPoolExecutor`
- **Action:** Call `_background_summarize` from chain
- **Assert:** Summary generated in background thread
- **Assert:** `sentinel_future` in `app.py` receives the result without blocking main chat stream

---

## 5. Integration Tests — Ingestion Pipeline

### T-ING-1: End-to-end single file ingestion
- **Action:** Ingest a `.py` file (< 100KB) into fresh collection
- **Assert:** ChromaDB collection created with correct chunk count
- **Assert:** All chunks have `source`, `file_extension`, `content_hash` metadata
- **Assert:** BM25 index pickle created on disk

### T-ING-2: End-to-end directory ingestion
- **Action:** Ingest a directory with 3 `.py` files and 1 `.env` file
- **Assert:** 3 files ingested (`.env` excluded)
- **Assert:** Total chunk count matches sum of individual file chunks

### T-ING-3: Re-ingestion after file modification
- **Action:** Ingest `file.py`, modify its content, re-ingest
- **Assert:** New chunks added (different content hash)
- **Assert:** Old chunks may persist (no automatic cleanup)

### T-ING-4: `AsyncIngestionTask` Lifecycle
- **Action:** Start `AsyncIngestionTask` on a codebase
- **Assert:** `status` transitions: `pending` -> `running` -> `done`
- **Assert:** `progress` updates linearly from 0.1 towards 1.0
- **Assert:** `current_step` reflects "Collecting codebase", "Embedding chunks", etc.

### T-ING-5: `summarize_document_for_pin` (Phase 2b)
- **Input:** 500-line Python file
- **Assert:** Returns extractive summary < 3000 chars
- **Assert:** Contains function/class signatures (not just raw truncation)

---

## 6. Integration Tests — Retrieval Pipeline

### T-RET-1: Vector-only retrieval (hybrid off)
- **Setup:** `ENABLE_HYBRID_SEARCH = False`, collection with 20 chunks
- **Action:** Query the collection
- **Assert:** Results returned (vector search only)
- **Assert:** Result count <= `RETRIEVER_K` (6)

### T-RET-2: Hybrid retrieval (BM25 + vector)
- **Setup:** `ENABLE_HYBRID_SEARCH = True`, collection with 20 chunks
- **Action:** Query with a keyword-heavy term present in BM25 corpus
- **Assert:** Results include keyword matches that vector-only might miss
- **Assert:** RRF scores computed (both BM25 and vector contribute)

### T-RET-3: Re-ranking narrows and validates results
- **Setup:** `USE_RERANKER = True`, `RERANK_TOP_K = 6`
- **Action:** Query with semantic overlap but distinct keyword mismatch
- **Assert:** Cross-encoder validates results; top score logged to telemetry
- **Assert:** Output contains exactly 6 documents re-ordered by relevance

### T-RET-4: ChromaDB filter — exclude file
- **Action:** Query with `exclude_file = "specific_file.py"`
- **Assert:** No results have `source == "specific_file.py"`

### T-RET-5: ChromaDB filter — file extensions
- **Action:** Query with `filter_extensions = [".py", ".js"]`
- **Assert:** All results have `file_extension` in `[".py", ".js"]`

### T-RET-6: Pinned content injection (sticky mode)
- **Setup:** `STICKY_PINNED_CONTEXT = True`, pinned file set
- **Action:** Query unrelated to pinned file content
- **Assert:** Pinned content still included in context (sticky = always inject)

### T-RET-7: Pinned content gating (non-sticky mode)
- **Setup:** `STICKY_PINNED_CONTEXT = False`, `PINNED_RELEVANCE_THRESHOLD = 0.40`
- **Action:** Query unrelated to pinned file (cosine sim < 0.40)
- **Assert:** Pinned content NOT injected

---

## 7. Integration Tests — Full Chain

### T-CHAIN-1: Auto-Specialist Routing (Phase 4)
- **Setup:** `ENABLE_AUTO_SPECIALIST = True`
- **Action:** Query "Write a Python script"
- **Assert:** `detect_specialty` -> `"CODE"`
- **Assert:** `SPECIALIST_MAPPING["CODE"]` model string used for the chain
- **Action:** Query "Analyze this math problem"
- **Assert:** `detect_specialty` -> `"REASONING"`

### T-CHAIN-2: `detect_force_retrieval` with cached sources
- **Setup:** Collection contains `app.py`
- **Action:** Query "How does app.py handle state?"
- **Assert:** `force_retrieval = True` (matches filename from cached sources)
- **Action:** Query "Reload everything"
- **Assert:** `force_retrieval = True` (matches keyword)

### T-CHAIN-3: Force retrieval override
- **Action:** Send `"reload the documents and tell me about config"`
- **Assert:** `force_retrieval = True` detected
- **Assert:** Retrieval runs regardless of cache state

### T-CHAIN-4: Model switching mid-session
- **Action:** Switch from `"qwen/qwen-turbo"` to `"deepseek/deepseek-r1:free"`
- **Assert:** Chain rebuilt with new model
- **Assert:** Next query uses new model for generation

### T-CHAIN-5: Claude model gets cache_control blocks
- **Setup:** Model = `"anthropic/claude-3.5-sonnet"`
- **Assert:** `is_cache_capable` returns `True`
- **Assert:** System prompt formatted with `cache_control: {"type": "ephemeral"}` blocks
- **Assert:** `anthropic-beta` header includes `ANTHROPIC_CACHE_BETA_HEADER`

### T-CHAIN-6: Non-Claude model gets plain string system prompt
- **Setup:** Model = `"google/gemini-2.0-flash"` or `"qwen/qwen-turbo"`
- **Assert:** `is_cache_capable` returns `False` (Implicit or not supported)
- **Assert:** System prompt is a plain string

### T-CHAIN-7: Zero-chunk document handling (Phase 3)
- **Setup:** Zero-chunk doc (> 10KB) in retrieval results
- **Assert:** Large zero-chunk docs are filtered out for non-cache providers
- **Assert:** Smaller zero-chunk docs (< 10KB) are retained

---

## 8. Integration Tests — Prompt Caching

### T-CACHE-1: Deterministic prompt prefix stability
- **Setup:** Two consecutive queries retrieving overlapping documents
- **Action:** Compare system prompt + established context between turns
- **Assert:** Byte-identical prefix up to the `<new_discoveries>` section
- **Assert:** Established docs sorted deterministically by `(source, chunk_index, content_hash)`

### T-CACHE-2: Provider cache profile lookup
- **Action:** Call with model `"google/gemma-4-..."`
- **Assert:** Profile matched: `"gemma"` -> `(8, 1028)`
- **Action:** Call with model `"nvidia/nemotron-..."`
- **Assert:** Profile matched: `"nemotron"` -> `(4, 1024)`

### T-CACHE-3: `TRUST_NATIVE_CACHE` toggle behavior
- **Setup:** `trust_native_cache = True`, semantic hit, provider has cache
- **Assert:** `skip_retrieval = False` (retrieval forced for prefix stability)
- **Setup:** `trust_native_cache = False`, semantic hit, previous docs exist
- **Assert:** `skip_retrieval = True` (semantic cache skip allowed)

---

## 9. Component Tests — App Layer (`app.py`)

### T-APP-1: `extract_usage_metadata` across providers
| Provider | Input Keys | Expected Output |
|----------|-----------|----------------|
| OpenRouter | `token_usage: {prompt_tokens: 100, completion_tokens: 50}` | `{input: 100, output: 50, total: 150}` |
| Anthropic | Cache headers with `cache_read_input_tokens`, `cache_creation_input_tokens` | `cache_read` and `cache_create` populated |
| Ollama | `prompt_eval_count: 200, eval_count: 100` | `{input: 200, output: 100, total: 300}` |

### T-APP-2: Telemetry & Dashboard Tracking (Phase 5)
- **Setup:** Query completed with 200 input tokens and 100 output tokens
- **Action:** Check `st.session_state.metrics_history`
- **Assert:** Turn data contains `input_tokens`, `output_tokens`, `cached_tokens`
- **Assert:** `specialist_counts` incremented for the model category used

### T-APP-3: Background usage fetching via `_bg_fetch_generation_usage`
- **Action:** Call `_bg_fetch_generation_usage` with generation ID
- **Assert:** Sends GET request to OpenRouter API
- **Assert:** Success result updates `st.session_state.token_usage` and triggers `st.rerun()`

### T-APP-4: `detect_force_retrieval` with source scanning
| Input | Expected |
|-------|----------|
| `"What is in app.py?"` | `True` (if app.py in cached sources) |
| `"Tell me about python"` | `False` (generic) |
| `"reload collections"` | `True` (keyword) |

### T-APP-5: Session state initialization
- **Assert:** `st.session_state.trust_native_cache` defaults to `True`
- **Assert:** `st.session_state.last_query` exists after first query
- **Assert:** Chain inputs dict includes `"trust_native_cache"` and `"last_query"` keys

---

## 10. Edge Cases & Regression Tests

### T-EDGE-1: Empty collection query
- **Setup:** Empty ChromaDB collection (0 documents)
- **Action:** Send a query
- **Assert:** No crash; LLM receives empty context; response indicates no documents found

### T-EDGE-2: Very long query (> 1000 chars)
- **Action:** Send a 2000-character query
- **Assert:** No crash; embedding computed; retrieval runs; response generated

### T-EDGE-3: Unicode/special characters in query
- **Action:** Send query with CJK characters, emojis, code blocks
- **Assert:** `_normalize_query` handles gracefully; no encoding errors

### T-EDGE-4: Concurrent ingestion + query
- **Setup:** Start async ingestion on a collection
- **Action:** Query the same collection while ingestion is in progress
- **Assert:** Query returns results from pre-ingestion state; no crash or corruption

### T-EDGE-5: Model string edge cases
| Input to `is_cache_capable` | Expected |
|------------------------------|----------|
| `"CLAUDE-3.5-SONNET"` (uppercase) | `True` (lowercased check) |
| `"my-claude-fork"` | `True` (contains "claude") |
| `"not-a-real-model"` | `False` |

### T-EDGE-6: Ollama unavailable during query rewrite
- **Setup:** FOLLOW-UP intent, Ollama not running
- **Assert:** `llm_rewrite.invoke()` raises exception
- **Assert:** Exception caught; falls back to original `user_input`
- **Assert:** No latency spike beyond the connection timeout

### T-EDGE-7: Sentinel with Ollama unavailable
- **Setup:** Sentinel trigger conditions met, Ollama not running
- **Assert:** Extractive fallback fires (bullet-point summary from message content)
- **Assert:** `sentinel_state` receives the extractive summary, not `None`

### T-EDGE-8: doc_id uniqueness in RRF
- **Setup:** Two zero-chunk documents from different sources, both with `chunk_index=0`
- **Assert:** `doc_id` tuples are different (if content_hash differs or source differs)
- **Known risk:** If `content_hash` is missing from metadata AND sources are identical, collision occurs

---

## Known Issues (Not Tested — Documented)

| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| BUG-3 | Ollama query rewrite: malformed prompt (Python `repr` of HumanMessage objects), synchronous blocking, no timeout | Open | Keep feature, fix implementation |
| RAG-4 | `doc_id` collision when `chunk_index=0` and `content_hash` missing for same source | Open | Low probability with content hash |
| LOGIC-9 | `detect_force_retrieval` false positive on "refresh" | Documented | Acceptable trade-off |
| LOGIC-1 | Semantic cache skip defeated for all providers when `TRUST_NATIVE_CACHE=True` | By design | Prefix stability prioritized |
| RAG-6 | Zero-chunk > 10KB silently dropped for non-cache models | By design | Prevents context overflow |

---

## Test Execution Notes

- **Environment:** Python 3.12+, requires `langchain`, `chromadb`, `sentence-transformers`, `streamlit`, `rank_bm25`
- **Mocking:** Use `unittest.mock.patch` for external calls (OpenRouter API, Ollama)
- **Fixtures:** Create temp ChromaDB directories for isolation; clean up after each test
- **Threading tests:** Use `threading.Barrier` or `concurrent.futures` to ensure true concurrency
- **Config overrides:** Monkeypatch `config` module attributes for test-specific values
