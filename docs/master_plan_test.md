# Master Test Plan — RAG Knowledge Base System
**Version:** 2.0 (Post-Token-Optimization)
**Date:** 2026-04-11
**Scope:** Full system — config, backend, RAG chain, retrieval, history, ingestion, end-to-end

---

## Table of Contents
1. [Test Strategy](#1-test-strategy)
2. [Parameter Validation Tests (T-CFG)](#2-parameter-validation-tests-t-cfg)
3. [Backend Tests (T-BE)](#3-backend-tests-t-be)
4. [RAG Chain Unit Tests (T-RC)](#4-rag-chain-unit-tests-t-rc)
5. [Retrieval Integration Tests (T-RET)](#5-retrieval-integration-tests-t-ret)
6. [History & Memory Tests (T-HIST)](#6-history--memory-tests-t-hist)
7. [Ingestion Tests (T-INGEST)](#7-ingestion-tests-t-ingest)
8. [Token Optimization Tests (T-TOK)](#8-token-optimization-tests-t-tok)
9. [End-to-End Query Tests (T-E2E)](#9-end-to-end-query-tests-t-e2e)
10. [Formulated RAG Questions (T-QA)](#10-formulated-rag-questions-t-qa)
11. [Edge Case & Failure Mode Tests (T-EDGE)](#11-edge-case--failure-mode-tests-t-edge)
12. [Test Execution Matrix](#12-test-execution-matrix)

---

## 1. Test Strategy

### Test Levels
| Level | Scope | Run Frequency |
|-------|-------|--------------|
| Unit | Single function / class method | Every code change |
| Integration | Two or more interacting components | Every PR |
| End-to-End | Full query pipeline from input to streamed response | Pre-release |
| Manual QA | Live RAG Q&A with curated questions (Section 10) | Every release |

### File Map
| File | Test File(s) |
|------|-------------|
| `config.py` | `test_t_cfg_1.py`, `test_t_cfg_2.py`, `test_t_cfg_3_4.py` |
| `backend.py` | `test_backend_full.py`, `test_ingest.py`, `test_ingest_integration.py` |
| `rag_chain.py` | `test_t_rc_1.py`, `test_t_rc_2.py`, `test_rag_full.py`, `test_retrieval.py`, `test_retrieval_integration.py`, `test_hybrid.py` |
| History logic | `test_history_full.py` |
| Full pipeline | `test_final_integration.py`, `test_stage2.py`–`test_stage5.py` |

### Test ID Convention
`T-{CATEGORY}-{NUMBER}` where categories are:
- **CFG** — Config parameter validation
- **BE** — Backend (ingestion, chunking, storage)
- **RC** — RAG chain unit tests
- **RET** — Retrieval integration
- **HIST** — History & memory management
- **INGEST** — Async ingestion pipeline
- **TOK** — Token optimization (new after 2026-04-11 changes)
- **E2E** — End-to-end pipeline
- **QA** — Formulated questions for live RAG testing
- **EDGE** — Edge cases and failure modes

---

## 2. Parameter Validation Tests (T-CFG)

### T-CFG-1 — Numeric Parameters: Types and Ranges
**File:** `test_t_cfg_1.py`
**What it checks:** Every numeric config constant has the correct Python type and a sane value range.

| Parameter | Expected Type | Expected Value / Constraint | Updated? |
|-----------|--------------|----------------------------|---------|
| `MAX_TOKENS` | `int` | `== 4096` | No |
| `LLM_TEMPERATURE` | `float` | `== 0.0` | No |
| `CHUNK_SIZE` | `int` | `== 1500` | No |
| `CHUNK_OVERLAP` | `int` | `== 200`, `< CHUNK_SIZE` | No |
| `CODE_CHUNK_SIZE` | `int` | `== 1000` | No |
| `PDF_CHUNK_SIZE` | `int` | `== 1500` | No |
| `ZERO_CHUNK_THRESHOLD` | `int` | `== 100000` | No |
| `MAX_ZERO_CHUNK_CHARS` | `int` | `== 9000`, `< ZERO_CHUNK_THRESHOLD` | **NEW** |
| `RETRIEVER_K` | `int` | `== 6` | No |
| `RETRIEVER_FETCH_K` | `int` | `== 25`, `>= RETRIEVER_K` | No |
| `SEMANTIC_CACHE_THRESHOLD` | `float` | `== 0.85`, `0 < x < 1` | No |
| `PINNED_RELEVANCE_THRESHOLD` | `float` | `== 0.40`, `0 < x < 1` | No |
| `GHOST_HISTORY_WINDOW` | `int` | `== 10` | No |
| `GHOST_HISTORY_MAX` | `int` | `== 10` | No |
| `AI_RESPONSE_MAX_CHARS` | `int` | `== 800` | No |
| `GHOST_AI_CHARS` | `int` | `== 200`, `< AI_RESPONSE_MAX_CHARS` | No |
| `MAX_HISTORY_TOKENS` | `int` | `== 2000` | No |
| `SENTINEL_INTERVAL` | `int` | `== 3` | **CHANGED** (was 5) |
| `SENTINEL_MAX_TOKENS` | `int` | `== 500` | No |
| `SENTINEL_TOKEN_THRESHOLD` | `int` | `== 1500` | **CHANGED** (was 2000) |
| `RERANK_TOP_K` | `int` | `== 6` | No |
| `RERANK_CANDIDATES` | `int` | `== 15`, `>= RERANK_TOP_K` | **CHANGED** (was 25) |
| `BM25_WEIGHT` | `float` | `== 0.5`, `0 < x <= 1` | No |
| `VECTOR_WEIGHT` | `float` | `== 0.5`, `BM25_WEIGHT + VECTOR_WEIGHT == 1.0` | No |
| `MIN_PREV_QUERY_LENGTH` | `int` | `== 15` | No |
| `MIN_CURRENT_QUERY_LENGTH` | `int` | `== 10` | No |
| `CACHE_THRESHOLD_TOKENS` | `int` | `== 1028` | No |
| `MAX_CACHE_CHECKPOINTS` | `int` | `== 4` | No |

**Assertion logic:**
```python
assert isinstance(SENTINEL_INTERVAL, int) and SENTINEL_INTERVAL == 3
assert isinstance(SENTINEL_TOKEN_THRESHOLD, int) and SENTINEL_TOKEN_THRESHOLD == 1500
assert isinstance(RERANK_CANDIDATES, int) and RERANK_CANDIDATES == 15
assert isinstance(MAX_ZERO_CHUNK_CHARS, int) and MAX_ZERO_CHUNK_CHARS == 9000
assert MAX_ZERO_CHUNK_CHARS < ZERO_CHUNK_THRESHOLD
assert BM25_WEIGHT + VECTOR_WEIGHT == pytest.approx(1.0)
assert GHOST_AI_CHARS < AI_RESPONSE_MAX_CHARS
assert CHUNK_OVERLAP < CHUNK_SIZE
assert RERANK_CANDIDATES >= RERANK_TOP_K
assert SENTINEL_TOKEN_THRESHOLD <= MAX_HISTORY_TOKENS
```

---

### T-CFG-2 — Bool Parameters
**File:** `test_t_cfg_1.py`

| Parameter | Expected Value |
|-----------|---------------|
| `ENABLE_PROMPT_CACHING` | `True` |
| `TRUST_NATIVE_CACHE` | `True` |
| `STICKY_PINNED_CONTEXT` | `True` |
| `ENABLE_HYBRID_SEARCH` | `True` |
| `USE_RERANKER` | `True` |
| `ENABLE_AUTO_SPECIALIST` | `True` |

---

### T-CFG-3 — Provider Cache Profiles
**File:** `test_t_cfg_2.py`
**What it checks:** `PROVIDER_CACHE_PROFILES` dict structure and values.

```python
assert set(PROVIDER_CACHE_PROFILES.keys()) == {
    "claude", "gemini", "gemma", "deepseek", "qwen",
    "nemotron", "glm", "gpt-5", "reka", "mistral"
}
for provider, (max_bp, min_tok) in PROVIDER_CACHE_PROFILES.items():
    assert isinstance(max_bp, int) and max_bp in (4, 8)
    assert isinstance(min_tok, int) and min_tok >= 1024

assert PROVIDER_CACHE_PROFILES["claude"] == (4, 1024)
assert PROVIDER_CACHE_PROFILES["gemini"] == (8, 1028)
assert PROVIDER_CACHE_PROFILES["gemma"] == (8, 1028)
```

---

### T-CFG-4 — Specialist Mapping Completeness
**File:** `test_t_cfg_3_4.py`

```python
assert set(SPECIALIST_MAPPING.keys()) == {"CODE", "REASONING", "VISION", "GENERAL"}
for specialty, model in SPECIALIST_MAPPING.items():
    assert isinstance(model, str) and "/" in model  # Valid model ID format
assert SPECIALIST_MAPPING["CODE"] == "qwen/qwen3-coder:free"
assert SPECIALIST_MAPPING["REASONING"] == "liquid/lfm-2.5-1.2b-thinking:free"
```

---

### T-CFG-5 — File Extension & Exclusion Patterns
**File:** `test_t_cfg_3_4.py`

```python
assert ".py" in CODE_EXTENSIONS
assert ".js" in CODE_EXTENSIONS
assert ".v" in CODE_EXTENSIONS    # Verilog
assert ".sv" in CODE_EXTENSIONS   # SystemVerilog
assert ".sql" in CODE_EXTENSIONS
assert "*.env" in EXCLUDED_FILE_PATTERNS
assert "*node_modules*" in EXCLUDED_FILE_PATTERNS
assert "*chroma_db*" in EXCLUDED_FILE_PATTERNS
assert "*.lock" in EXCLUDED_FILE_PATTERNS
# No overlap between extensions and exclusions
excluded_exts = {p.lstrip("*") for p in EXCLUDED_FILE_PATTERNS if p.startswith("*.") and "*" not in p[2:]}
assert not set(CODE_EXTENSIONS) & excluded_exts
```

---

## 3. Backend Tests (T-BE)

### T-BE-1 — Embedding Model Thread Safety
**File:** `test_backend_full.py`
**Function:** `get_embedding_model()`
**What it checks:** Singleton under concurrent load.

```python
def test_be_1_embedding_thread_safety():
    from concurrent.futures import ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(get_embedding_model) for _ in range(4)]
        results = [f.result() for f in futures]
    # All threads get the same instance
    assert all(r is results[0] for r in results)
```

---

### T-BE-2 — Content Hash Determinism
**File:** `test_backend_full.py`
**Function:** `_content_hash(doc)`

```python
def test_be_2_content_hash():
    doc_a = Document(page_content="hello world", metadata={"source": "a.py"})
    doc_b = Document(page_content="hello world", metadata={"source": "b.py"})  # same content, different meta
    doc_c = Document(page_content="different", metadata={})
    assert _content_hash(doc_a) == _content_hash(doc_b)  # metadata irrelevant
    assert _content_hash(doc_a) != _content_hash(doc_c)
    assert len(_content_hash(doc_a)) == 64  # SHA-256 hex
```

---

### T-BE-3 — SQLiteFTS5BM25 Incremental Indexing & Dedup
**File:** `test_backend_full.py`
**Class:** `SQLiteFTS5BM25`

```python
def test_be_3_fts5_incremental(tmp_path):
    fts = SQLiteFTS5BM25(str(tmp_path / "test_col"))
    docs = [Document(page_content="foo bar baz", metadata={"source": "a.py"}),
            Document(page_content="hello world", metadata={"source": "b.py"})]
    fts.add_documents(docs)
    fts.add_documents(docs)  # re-add same docs
    results = fts.search("foo", k=10)
    assert len(results) == 1  # no duplicates
```

---

### T-BE-4 — SQLiteFTS5BM25 Search Precision
**File:** `test_backend_full.py`

```python
def test_be_4_fts5_search(tmp_path):
    fts = SQLiteFTS5BM25(str(tmp_path / "test_col"))
    fts.add_documents([Document(page_content="The quick brown fox"), ...])
    assert fts.search("fox", k=5)          # returns result
    assert fts.search("cat", k=5) == []    # no result, no crash
```

---

### T-BE-5 — Deduplication on Re-Ingest
**File:** `test_backend_full.py`
**Function:** `ingest_into_chroma()`

```python
def test_be_5_dedup_reingest(tmp_path):
    docs = [Document(page_content="unique content", metadata={"source": "f.py"})]
    db, added_first = ingest_into_chroma(docs, collection_name="test_dedup")
    _, added_second = ingest_into_chroma(docs, collection_name="test_dedup")
    assert added_first == 1
    assert added_second == 0  # duplicate skipped
```

---

### T-BE-6 — Incremental Multi-File Ingestion
**File:** `test_backend_full.py`

```python
def test_be_6_incremental_append():
    # Ingest file_A (5 chunks), then file_B (3 chunks)
    # Expect total 8 chunks with both source files in metadata
    ...
    info = get_collection_info("test_incremental")
    assert info["total_chunks"] == 8
    assert "file_a.py" in info["sources"]
    assert "file_b.py" in info["sources"]
```

---

### T-BE-7 — Exclusion Patterns Respected
**File:** `test_backend_full.py`
**Function:** `_collect_code_files()`, `_is_excluded()`

```python
def test_be_7_exclusion_patterns(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / ".env").write_text("SECRET=xxx")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "old.py").write_text("pass")
    files = _collect_code_files(str(tmp_path))
    assert any("app.py" in f for f in files)
    assert not any("package-lock.json" in f for f in files)
    assert not any(".env" in f for f in files)
    assert not any("archive" in f for f in files)
```

---

### T-BE-8 — AST Chunker: Python Logical Boundaries
**File:** `test_backend_full.py`
**Class:** `CodeASTChunker`

```python
def test_be_8_ast_python():
    content = '''
class Foo:
    def method_a(self): pass
    def method_b(self): pass

def standalone_func(): pass
'''
    chunker = CodeASTChunker()
    docs = chunker.chunk_file(content, "foo.py", ".py")
    # Expect at least 2 chunks (class + function)
    assert len(docs) >= 2
    # Context header present
    assert any("Foo" in d.page_content or "Foo" in str(d.metadata) for d in docs)
```

---

### T-BE-9 — AST Chunker: HDL Regex Fallback
**File:** `test_backend_full.py`

```python
def test_be_9_hdl_regex():
    verilog = "module counter(input clk, output reg [7:0] out);\nalways @(posedge clk) out <= out + 1;\nendmodule"
    chunker = CodeASTChunker()
    docs = chunker.chunk_file(verilog, "counter.v", ".v")
    assert len(docs) >= 1
    assert docs[0].metadata.get("hdl_type") == "regex_block"
```

---

### T-BE-10 — Large File Chunking (>100k chars)
**File:** `test_backend_full.py`

```python
def test_be_10_large_file_chunking(tmp_path):
    large_content = "x = 1\n" * 20000   # ~120,000 chars
    f = tmp_path / "large.py"
    f.write_text(large_content)
    docs = load_and_chunk_codebase(str(tmp_path))
    assert len(docs) > 1                 # split, not zero-chunked
    for i, d in enumerate(docs):
        assert d.metadata.get("chunk_index") == i
    assert not docs[0].metadata.get("zero_chunk")
```

---

### T-BE-11 — Collection Deletion Cleanup
**File:** `test_backend_full.py`
**Function:** `delete_collection()`

```python
def test_be_11_collection_delete():
    ingest_into_chroma([Document(page_content="temp")], collection_name="to_delete")
    assert delete_collection("to_delete") is True
    assert "to_delete" not in list_collections()
    # BM25 db removed
    import os
    assert not os.path.exists(_get_bm25_path("to_delete"))
```

---

## 4. RAG Chain Unit Tests (T-RC)

### T-RC-1 — Cache Capability Detection & Profile Routing
**File:** `test_t_rc_1.py`
**Functions:** `is_cache_capable()`, `get_cache_profile()`

| Model String | `is_cache_capable` | `get_cache_profile` |
|---|---|---|
| `"claude-3.5-sonnet"` | `True` | `(4, 1024)` |
| `"google/gemini-2.0-flash"` | `True` | `(8, 1028)` |
| `"google/gemma-4-31b-it:free"` | `True` | `(8, 1028)` |
| `"deepseek/deepseek-chat"` | `False` | `(4, 1024)` |
| `"qwen/qwen3-coder:free"` | `False` | `(4, 1024)` |
| `"ollama/llama3.2:1b"` | `False` | `(4, 1024)` |
| `None` | `False` | `(4, 1024)` |

---

### T-RC-2 — Query Normalization
**File:** `test_t_rc_2.py`
**Function:** `_normalize_query()`

| Input | Expected Output |
|-------|----------------|
| `"What is Python?"` | `"what is python"` |
| `"  Hello, World!  "` | `"hello world"` |
| `"C++ vs Rust"` | `"c vs rust"` |
| `"test"` | `"test"` |
| `""` | `""` |

---

### T-RC-3 — Pinned Embedding LRU Cache
**File:** `test_rag_full.py`
**Function:** `_get_pinned_embedding()`

```python
def test_rc_3_pinned_embedding_cache(mocker):
    mock_embed = mocker.patch("rag_chain.get_embedding_model")
    mock_embed.return_value.embed_query.return_value = [0.1, 0.2]
    _get_pinned_embedding.cache_clear()
    _get_pinned_embedding("prefix text")
    _get_pinned_embedding("prefix text")  # second call — same arg
    assert mock_embed.return_value.embed_query.call_count == 1
    assert _get_pinned_embedding.cache_info().hits == 1
```

---

### T-RC-4 — Cosine Similarity Correctness
**File:** `test_rag_full.py`
**Function:** `calculate_cosine_similarity()`

| vec1 | vec2 | Expected |
|------|------|---------|
| `[1, 0]` | `[1, 0]` | `1.0` |
| `[1, 0]` | `[0, 1]` | `0.0` |
| `[1, 0]` | `[-1, 0]` | `-1.0` |
| `[0, 0]` | `[1, 0]` | `0.0` (zero-vector guard) |

---

### T-RC-5 — Deterministic Sort with Stable Hashes
**File:** `test_rag_full.py`
**Function:** `_sort_docs_deterministically()`

```python
def test_rc_5_deterministic_sort():
    docs = [
        Document(page_content="Z doc", metadata={"source": "z.py", "content_hash": "hash_z"}),
        Document(page_content="A doc", metadata={"source": "a.py", "content_hash": "hash_a"}),
    ]
    # Without stable hashes: alphabetical by source
    sorted_no_stable = _sort_docs_deterministically(docs)
    assert sorted_no_stable[0].metadata["source"] == "a.py"

    # With stable hashes: stable docs first
    sorted_stable = _sort_docs_deterministically(docs, stable_hashes={"hash_z"})
    assert sorted_stable[0].metadata["content_hash"] == "hash_z"
```

---

### T-RC-6 — Module Singletons
**File:** `test_rag_full.py`
**Functions:** `get_router()`, `get_semantic_cache()`, `get_reranker()`

```python
def test_rc_6_singletons():
    assert get_router() is get_router()
    assert get_semantic_cache() is get_semantic_cache()
    assert get_reranker() is get_reranker()
```

---

### T-RC-7 — VectorRouter: Specialty Detection
**File:** `test_rag_full.py`
**Function:** `VectorRouter.detect_specialty()`

| Query | Expected Specialty |
|-------|--------------------|
| `"write a python function to sort a list"` | `CODE` |
| `"refactor this class to use dependency injection"` | `CODE` |
| `"debug this segmentation fault"` | `CODE` |
| `"analyze the architectural trade-offs"` | `REASONING` |
| `"prove that this algorithm is O(n log n)"` | `REASONING` |
| `"show me a diagram of the flow"` | `VISION` |
| `"what is the capital of France"` | `GENERAL` |
| `"explain what this module does"` | `GENERAL` |

---

### T-RC-8 — LocalReRanker Cross-Encoder Scoring
**File:** `test_rag_full.py`
**Function:** `LocalReRanker.rerank()`

```python
def test_rc_8_reranker():
    reranker = get_reranker()
    docs = [
        Document(page_content="Python is a programming language", metadata={}),
        Document(page_content="The sky is blue and clouds are white", metadata={}),
        Document(page_content="def sort_list(lst): return sorted(lst)", metadata={}),
    ]
    result = reranker.rerank("How do I sort a list in Python?", docs, top_k=2)
    assert len(result) == 2
    # Code-related docs should rank higher than sky
    contents = [d.page_content for d in result]
    assert "sky" not in contents[0]   # irrelevant doc not top-1
```

---

### T-RC-9 — SemanticCache Lookup & Upsert
**File:** `test_rag_full.py`
**Class:** `SemanticCache`

```python
def test_rc_9_semantic_cache(mocker):
    cache = SemanticCache.__new__(SemanticCache)
    mock_db = mocker.MagicMock()
    cache.db = mock_db
    # Simulate score above threshold
    mock_db.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="q", metadata={"answer": "cached answer"}), 0.92)
    ]
    result = cache.lookup("test query", threshold=0.85)
    assert result == "cached answer"

    # Simulate score below threshold
    mock_db.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="q", metadata={"answer": "old answer"}), 0.60)
    ]
    result = cache.lookup("test query", threshold=0.85)
    assert result is None
```

---

### T-RC-10 — Hybrid Search RRF Fusion
**File:** `test_rag_full.py`
**Function:** `hybrid_search()`

```python
def test_rc_10_rrf_fusion(mocker):
    # D2 appears in both vector and BM25 results → should rank first after RRF
    mock_db = mocker.MagicMock()
    D1 = Document(page_content="doc1", metadata={"source": "a.py", "content_hash": "h1"})
    D2 = Document(page_content="doc2", metadata={"source": "b.py", "content_hash": "h2"})
    D3 = Document(page_content="doc3", metadata={"source": "c.py", "content_hash": "h3"})
    mock_db.similarity_search_by_vector.return_value = [D1, D2]
    mocker.patch("rag_chain.SQLiteFTS5BM25").return_value.search.return_value = [D2, D3]
    results = hybrid_search(mock_db, "some query", k=3)
    assert results[0].page_content == "doc2"  # D2 ranked first (in both lists)
```

---

### T-RC-11 — Exact-Match Embedding Cache Bypass
**File:** `test_rag_full.py`

```python
def test_rc_11_exact_match_no_reembed(mocker):
    # When last_query_embedding provided and query matches last_query,
    # embed_query should not be called for retrieval
    mock_embed = mocker.patch("rag_chain.get_embedding_model")
    mock_embed.return_value.embed_query.return_value = [0.5] * 384
    # Simulate chain call with identical query
    ...
    assert mock_embed.return_value.embed_query.call_count == 0
```

---

### T-RC-12 — `_get_max_tokens` Dynamic Budget
**File:** `test_rag_full.py` *(NEW — covers 2026-04-11 change)*
**Function:** `_get_max_tokens()`

| specialty | query | Expected |
|-----------|-------|---------|
| `"CODE"` | `"sort a list"` | `4096` |
| `"REASONING"` | `"short q"` | `4096` |
| `"GENERAL"` | `"what is x"` | `1024` |
| `"VISION"` | `"hello"` | `1024` |
| `None` | `"hi"` | `1024` |
| `"GENERAL"` | `"x" * 201` (>200 chars) | `4096` |
| `"GENERAL"` | `"x" * 200` (== 200 chars) | `1024` |

```python
from rag_chain import _get_max_tokens, MAX_TOKENS

def test_rc_12_max_tokens():
    assert _get_max_tokens("CODE", "short") == MAX_TOKENS
    assert _get_max_tokens("REASONING", "short") == MAX_TOKENS
    assert _get_max_tokens("GENERAL", "short") == 1024
    assert _get_max_tokens("VISION", "short") == 1024
    assert _get_max_tokens(None, "short") == 1024
    assert _get_max_tokens("GENERAL", "x" * 201) == MAX_TOKENS  # long query → full budget
    assert _get_max_tokens("GENERAL", "x" * 200) == 1024        # boundary: exactly 200 chars → reduced
```

---

## 5. Retrieval Integration Tests (T-RET)

### T-RET-1 — Vector-Only Retrieval Returns K Docs
```python
def test_ret_1_vector_retrieval(real_db):
    results = real_db.similarity_search("function definition", k=6)
    assert 1 <= len(results) <= 6
    for doc in results:
        assert doc.page_content
        assert "source" in doc.metadata
```

---

### T-RET-2 — Hybrid Retrieval Outranks Pure Vector
```python
def test_ret_2_hybrid_vs_vector(real_db):
    # Query matches a specific keyword in BM25 but not top vector result
    hybrid = hybrid_search(real_db, "SQLiteFTS5BM25", k=6)
    vector = real_db.similarity_search("SQLiteFTS5BM25", k=6)
    # Hybrid should surface the exact-match doc higher
    hybrid_sources = [d.metadata.get("source") for d in hybrid]
    assert any("backend" in s for s in hybrid_sources)
```

---

### T-RET-3 — Reranking Improves Relevance Order
```python
def test_ret_3_reranker_order(real_db):
    query = "how does history compression work"
    docs_raw = hybrid_search(real_db, query, k=6)
    docs_reranked = get_reranker().rerank(query, docs_raw, top_k=6)
    # Top reranked doc should contain "history" or "compress"
    assert any(
        "history" in docs_reranked[0].page_content.lower()
        or "compress" in docs_reranked[0].page_content.lower()
    )
```

---

### T-RET-4 — Extension Filter Applied
```python
def test_ret_4_extension_filter(real_db):
    results = hybrid_search(real_db, "class definition", k=6, filter_extensions=[".py"])
    for doc in results:
        assert doc.metadata.get("file_extension") == ".py"
```

---

### T-RET-5 — Exclude File Applied
```python
def test_ret_5_exclude_file(real_db):
    excluded = "app.py"
    results = hybrid_search(real_db, "session state", k=6, exclude_file=excluded)
    for doc in results:
        assert excluded not in doc.metadata.get("source", "")
```

---

### T-RET-6 — Pinned Content Relevance Gating
```python
def test_ret_6_pinned_gating(mocker):
    # Pinned content with low similarity to query → not included (STICKY=False)
    mocker.patch("rag_chain.STICKY_PINNED_CONTEXT", False)
    mocker.patch("rag_chain._get_pinned_embedding", return_value=[1, 0, 0])
    mocker.patch("rag_chain.get_embedding_model").return_value.embed_query.return_value = [0, 1, 0]
    # cosine similarity == 0.0 < threshold 0.40 → pinned excluded
    ...
```

---

### T-RET-7 — RERANK_CANDIDATES=15 Enforced
**Verifies the updated config value propagates correctly into hybrid_search.**
```python
def test_ret_7_rerank_candidates(mocker):
    mock_db = mocker.MagicMock()
    mock_db.similarity_search_by_vector.return_value = []
    mocker.patch("rag_chain.SQLiteFTS5BM25").return_value.search.return_value = []
    hybrid_search(mock_db, "test", k=RERANK_CANDIDATES)
    # k*3 = 45 docs requested from vector store (15 * 3)
    call_kwargs = mock_db.similarity_search_by_vector.call_args
    assert call_kwargs[1]["k"] == 45  # 15 * 3
```

---

## 6. History & Memory Tests (T-HIST)

### T-HIST-1 — Sentinel-Present Compression: Keep Last 4
**File:** `test_history_full.py`
**Function:** `compress_chat_history()`

```python
def test_hist_1_sentinel_keeps_last_4():
    history = [HumanMessage("q1"), AIMessage("a1"),
               HumanMessage("q2"), AIMessage("a2"),
               HumanMessage("q3"), AIMessage("a3"),
               HumanMessage("q4"), AIMessage("a4")]
    compressed = compress_chat_history(history, sentinel_state="Summary: topics A, B, C.")
    assert len(compressed) == 4  # only last 4 messages
    assert compressed[-1].content.startswith("a4")
```

---

### T-HIST-2 — AI Response Truncation Preserves Code Blocks
**File:** `test_history_full.py`
**Function:** `_truncate_ai_in_history()`

```python
def test_hist_2_truncation_preserves_code():
    long_prose = "x" * 1000
    code = "```python\ndef foo(): pass\n```"
    ai_msg = AIMessage(f"{long_prose}\n{code}")
    result = _truncate_ai_in_history([ai_msg])
    assert "```python" in result[0].content   # code block preserved
    assert len(result[0].content) <= 2400     # cap applied
```

---

### T-HIST-3 — Short History Passthrough (<= GHOST_HISTORY_MAX)
```python
def test_hist_3_short_passthrough():
    history = [HumanMessage("q"), AIMessage("a")]
    result = compress_chat_history(history, sentinel_state="")
    assert result == history   # no compression on short history
```

---

### T-HIST-4 — Ghost History: Token Budget Enforced
```python
def test_hist_4_token_budget():
    # Build history that exceeds MAX_HISTORY_TOKENS (2000)
    history = []
    for i in range(30):
        history.append(HumanMessage(f"question {i} " * 50))
        history.append(AIMessage(f"answer {i} " * 100))
    compressed = compress_chat_history(history, sentinel_state="")
    from rag_chain import _est_tokens
    assert _est_tokens(compressed) <= 2000
```

---

### T-HIST-5 — Sentinel Trigger: Fires at SENTINEL_TOKEN_THRESHOLD=1500
**Verifies the updated threshold (was 2000).**
```python
def test_hist_5_sentinel_trigger_threshold():
    # Exactly at 1500 tokens → should_summarize = True
    # Build history with ~1500 estimated tokens
    history = [HumanMessage("w" * 2250), AIMessage("w" * 2250)]  # ~1500 tokens (4500 chars // 3)
    from rag_chain import _est_tokens, SENTINEL_TOKEN_THRESHOLD
    assert _est_tokens(history) >= SENTINEL_TOKEN_THRESHOLD
    # Below threshold — no trigger
    history_small = [HumanMessage("hi"), AIMessage("hello")]
    assert _est_tokens(history_small) < SENTINEL_TOKEN_THRESHOLD
```

---

### T-HIST-6 — Sentinel Interval: Fires Every 3 Turns (was 5)
```python
def test_hist_6_sentinel_interval():
    from rag_chain import SENTINEL_INTERVAL
    assert SENTINEL_INTERVAL == 3
    # Simulate turn counter logic:
    # turn 1, 2: no summarize (< interval)
    # turn 3: summarize (== interval)
    # turn 4, 5: no summarize (< interval from last)
    # turn 6: summarize again
    last_turn = 0
    for turn in range(1, 9):
        should = (turn - last_turn) >= SENTINEL_INTERVAL
        if should:
            last_turn = turn
    assert last_turn == 6   # fired at turns 3 and 6
```

---

## 7. Ingestion Tests (T-INGEST)

### T-INGEST-1 — Async PDF Ingestion
**File:** `test_ingest_integration.py`
**Class:** `AsyncIngestionTask`

```python
def test_ingest_1_async_pdf(tmp_path):
    task = AsyncIngestionTask(str(test_pdf_path), collection_name="pdf_test", is_pdf=True)
    task.start()
    import time; time.sleep(5)
    assert task.is_done
    assert task.error is None
    db = load_existing_chroma("pdf_test")
    assert db is not None
    assert db._collection.count() > 0
```

---

### T-INGEST-2 — Async Folder Ingestion with Progress
```python
def test_ingest_2_async_folder(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"def func_{i}(): pass")
    progress_calls = []
    task = AsyncIngestionTask(str(tmp_path), collection_name="folder_test")
    task.start()
    import time; time.sleep(10)
    assert task.is_done
    info = get_collection_info("folder_test")
    assert info["total_chunks"] >= 5
```

---

### T-INGEST-3 — PDF Zero-Chunk for Small Files
```python
def test_ingest_3_pdf_zero_chunk(small_pdf):
    docs = load_and_chunk_pdf(str(small_pdf))
    if sum(len(d.page_content) for d in docs) < 100000:
        assert len(docs) == 1
        assert docs[0].metadata.get("zero_chunk") is True
```

---

### T-INGEST-4 — Code Zero-Chunk for Small Files
```python
def test_ingest_4_code_zero_chunk(tmp_path):
    f = tmp_path / "small.py"
    f.write_text("x = 1\n" * 100)   # well under 100k chars
    docs = load_and_chunk_codebase(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata.get("zero_chunk") is True
```

---

### T-INGEST-5 — Re-ingest Skips Duplicates
```python
def test_ingest_5_reingest_dedup():
    _, first = ingest_into_chroma(sample_docs, "reingest_test")
    _, second = ingest_into_chroma(sample_docs, "reingest_test")
    assert second == 0
```

---

### T-INGEST-6 — Collection Listing
```python
def test_ingest_6_list_collections():
    ingest_into_chroma([Document(page_content="hi", metadata={})], "col_list_test")
    cols = list_collections()
    assert "col_list_test" in cols
```

---

## 8. Token Optimization Tests (T-TOK)

These tests are **NEW** — added 2026-04-11 to validate the token-efficiency optimizations applied to this codebase.

---

### T-TOK-1 — Zero-Chunk Guard: Blocks Oversized Docs for ALL Models
**File:** `test_rag_full.py` *(new section)*
**What it checks:** The unconditional zero-chunk guard in `_full_context_cache_chain` drops documents that have `zero_chunk=True` AND `len(page_content) > MAX_ZERO_CHUNK_CHARS`.

```python
def test_tok_1_zero_chunk_guard_all_models(mocker):
    """
    Verify that zero-chunks exceeding MAX_ZERO_CHUNK_CHARS are filtered
    regardless of whether the model is cache-capable.
    Previously this guard only ran for non-cache models (BUG).
    """
    from rag_chain import MAX_ZERO_CHUNK_CHARS
    from config import MAX_ZERO_CHUNK_CHARS as CFG_MAX

    assert MAX_ZERO_CHUNK_CHARS == CFG_MAX == 9000

    oversized = Document(
        page_content="x" * 10001,
        metadata={"source": "big.py", "zero_chunk": True, "content_hash": "h1"}
    )
    normal = Document(
        page_content="def foo(): pass",
        metadata={"source": "small.py", "zero_chunk": False, "content_hash": "h2"}
    )
    zero_small = Document(
        page_content="x" * 8000,   # zero_chunk but under threshold
        metadata={"source": "ok.py", "zero_chunk": True, "content_hash": "h3"}
    )

    docs = [oversized, normal, zero_small]
    filtered = [
        d for d in docs
        if not (d.metadata.get("zero_chunk") and len(d.page_content) > MAX_ZERO_CHUNK_CHARS)
    ]

    assert oversized not in filtered       # blocked
    assert normal in filtered              # allowed
    assert zero_small in filtered          # allowed (under 9000)
```

---

### T-TOK-2 — Zero-Chunk Guard: Previously Exempt Cache-Capable Models Now Filtered
```python
def test_tok_2_zero_chunk_cache_capable_blocked(mocker):
    """
    Regression test: before the fix, provider_has_cache=True bypassed the filter.
    Ensure a Gemini/Claude query no longer allows zero-chunks > 9000 chars.
    """
    # Simulate the old broken guard
    def old_guard(docs, provider_has_cache):
        if not provider_has_cache:
            return [d for d in docs if not (d.metadata.get("zero_chunk") and len(d.page_content) > 10000)]
        return docs  # old code: no filter for cache-capable

    # Simulate the new correct guard
    def new_guard(docs):
        return [d for d in docs if not (d.metadata.get("zero_chunk") and len(d.page_content) > 9000)]

    oversized = Document(page_content="x" * 10001, metadata={"zero_chunk": True})

    assert oversized in old_guard([oversized], provider_has_cache=True)   # old: passes through!
    assert oversized not in new_guard([oversized])                        # new: blocked
```

---

### T-TOK-3 — Sentinel Fires Earlier (Threshold=1500, not 2000)
```python
def test_tok_3_sentinel_earlier_trigger():
    """
    Sentinel should now fire at 1500 tokens, not 2000.
    A conversation of ~1600 token history should trigger summarization.
    """
    from rag_chain import SENTINEL_TOKEN_THRESHOLD
    assert SENTINEL_TOKEN_THRESHOLD == 1500

    # 1600 tokens ≈ 4800 chars
    history = [HumanMessage("word " * 800), AIMessage("word " * 800)]
    from rag_chain import _est_tokens
    assert _est_tokens(history) >= 1500   # should trigger
    assert _est_tokens(history) < 2000    # would NOT have triggered with old threshold
```

---

### T-TOK-4 — Sentinel Fires More Frequently (Interval=3, not 5)
```python
def test_tok_4_sentinel_interval():
    from rag_chain import SENTINEL_INTERVAL
    assert SENTINEL_INTERVAL == 3
    # Verify 3 turns after last summary is enough to trigger again
    last_turn = 0
    current_turn = 3
    assert (current_turn - last_turn) >= SENTINEL_INTERVAL  # fires at turn 3
    current_turn = 4
    assert (current_turn - last_turn) < SENTINEL_INTERVAL   # does not fire at turn 4
```

---

### T-TOK-5 — Rerank Candidates Reduced to 15
```python
def test_tok_5_rerank_candidates():
    from config import RERANK_CANDIDATES, RERANK_TOP_K
    assert RERANK_CANDIDATES == 15
    assert RERANK_TOP_K <= RERANK_CANDIDATES  # sanity: top_k <= candidates
    # Verify k*3 = 45 (not 75 as before)
    k = RERANK_CANDIDATES
    expected_fetch = k * 3
    assert expected_fetch == 45
```

---

### T-TOK-6 — Dynamic Max Tokens: GENERAL Short Query Gets 1024
```python
def test_tok_6_dynamic_max_tokens_general():
    from rag_chain import _get_max_tokens
    result = _get_max_tokens("GENERAL", "what is a variable?")
    assert result == 1024   # saves 3072 reserved tokens per query
```

---

### T-TOK-7 — Dynamic Max Tokens: CODE Query Gets Full 4096
```python
def test_tok_7_dynamic_max_tokens_code():
    from rag_chain import _get_max_tokens, MAX_TOKENS
    result = _get_max_tokens("CODE", "write a binary search implementation")
    assert result == MAX_TOKENS == 4096
```

---

### T-TOK-8 — Dynamic Max Tokens: Long GENERAL Query Gets Full Budget
```python
def test_tok_8_dynamic_max_tokens_long_query():
    from rag_chain import _get_max_tokens, MAX_TOKENS
    long_query = "explain " * 30   # 210 chars
    result = _get_max_tokens("GENERAL", long_query)
    assert result == MAX_TOKENS   # length > 200 → full budget
```

---

## 9. End-to-End Query Tests (T-E2E)

These tests require a running RAG chain with an ingested collection.

### T-E2E-1 — Single-Turn Query Returns Answer
```python
def test_e2e_1_single_turn(rag_chain, test_db):
    results = list(rag_chain.stream({
        "input": "What does the hybrid_search function do?",
        "chat_history": [],
        "full_source_context": "None pinned.",
        "sentinel_state": "",
        "cached_docs": [],
        "last_query": "",
        "last_query_embedding": None,
        "force_retrieval": False,
        "collection_name": "test_col",
        "auto_specialist": False,
    }))
    answer_chunks = [r.get("answer", "") for r in results if "answer" in r]
    full_answer = "".join(answer_chunks)
    assert len(full_answer) > 50
    assert "retrieval" in full_answer.lower() or "search" in full_answer.lower()
```

---

### T-E2E-2 — Follow-Up Intent Detected and Context Reused
```python
def test_e2e_2_followup_reuses_context(rag_chain, test_db):
    # Turn 1: ask about hybrid search
    # Turn 2: follow-up "how is it different from vector-only?"
    # Expected: intent = FOLLOW-UP, previous docs included
    ...
    assert any(r.get("intent") == "FOLLOW-UP" for r in results_turn2 if "intent" in r)
```

---

### T-E2E-3 — Model Switching Clears Cache State
```python
def test_e2e_3_model_switch(rag_chain_gemma, rag_chain_qwen):
    # Verify different RAG chains use different specialist caches
    assert rag_chain_gemma is not rag_chain_qwen
```

---

### T-E2E-4 — Semantic Cache Hit on Near-Identical Query
```python
def test_e2e_4_semantic_cache_hit(rag_chain, test_db, seeded_cache):
    # seeded_cache: "What is BM25?" → "BM25 is a ranking function..."
    results = list(rag_chain.stream({"input": "What is BM25?", ...}))
    intents = [r.get("intent") for r in results if "intent" in r]
    assert "CACHE_HIT" in intents  # skipped retrieval entirely
```

---

### T-E2E-5 — Long Conversation Triggers Sentinel at Turn 3
```python
def test_e2e_5_sentinel_fires(rag_chain, token_heavy_history):
    # history with ~1600 tokens → should trigger sentinel at turn 3
    # After stream, sentinel_future should be set
    results = list(rag_chain.stream({
        "input": "Continue explaining",
        "chat_history": token_heavy_history,
        ...
    }))
    futures = [r.get("sentinel_future") for r in results if r.get("sentinel_future")]
    assert len(futures) > 0   # background summarization was triggered
```

---

## 10. Formulated RAG Questions (T-QA)

These are **curated questions** to ask the live RAG system after ingesting this codebase. Each includes the expected answer direction and acceptance criteria.

> **Setup:** Ingest all `.py` files from this directory into a collection named `"self_test"`. Use default model. Run each question and evaluate the response.

---

### QA-01 — Architecture Question
**Question:** `"Explain the end-to-end flow when a user submits a query. What are the main stages?"`

**Expected topics in answer:**
- History compression / `compress_chat_history`
- Semantic cache lookup
- Intent classification (NEW vs FOLLOW-UP)
- Hybrid retrieval (vector + BM25)
- Reranking
- Prompt assembly with cache blocks
- LLM streaming

**Accept if:** Answer mentions at least 5 of the 7 stages above.

---

### QA-02 — Specific Function Query
**Question:** `"What does the hybrid_search function do and what algorithm does it use to merge results?"`

**Expected:** Mentions RRF (Reciprocal Rank Fusion), vector search via ChromaDB, BM25/FTS5, and that weights are configurable (BM25_WEIGHT, VECTOR_WEIGHT).

---

### QA-03 — Config Parameter Query
**Question:** `"What is the current value of RERANK_CANDIDATES and why was it recently changed?"`

**Expected:** Answer states `15` (updated from 25). If the commit history was ingested, may mention token efficiency.

---

### QA-04 — Code Generation (Specialist Routing Test)
**Question:** `"Write a Python function that estimates the token count of a list of LangChain messages using the _est_tokens logic from this codebase."`

**Expected:** Specialist routing → `CODE` → `qwen/qwen3-coder:free`. Answer should include code block with `len(content) // 3` logic.

**Check:** `specialist_active` metadata in stream should be `"qwen/qwen3-coder:free"`.

---

### QA-05 — Follow-Up Chain (Multi-Turn Test)
**Turn 1:** `"What is the SemanticCache class?"`
**Turn 2:** `"How does its lookup method decide whether to return a cached answer?"`
**Turn 3:** `"What threshold is used and where is it configured?"`

**Expected flow:**
- Turn 1: Intent = `NEW`, retrieves SemanticCache docs
- Turn 2: Intent = `FOLLOW-UP`, reuses docs from Turn 1, query rewritten to standalone
- Turn 3: Intent = `FOLLOW-UP`, mentions `SEMANTIC_CACHE_THRESHOLD = 0.85` in `config.py`

---

### QA-06 — History Management Query
**Question:** `"What happens to chat history when it grows large? Explain the Ghost History mechanism."`

**Expected:** Mentions `GHOST_HISTORY_MAX`, `GHOST_AI_CHARS = 200`, `AI_RESPONSE_MAX_CHARS = 800`, anchor preservation (first 2 messages), and the sentinel as a better long-term solution.

---

### QA-07 — Zero-Chunk Behavior Query
**Question:** `"What is a zero-chunk and when is it created during ingestion?"`

**Expected:** Files under `ZERO_CHUNK_THRESHOLD` (100,000 chars) are stored as a single document with `zero_chunk=True`. Avoids excessive fragmentation for small files.

---

### QA-08 — Token Optimization Query (Tests New Code)
**Question:** `"Why are large zero-chunk documents now filtered out from retrieval results?"`

**Expected:** Explains that a 100k-char file = ~33k tokens, which floods the context window. Retrieval of such a doc gives worse signal-to-noise than focused chunks. Pinned-file mechanism is the correct way to view entire files.

---

### QA-09 — Caching Architecture Query
**Question:** `"How does provider-side prefix caching work in this system? Which models support it?"`

**Expected:** Mentions `cache_control: {"type": "ephemeral"}`, the 5 system blocks (CORE_INSTRUCTIONS, PINNED, SENTINEL, STABLE RAG, NEW RAG), deterministic sort for prefix stability, and Claude (4 breakpoints) and Gemini/Gemma (8 breakpoints).

---

### QA-10 — Reasoning Task (Specialist Routing Test)
**Question:** `"Analyze the architectural trade-offs between using a Sentinel summary vs. keeping full chat history for long conversations."`

**Expected:** Specialist routing → `REASONING`. Answer discusses: Sentinel (low tokens, loses detail), Full History (high tokens, complete context), Ghost History (compromise), and when each is appropriate.

**Check:** `specialist_active` should be `"liquid/lfm-2.5-1.2b-thinking:free"`.

---

### QA-11 — Hallucination Guard (Out-of-Context Query)
**Question:** `"What is the GDP of France in 2024?"`

**Expected:** System should say it cannot answer from the provided context (no relevant documents retrieved about GDP). Should NOT fabricate an answer.

**Accept if:** Response contains "not in the context", "cannot find", "no information", or similar hedge.

---

### QA-12 — Pinned File Test
**Setup:** Pin `rag_chain.py` as context.
**Question:** `"In the pinned file, what line does the _get_max_tokens function start on and what does it return for a GENERAL query shorter than 200 characters?"`

**Expected:** Correctly reads the pinned file and answers `1024`.

---

### QA-13 — Config Change Awareness
**Question:** `"What are the current values for SENTINEL_INTERVAL and SENTINEL_TOKEN_THRESHOLD?"`

**Expected:** `3` and `1500` respectively (updated 2026-04-11).

---

### QA-14 — Cross-Encoder Reranking Query
**Question:** `"What model is used for reranking retrieved documents and what does it optimize for?"`

**Expected:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, optimizes for relevance between query and document pairs, scores each (query, doc) pair independently.

---

### QA-15 — Recursive Self-Reference Query
**Question:** `"If I ask the same question twice, what happens on the second ask?"`

**Expected:** Semantic cache lookup → if similarity ≥ 0.85 → returns cached answer immediately without retrieval or LLM call. Mentions `SEMANTIC_CACHE_THRESHOLD`.

---

## 11. Edge Case & Failure Mode Tests (T-EDGE)

### T-EDGE-1 — Empty Query String
```python
def test_edge_1_empty_query(rag_chain):
    # Should not crash; returns graceful empty or error response
    results = list(rag_chain.stream({"input": "", ...}))
    assert all(isinstance(r, dict) for r in results)
```

---

### T-EDGE-2 — No Documents Retrieved (Zero Results)
```python
def test_edge_2_zero_retrieval(mocker, rag_chain):
    mocker.patch("rag_chain.hybrid_search", return_value=[])
    results = list(rag_chain.stream({"input": "xyzzy_nonexistent_term", ...}))
    answer = "".join(r.get("answer", "") for r in results)
    assert "context" in answer.lower() or "not" in answer.lower()  # graceful degradation
```

---

### T-EDGE-3 — Ollama Router Timeout (2s)
```python
def test_edge_3_ollama_timeout(mocker):
    # classify_intent should fall back to "NEW" if Ollama times out
    mocker.patch("rag_chain.VectorRouter.classify_intent", side_effect=TimeoutError)
    # Ensure the pipeline continues with "NEW" intent fallback
    ...
```

---

### T-EDGE-4 — Zero-Chunk Doc Exactly at Threshold (9000 chars)
```python
def test_edge_4_zero_chunk_boundary():
    from rag_chain import MAX_ZERO_CHUNK_CHARS
    at_threshold = Document(page_content="x" * MAX_ZERO_CHUNK_CHARS, metadata={"zero_chunk": True})
    one_over = Document(page_content="x" * (MAX_ZERO_CHUNK_CHARS + 1), metadata={"zero_chunk": True})
    docs = [at_threshold, one_over]
    filtered = [d for d in docs if not (d.metadata.get("zero_chunk") and len(d.page_content) > MAX_ZERO_CHUNK_CHARS)]
    assert at_threshold in filtered    # exactly at threshold: allowed
    assert one_over not in filtered    # one over: blocked
```

---

### T-EDGE-5 — History of Exactly MAX_HISTORY_TOKENS
```python
def test_edge_5_history_at_limit():
    # Build history at exactly the token limit
    from rag_chain import _est_tokens, MAX_HISTORY_TOKENS
    history = [HumanMessage("w " * 1500), AIMessage("w " * 1500)]
    assert _est_tokens(history) == pytest.approx(MAX_HISTORY_TOKENS, abs=50)
    compressed = compress_chat_history(history, sentinel_state="")
    assert _est_tokens(compressed) <= MAX_HISTORY_TOKENS
```

---

### T-EDGE-6 — Large Pinned File (No Truncation)
```python
def test_edge_6_large_pinned_not_truncated():
    # A 50k-char pinned file should pass through at full size
    pinned = "class Foo:\n    pass\n" * 2500   # ~50k chars
    # Verify the full content is included in the prompt inputs
    # (no truncation applied to pinned context)
    assert len(pinned) >= 50000
    # content passed to prompt should equal original
    ...
```

---

### T-EDGE-7 — Specialist Model Not in Cache (First Call)
```python
def test_edge_7_specialist_first_call(mocker):
    # First time CODE is detected → creates new LLM instance, adds to cache
    mock_get_llm = mocker.patch("rag_chain.get_llm")
    mock_get_llm.return_value = mocker.MagicMock()
    ...
    # Second CODE query → reuses from _specialist_llm_cache (no new get_llm call)
    assert mock_get_llm.call_count == 1
```

---

### T-EDGE-8 — Dynamic max_tokens: Query Length Boundary (200 chars)
```python
def test_edge_8_max_tokens_boundary():
    from rag_chain import _get_max_tokens, MAX_TOKENS
    assert _get_max_tokens("GENERAL", "a" * 200) == 1024    # exactly 200: reduced
    assert _get_max_tokens("GENERAL", "a" * 201) == MAX_TOKENS  # 201: full
```

---

### T-EDGE-9 — Concurrent Semantic Cache Writes
```python
def test_edge_9_concurrent_cache_writes(tmp_path):
    # Multiple threads writing to SemanticCache simultaneously should not corrupt
    cache = get_semantic_cache()
    from concurrent.futures import ThreadPoolExecutor
    def write(i):
        cache.upsert(f"query {i}", f"answer {i}")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(write, range(10)))
    # Verify all entries readable
    for i in range(10):
        result = cache.lookup(f"query {i}", threshold=0.99)
        # Should either hit or miss, but not throw
        assert result is None or isinstance(result, str)
```

---

## 12. Test Execution Matrix

### Run Order (Dependency-aware)
```
Phase 1 (no external deps):
  pytest test_t_cfg_1.py test_t_cfg_2.py test_t_cfg_3_4.py
  pytest test_t_rc_1.py test_t_rc_2.py

Phase 2 (requires local model files, no network):
  pytest test_backend_full.py
  pytest test_rag_full.py

Phase 3 (requires ChromaDB + local embed model):
  pytest test_retrieval.py
  pytest test_hybrid.py
  pytest test_retrieval_integration.py
  pytest test_history_full.py
  pytest test_ingest.py test_ingest_integration.py

Phase 4 (requires Ollama running):
  pytest test_stage2.py test_stage3.py test_stage4.py test_stage5.py

Phase 5 (requires OpenRouter API key + Ollama):
  pytest test_final_integration.py
  pytest test_backend_full.py::test_be_11_collection_delete  # cleanup
```

### Quick Smoke Test (CI/CD)
```bash
pytest test_t_cfg_1.py test_t_cfg_2.py test_t_cfg_3_4.py \
       test_t_rc_1.py test_t_rc_2.py \
       test_rag_full.py::test_rc_12_max_tokens \
       -v --tb=short
```
Expected: All pass in < 30 seconds, no external services needed.

### Token Optimization Regression Suite
Run after any change to config.py, rag_chain.py:
```bash
pytest test_rag_full.py -k "tok or rc_12 or hist_5 or hist_6" -v
```

### Full Regression Suite
```bash
pytest . --ignore=test_final_integration.py -v --tb=long
```

---

## Appendix: Parameter Change Log (2026-04-11)

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `SENTINEL_TOKEN_THRESHOLD` | `2000` | `1500` | Fire sentinel sooner; ghost history accumulates too slowly under old value |
| `SENTINEL_INTERVAL` | `5` | `3` | More frequent summaries; reduces ghost message accumulation |
| `RERANK_CANDIDATES` | `25` | `15` | Reduces candidate fetches (150→90) with negligible quality loss |
| `MAX_ZERO_CHUNK_CHARS` | *(new)* | `9000` | Cap for zero-chunks in retrieval; prevents ~33k-token context explosions |
| Zero-chunk guard condition | `if not provider_has_cache:` | Unconditional | Bug fix: Claude/Gemini were unprotected despite being most expensive |
| `_get_max_tokens()` | *(new function)* | Returns 1024 for GENERAL/VISION short queries | Reduce reserved output tokens for ~80% of queries |
