import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Diagnostic script: Trace an L4 query through every pipeline stage.
Prints what survives at each step to find where config.py disappears.
"""
import os, sys, json

# Ensure project imports work
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CHROMA_DB_DIR, ZERO_CHUNK_THRESHOLD, MAX_ZERO_CHUNK_CHARS,
    RERANK_TOP_K, BM25_WEIGHT, VECTOR_WEIGHT, EMBEDDING_MODEL_NAME,
)

L4_QUERY = "How does ENABLE_PROMPT_CACHING propagate step-by-step through the codebase and alter the final API payload?"

print("=" * 70)
print("L4 DIAGNOSTIC TRACE")
print("=" * 70)
print(f"Query: {L4_QUERY}")
print(f"ZERO_CHUNK_THRESHOLD: {ZERO_CHUNK_THRESHOLD}")
print(f"MAX_ZERO_CHUNK_CHARS: {MAX_ZERO_CHUNK_CHARS}")
print(f"RERANK_TOP_K: {RERANK_TOP_K}")
print()

# ── STAGE 0: What was ingested? ──────────────────────────────────────
print("=" * 70)
print("STAGE 0: INGESTION — What's in ChromaDB?")
print("=" * 70)

from backend import load_existing_chroma, SQLiteFTS5BM25, get_embedding_model
# Auto-detect collection name
import chromadb
_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collections = [c for c in _client.list_collections() if c.name != "semantic_cache"]
if not _collections:
    print("ERROR: No ChromaDB collection found. Run ingestion first.")
    sys.exit(1)
COLL_NAME = _collections[0].name
print(f"Using collection: {COLL_NAME} ({_collections[0].count()} chunks)")

db = load_existing_chroma(COLL_NAME)
if db is None:
    print("ERROR: Could not load ChromaDB. Run ingestion first.")
    sys.exit(1)

all_data = db.get(include=["metadatas", "documents"])
docs_meta = list(zip(all_data["documents"], all_data["metadatas"]))
print(f"Total chunks in ChromaDB: {len(docs_meta)}")

# Find config.py chunks
config_chunks = [(d, m) for d, m in docs_meta if m.get("source", "").endswith("config.py")]
print(f"config.py chunks: {len(config_chunks)}")
for d, m in config_chunks:
    print(f"  - zero_chunk={m.get('zero_chunk')}, len={len(d)}, "
          f"has_calls={bool(m.get('calls_functions'))}, "
          f"has_consts={bool(m.get('references_constants'))}, "
          f"hash={m.get('content_hash', '?')[:12]}")
    # Check if ENABLE_PROMPT_CACHING is in the content
    if "ENABLE_PROMPT_CACHING" in d:
        print(f"    [YES] Contains 'ENABLE_PROMPT_CACHING'")
    else:
        print(f"    [NO] Does NOT contain 'ENABLE_PROMPT_CACHING'")

# Find chunks that reference ENABLE_PROMPT_CACHING in metadata
ref_chunks = [(d, m) for d, m in docs_meta
              if "ENABLE_PROMPT_CACHING" in m.get("references_constants", "")]
print(f"\nChunks with ENABLE_PROMPT_CACHING in references_constants metadata: {len(ref_chunks)}")
for d, m in ref_chunks:
    src = os.path.basename(m.get("source", "?"))
    print(f"  - {src} chunk_index={m.get('chunk_index')} "
          f"refs={m.get('references_constants', '')[:80]}")

# Find chunks that contain ENABLE_PROMPT_CACHING in text
text_chunks = [(d, m) for d, m in docs_meta if "ENABLE_PROMPT_CACHING" in d]
print(f"\nChunks with ENABLE_PROMPT_CACHING in page_content: {len(text_chunks)}")
for d, m in text_chunks:
    src = os.path.basename(m.get("source", "?"))
    print(f"  - {src} chunk_index={m.get('chunk_index')} zero={m.get('zero_chunk')} len={len(d)}")

# ── STAGE 1: BM25 (FTS5) — Full query ──────────────────────────────
print()
print("=" * 70)
print("STAGE 1a: BM25 — Full NL query")
print("=" * 70)

fts = SQLiteFTS5BM25(COLL_NAME)
clean_query = "".join(c if c.isalnum() or c.isspace() else " " for c in L4_QUERY)
print(f"Cleaned query: {clean_query}")
print(f"FTS5 requires ALL these terms (implicit AND):")

bm25_full = fts.search(L4_QUERY, k=36)
print(f"BM25 results (full query): {len(bm25_full)}")
config_in_bm25 = [d for d in bm25_full if d.metadata.get("source", "").endswith("config.py")]
print(f"  config.py in results: {len(config_in_bm25)}")
for i, d in enumerate(bm25_full[:5]):
    src = os.path.basename(d.metadata.get("source", "?"))
    print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')} len={len(d.page_content)}")

# ── STAGE 1b: BM25 — Anchor-only query (Fix J) ─────────────────────
print()
print("=" * 70)
print("STAGE 1b: BM25 — Anchor-only query (Fix J)")
print("=" * 70)

anchor = "ENABLE_PROMPT_CACHING"
anchor_query = anchor.replace("_", " ")
print(f"Anchor query: '{anchor_query}'")

bm25_anchor = fts.search(anchor_query, k=5)
print(f"BM25 results (anchor only): {len(bm25_anchor)}")
config_in_anchor = [d for d in bm25_anchor if d.metadata.get("source", "").endswith("config.py")]
print(f"  config.py in results: {len(config_in_anchor)}")
for i, d in enumerate(bm25_anchor):
    src = os.path.basename(d.metadata.get("source", "?"))
    has_flag = "ENABLE_PROMPT_CACHING" in d.page_content
    print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')} len={len(d.page_content)} has_flag={has_flag}")

# ── STAGE 1c: FTS5 search_by_constant (Fix G) ──────────────────────
print()
print("=" * 70)
print("STAGE 1c: FTS5 search_by_constant (Fix G)")
print("=" * 70)

const_docs = fts.search_by_constant(anchor, k=10)
print(f"search_by_constant('{anchor}'): {len(const_docs)} results")
for i, d in enumerate(const_docs):
    src = os.path.basename(d.metadata.get("source", "?"))
    print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')} "
          f"consts={d.metadata.get('references_constants', '')[:60]}")

# ── STAGE 1d: FTS5 search_by_call (Fix E) ──────────────────────────
print()
print("=" * 70)
print("STAGE 1d: FTS5 search_by_call (Fix E)")
print("=" * 70)

call_docs = fts.search_by_call(anchor, k=10)
print(f"search_by_call('{anchor}'): {len(call_docs)} results")
for i, d in enumerate(call_docs):
    src = os.path.basename(d.metadata.get("source", "?"))
    print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')}")

# ── STAGE 2: Vector Search ──────────────────────────────────────────
print()
print("=" * 70)
print("STAGE 2: Vector Search (ChromaDB similarity)")
print("=" * 70)

vector_docs = db.similarity_search(L4_QUERY, k=36)
print(f"Vector results: {len(vector_docs)}")
config_in_vector = [d for d in vector_docs if d.metadata.get("source", "").endswith("config.py")]
print(f"  config.py in results: {len(config_in_vector)}")
for i, d in enumerate(vector_docs[:5]):
    src = os.path.basename(d.metadata.get("source", "?"))
    has_flag = "ENABLE_PROMPT_CACHING" in d.page_content
    print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')} has_flag={has_flag}")

# ── STAGE 3: Reranker ───────────────────────────────────────────────
print()
print("=" * 70)
print("STAGE 3: Cross-encoder Reranker")
print("=" * 70)

from rag_chain import get_reranker
reranker = get_reranker()

# Simulate the merged pool: main hybrid + Fix J anchor + Fix G constants
merged_pool = list(bm25_full) + list(bm25_anchor) + list(const_docs)
# Deduplicate
seen = set()
deduped = []
for d in merged_pool:
    h = d.metadata.get("content_hash", d.page_content[:200])
    if h not in seen:
        deduped.append(d)
        seen.add(h)

print(f"Merged pool (deduped): {len(deduped)}")
config_in_pool = [d for d in deduped if d.metadata.get("source", "").endswith("config.py")]
print(f"  config.py in merged pool: {len(config_in_pool)}")

if reranker and reranker.model:
    reranked = reranker.rerank(L4_QUERY, deduped, top_k=RERANK_TOP_K)
    print(f"After reranking (top_k={RERANK_TOP_K}): {len(reranked)}")
    config_in_reranked = [d for d in reranked if d.metadata.get("source", "").endswith("config.py")]
    print(f"  config.py survived reranking: {len(config_in_reranked)}")
    for i, d in enumerate(reranked):
        src = os.path.basename(d.metadata.get("source", "?"))
        has_flag = "ENABLE_PROMPT_CACHING" in d.page_content
        print(f"  [{i}] {src} chunk={d.metadata.get('chunk_index')} has_flag={has_flag} len={len(d.page_content)}")
else:
    print("  Reranker not loaded — skipping")
    reranked = deduped[:RERANK_TOP_K]

# ── STAGE 4: Zero-chunk filter ──────────────────────────────────────
print()
print("=" * 70)
print("STAGE 4: Zero-chunk filter (MAX_ZERO_CHUNK_CHARS={})".format(MAX_ZERO_CHUNK_CHARS))
print("=" * 70)

filtered = [
    d for d in reranked
    if not (d.metadata.get("zero_chunk") and len(d.page_content) > MAX_ZERO_CHUNK_CHARS)
]
removed = len(reranked) - len(filtered)
print(f"Removed by zero-chunk filter: {removed}")
config_in_final = [d for d in filtered if d.metadata.get("source", "").endswith("config.py")]
print(f"config.py in final context: {len(config_in_final)}")

# ── STAGE 5: Fix H post-reranker injection simulation ───────────────
print()
print("=" * 70)
print("STAGE 5: Fix H post-reranker injection")
print("=" * 70)

reranked_hashes = {d.metadata.get("content_hash", d.page_content[:200]) for d in filtered}
injected_count = 0
for d in deduped:
    h = d.metadata.get("content_hash", d.page_content[:200])
    if h in reranked_hashes:
        continue
    if anchor in d.page_content:
        is_zero = d.metadata.get("zero_chunk", False)
        ref_consts = d.metadata.get("references_constants", "")
        if is_zero or anchor in ref_consts:
            src = os.path.basename(d.metadata.get("source", "?"))
            print(f"  Would inject: {src} chunk={d.metadata.get('chunk_index')} zero={is_zero} len={len(d.page_content)}")
            injected_count += 1
            if injected_count >= 2:
                break

if injected_count == 0:
    print("  Nothing to inject — config.py not in pre-rerank pool OR already in final")

print()
print("=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print(f"config.py in ChromaDB:          {len(config_chunks) > 0}")
print(f"config.py found by BM25 full:   {len(config_in_bm25) > 0}")
print(f"config.py found by BM25 anchor: {len(config_in_anchor) > 0}")
print(f"config.py found by FTS5 const:  N/A (config.py is zero-chunk, no constants metadata)")
print(f"config.py found by vector:      {len(config_in_vector) > 0}")
print(f"config.py survived reranking:   {len(config_in_reranked) > 0 if reranker else 'N/A'}")
print(f"config.py in final context:     {len(config_in_final) > 0}")
print()
print("FIRST STAGE WHERE config.py DISAPPEARS = the bug location")
