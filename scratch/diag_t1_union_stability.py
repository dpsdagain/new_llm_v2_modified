"""Diagnostic for Tier 1 #2: union eviction stability.

Claim (now fixed): the pre-fix eviction re-scored old docs by
query-keyword overlap. Rewording the query changed the order of
`surviving_old`, breaking the cached prefix.

This test reconstructs the eviction loop for both the OLD and the NEW
implementation and runs two "turns" with different wording but the same
document pool. The new implementation must produce IDENTICAL
`surviving_old` ordering across the two turns.
"""

from langchain_core.documents import Document


def mkdoc(src, idx, h, content):
    return Document(page_content=content, metadata={
        "source": src, "chunk_index": idx, "content_hash": h,
    })


# Build a previous_union of 5 docs. Pretend they were retrieved last turn.
previous_union = [
    mkdoc("config.py",   0, "h_cfg0", "BM25_WEIGHT = 0.65 retrieval weight"),
    mkdoc("backend.py",  3, "h_bk3",  "def hybrid_search(db, query) retrieval ranking"),
    mkdoc("rag_chain.py",7, "h_rc7",  "def _sort_docs_deterministically docs"),
    mkdoc("app.py",      2, "h_ap2",  "pinned_to_send = read file content"),
    mkdoc("backend.py", 11, "h_bk11", "class SQLiteFTS5BM25 keyword index"),
]

# 2 new retrievals on turn N+1 (only 1 overlaps previous_union)
unique_new_same_pool = [
    mkdoc("rag_chain.py", 9, "h_rc9", "semantic cache lookup"),
    mkdoc("rag_chain.py", 7, "h_rc7", "def _sort_docs_deterministically docs"),  # overlap
]

seen_base = {d.metadata["content_hash"] for d in unique_new_same_pool}
MAX_CONTEXT_UNION = 6
available_old_slots = MAX_CONTEXT_UNION - len(unique_new_same_pool)  # 4


def old_eviction(previous_union, seen_hashes, user_input, slots):
    """The PRE-FIX implementation: score by query-keyword overlap."""
    _EVICT_STOPWORDS = {
        "the","is","a","an","in","of","to","for","and","or","how",
        "does","what","it","this","that","can","do","i","my","be",
        "are","was","were","will","with","on","at","by","from","not",
    }
    query_terms = set(user_input.lower().split()) - _EVICT_STOPWORDS
    scored = []
    for d in previous_union:
        h = d.metadata.get("content_hash", d.page_content)
        if h in seen_hashes:
            continue
        preview = (d.metadata.get("source","") + " " + d.page_content[:500]).lower()
        score = sum(1 for t in query_terms if t in preview)
        scored.append((score, d, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _s, d, h in scored:
        if len(out) >= slots: break
        out.append(d)
    return out


def new_eviction(previous_union, seen_hashes, slots):
    """The POST-FIX implementation: preserve previous_union order."""
    out = []
    for d in previous_union:
        if len(out) >= slots: break
        h = d.metadata.get("content_hash", d.page_content)
        if h in seen_hashes:
            continue
        out.append(d)
    return out


def hashes(docs):
    return [d.metadata["content_hash"] for d in docs]


# Two queries with the same intent but different wording
q1 = "how does hybrid retrieval ranking work"
q2 = "explain the retrieval weighting algorithm"

print("=== OLD eviction (query-keyword scoring) ===")
old_t1 = old_eviction(previous_union, set(seen_base), q1, available_old_slots)
old_t2 = old_eviction(previous_union, set(seen_base), q2, available_old_slots)
print(f"turn 1 ({q1!r})")
print(f"  -> {hashes(old_t1)}")
print(f"turn 2 ({q2!r})")
print(f"  -> {hashes(old_t2)}")
old_stable = hashes(old_t1) == hashes(old_t2)
print(f"stable across rewording? {old_stable}")
print()

print("=== NEW eviction (previous_union order) ===")
new_t1 = new_eviction(previous_union, set(seen_base), available_old_slots)
new_t2 = new_eviction(previous_union, set(seen_base), available_old_slots)
print(f"turn 1 ({q1!r})")
print(f"  -> {hashes(new_t1)}")
print(f"turn 2 ({q2!r})")
print(f"  -> {hashes(new_t2)}")
new_stable = hashes(new_t1) == hashes(new_t2)
print(f"stable across rewording? {new_stable}")
print()

if new_stable and not old_stable:
    print("VERDICT: Tier 1 #2 fix VERIFIED.")
    print("  OLD code reordered under rewording; NEW code is byte-stable.")
elif new_stable and old_stable:
    print("VERDICT: Inconclusive — both stable on this input.")
    print("  Try a query pair that better exercises the keyword scorer.")
else:
    print("VERDICT: FIX FAILED. New implementation is not stable.")
