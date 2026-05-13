"""Diagnostic for Tier 1 #1 (v5): Semantic cache scope.

Proves the fix: after the change, the cache MUST reject a hit when pinned
content or model changes, even for the same query.

Without a real embedding model this can't run end-to-end, so we monkey-
patch the ChromaDB call with a stub that returns a perfect-score match.
That isolates the scope-check logic from the embedding layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import SemanticCache


class _StubDoc:
    def __init__(self, metadata): self.metadata = metadata


def make_cache_with_stub(stored_meta):
    """Return a SemanticCache whose db.similarity_search returns ONE perfect hit."""
    # Construct without triggering ChromaDB init
    cache = SemanticCache.__new__(SemanticCache)
    class _StubDB:
        def similarity_search_with_relevance_scores(self, q, k=3):
            return [(_StubDoc(stored_meta), 0.999)]
        def add_texts(self, texts, metadatas):  # unused in lookup tests
            pass
    cache.db = _StubDB()
    return cache


# Baseline: entry stored with file A, model X, collection K
stored_meta = {
    "answer": "CACHED ANSWER FOR FILE A / MODEL X",
    "source_collection": "K",
    "pinned_fp": SemanticCache._pinned_fingerprint("FILE A CONTENT"),
    "model_fp": SemanticCache._model_fingerprint("claude-sonnet"),
}
cache = make_cache_with_stub(stored_meta)

print("=== Cache scope enforcement ===")
# 1) Same pinned, same model, same collection → HIT
r1 = cache.lookup("q", threshold=0.98, collection_scope="K",
                  pinned_content="FILE A CONTENT", model="claude-sonnet")
print(f"[1] same everything        → {r1!r}")
assert r1 == stored_meta["answer"], "expected hit"

# 2) Different pinned file content → MISS
r2 = cache.lookup("q", threshold=0.98, collection_scope="K",
                  pinned_content="FILE B CONTENT", model="claude-sonnet")
print(f"[2] different pinned file  → {r2!r}")
assert r2 is None, "expected miss on pinned-fp mismatch"

# 3) Different model → MISS
r3 = cache.lookup("q", threshold=0.98, collection_scope="K",
                  pinned_content="FILE A CONTENT", model="deepseek-chat")
print(f"[3] different model        → {r3!r}")
assert r3 is None, "expected miss on model-fp mismatch"

# 4) Different collection → MISS
r4 = cache.lookup("q", threshold=0.98, collection_scope="OTHER",
                  pinned_content="FILE A CONTENT", model="claude-sonnet")
print(f"[4] different collection   → {r4!r}")
assert r4 is None, "expected miss on collection mismatch"

# 5) Unpinned on both sides → HIT (regression: the "no pin" bucket works)
stored_unpinned = dict(stored_meta,
                       pinned_fp=SemanticCache._pinned_fingerprint(None))
cache_unpinned = make_cache_with_stub(stored_unpinned)
r5 = cache_unpinned.lookup("q", threshold=0.98, collection_scope="K",
                           pinned_content=None, model="claude-sonnet")
print(f"[5] unpinned→unpinned      → {r5!r}")
assert r5 == stored_meta["answer"], "expected hit when both sides unpinned"

# 6) Legacy entry (no pinned_fp/model_fp fields) → HIT (backward-compat)
legacy = {"answer": "LEGACY", "source_collection": "K"}
cache_legacy = make_cache_with_stub(legacy)
r6 = cache_legacy.lookup("q", threshold=0.98, collection_scope="K",
                         pinned_content="FILE A CONTENT", model="claude-sonnet")
print(f"[6] legacy entry, any pin  → {r6!r}")
assert r6 == "LEGACY", "expected backward-compat hit on legacy entry"

# 7) File edited → different fingerprint → MISS (stale-answer protection)
r7 = cache.lookup("q", threshold=0.98, collection_scope="K",
                  pinned_content="FILE A CONTENT EDITED", model="claude-sonnet")
print(f"[7] pinned file edited     → {r7!r}")
assert r7 is None, "expected miss after file edit"

print()
print("VERDICT: Tier 1 #1 fix VERIFIED. Cache correctly rejects mismatches")
print("         on pinned content, model, and collection — and still hits")
print("         on legacy entries / unpinned matches.")
