"""Diagnostic for Tier 1 #1: rag_chain.py:1466 `stable_hashes=None`.

Claim: passing `stable_hashes=None` to `_sort_docs_deterministically` on the
`established_docs` list (pre-filtered to contain only docs whose hash IS in
stable_hashes) breaks prefix-cache stability across turns.

Counter-analysis: in `_sort_docs_deterministically`, `is_new` is constant
across the list because every element is either in-set or not. A constant
primary key doesn't affect ordering. If the input list and its content
are identical between turns, the output is identical regardless of
`stable_hashes`.

This script proves one side or the other by calling the actual function
with real document lists.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag_chain import _sort_docs_deterministically


def mkdoc(src, idx, h, content):
    return Document(page_content=content, metadata={
        "source": src, "chunk_index": idx, "content_hash": h,
    })


# Simulate 4 established docs across two turns with same content.
docs_t1 = [
    mkdoc("b.py", 2, "h_b2", "beta chunk two"),
    mkdoc("a.py", 1, "h_a1", "alpha chunk one"),
    mkdoc("c.py", 0, "h_c0", "gamma chunk zero"),
    mkdoc("a.py", 0, "h_a0", "alpha chunk zero"),
]
docs_t2 = list(docs_t1)  # same content, may be in different initial order
import random
random.seed(42)
random.shuffle(docs_t2)

stable_hashes = {"h_b2", "h_a1", "h_c0", "h_a0"}

# Case A: current code — stable_hashes=None (the alleged bug)
t1_A = _sort_docs_deterministically(docs_t1, stable_hashes=None)
t2_A = _sort_docs_deterministically(docs_t2, stable_hashes=None)

# Case B: proposed fix — stable_hashes=stable_hashes
t1_B = _sort_docs_deterministically(docs_t1, stable_hashes=stable_hashes)
t2_B = _sort_docs_deterministically(docs_t2, stable_hashes=stable_hashes)


def fmt(docs):
    return [(d.metadata["source"], d.metadata["chunk_index"], d.metadata["content_hash"]) for d in docs]


print("=== ESTABLISHED DOCS (all hashes in stable_hashes) ===")
print("Case A (current, stable_hashes=None):")
print(f"  turn 1: {fmt(t1_A)}")
print(f"  turn 2: {fmt(t2_A)}")
print(f"  stable across turns? {fmt(t1_A) == fmt(t2_A)}")
print()
print("Case B (proposed, stable_hashes=set):")
print(f"  turn 1: {fmt(t1_B)}")
print(f"  turn 2: {fmt(t2_B)}")
print(f"  stable across turns? {fmt(t1_B) == fmt(t2_B)}")
print()
print(f"Case A == Case B on turn 1? {fmt(t1_A) == fmt(t1_B)}")
print(f"Case A == Case B on turn 2? {fmt(t2_A) == fmt(t2_B)}")
print()

# Now test with NEW docs (none in stable_hashes) — simulating the `new_docs` list
new_docs_t1 = [
    mkdoc("x.py", 1, "h_x1", "x one"),
    mkdoc("y.py", 0, "h_y0", "y zero"),
]
new_docs_t2 = list(reversed(new_docs_t1))

nt1_A = _sort_docs_deterministically(new_docs_t1, stable_hashes=None)
nt2_A = _sort_docs_deterministically(new_docs_t2, stable_hashes=None)
nt1_B = _sort_docs_deterministically(new_docs_t1, stable_hashes=stable_hashes)
nt2_B = _sort_docs_deterministically(new_docs_t2, stable_hashes=stable_hashes)

print("=== NEW DOCS (no hashes in stable_hashes) ===")
print(f"Case A turn 1: {fmt(nt1_A)}")
print(f"Case A turn 2: {fmt(nt2_A)}")
print(f"Case B turn 1: {fmt(nt1_B)}")
print(f"Case B turn 2: {fmt(nt2_B)}")
print(f"Case A == Case B? {fmt(nt1_A) == fmt(nt1_B) and fmt(nt2_A) == fmt(nt2_B)}")
print()

# ================================================================
# VERDICT
# ================================================================
est_same = fmt(t1_A) == fmt(t1_B) and fmt(t2_A) == fmt(t2_B)
new_same = fmt(nt1_A) == fmt(nt1_B) and fmt(nt2_A) == fmt(nt2_B)

if est_same and new_same:
    print("VERDICT: Tier 1 #1 is a FALSE CLAIM.")
    print("  Passing stable_hashes=None vs =set produces identical output")
    print("  because the input lists are pre-split to be pure (all in-set or")
    print("  all out-of-set). is_new is constant, so it doesn't affect order.")
    print("  The fix is a no-op. The real cache-breaker lives elsewhere.")
else:
    print("VERDICT: Tier 1 #1 is a REAL bug. Output differs:")
    print(f"  established differs: {not est_same}")
    print(f"  new differs: {not new_same}")
