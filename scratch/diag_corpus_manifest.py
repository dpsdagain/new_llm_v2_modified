"""Diagnostic: corpus manifest formatter.

Proves _format_corpus_manifest emits a stable, complete file list for
the active collection, so meta-questions ("how many files", "list all")
have authoritative ground truth in the prompt.

Stability check: calling twice must produce byte-identical output (no
hidden time/random ordering).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import _format_corpus_manifest
from backend import list_collections, get_collection_info


def main():
    colls = list_collections() or []
    if not colls:
        print("no collections found — ingest something first")
        return

    print(f"collections found: {colls}\n")
    for c in colls:
        info = get_collection_info(c)
        print(f"=== collection: {c} ===")
        print(f"  raw count   : {info.get('count')}")
        print(f"  raw sources : {len(info.get('sources', []))} files")
        m1 = _format_corpus_manifest(c)
        m2 = _format_corpus_manifest(c)
        print(f"  stable      : {m1 == m2}")
        print(f"  manifest len: {len(m1)} chars (~{len(m1)//4} tokens)")
        print(f"--- manifest preview (first 1200 chars) ---")
        print(m1[:1200])
        if len(m1) > 1200:
            print(f"... [{len(m1)-1200} more chars]")
        print()


if __name__ == "__main__":
    main()
