import os
import sys
from langchain_chroma import Chroma
from backend import get_embedding_model
from config import CHROMA_DB_DIR

def debug_detailed():
    print("--- 🔍 DETAILED CHROMADB DEBUG ---")
    embedding = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding,
        collection_name="default",
    )
    data = db.get(include=["metadatas"])
    sources = set()
    for meta in data.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    
    print(f"Total Sources: {len(sources)}")
    for s in sorted(sources):
        print(f" - {s}")
    
    # Check for "zero-chunking threshold" keywords specifically
    print("\n--- 🔎 KEYWORD SEARCH TEST ---")
    results = db.similarity_search("zero-chunking threshold", k=5)
    print(f"Found {len(results)} matches for 'zero-chunking threshold'")
    for i, res in enumerate(results):
        print(f"Match {i}: {res.metadata.get('source')} (Score logic: MMR)")

if __name__ == "__main__":
    debug_detailed()
