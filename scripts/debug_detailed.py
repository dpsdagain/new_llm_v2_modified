import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from config import CHROMA_DB_DIR

def debug_detailed():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    coll = client.get_collection("default")
    count = coll.count()
    print(f"Total chunks in 'default': {count}")
    
    results = coll.get(include=["metadatas", "documents"])
    ids = results.get("ids", [])
    metas = results.get("metadatas", [])
    docs = results.get("documents", [])
    
    files_info = {}
    for did, m, d in zip(ids, metas, docs):
        src = m.get("source", "Unknown")
        if src not in files_info:
            files_info[src] = []
        files_info[src].append({
            "id": did,
            "zero_chunk": m.get("zero_chunk"),
            "len": len(d),
            "chunk_index": m.get("chunk_index")
        })
    
    print("\nFile Statistics:")
    for src, chunks in files_info.items():
        fname = src.split("\\")[-1]
        chunk_count = len(chunks)
        is_zc = all(c["zero_chunk"] for c in chunks)
        max_len = max(c["len"] for c in chunks)
        print(f"File: {fname:25} | Chunks: {chunk_count:3} | ZeroChunk: {str(is_zc):5} | MaxLen: {max_len:7}")

if __name__ == "__main__":
    debug_detailed()
