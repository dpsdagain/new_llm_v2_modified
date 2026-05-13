import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import os
from config import CHROMA_DB_DIR

def debug_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    colls = client.list_collections()
    print(f"Collections: {[c.name for c in colls]}")
    
    for coll_name in [c.name for c in colls]:
        print(f"\n--- Collection: {coll_name} ---")
        coll = client.get_collection(coll_name)
        count = coll.count()
        print(f"Count: {count}")
        if count > 0:
            results = coll.get(limit=100, include=["metadatas"])
            metas = results.get("metadatas", [])
            sources = {m.get("source") for m in metas if m}
            print(f"Unique sources (first 100 chunks): {sources}")
            
            # Check for rag_chain.py specifically
            all_sources = set()
            offset = 0
            limit = 100
            while offset < count:
                batch = coll.get(limit=limit, offset=offset, include=["metadatas"])
                for m in batch.get("metadatas", []):
                    if m and "source" in m:
                        all_sources.add(m["source"])
                offset += limit
            print(f"Total unique sources: {len(all_sources)}")
            print(f"Is rag_chain.py in sources? {any('rag_chain.py' in s for s in all_sources)}")
            print(f"Is config.py in sources? {any('config.py' in s for s in all_sources)}")

if __name__ == "__main__":
    debug_db()
