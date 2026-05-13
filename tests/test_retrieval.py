import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
from langchain_chroma import Chroma
from backend import get_embedding_model

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_exact_retrieval():
    query = "What is the specific value of ZERO_CHUNK_THRESHOLD in config.py"
    
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    embeddings = get_embedding_model()
    
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="default"
    )
    
    print(f"--- 🔍 Top 4 Retrieval for: '{query}' ---")
    results = db.similarity_search_with_relevance_scores(query, k=4)
    
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "Unknown")
        c_idx = doc.metadata.get("chunk_index", "?")
        print(f"[{i}] Score: {score:.4f} | Source: {source} | Chunk: {c_idx}")
        if "ZERO_CHUNK_THRESHOLD" in doc.page_content:
            print("   ✨ KEYWORD FOUND")
        else:
            print("   ❌ NOT FOUND")

if __name__ == "__main__":
    test_exact_retrieval()
