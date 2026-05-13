import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
from langchain_core.messages import HumanMessage
from rag_chain import build_rag_chain
from config import ENABLE_HYBRID_SEARCH

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_hybrid_flow():
    query = "What is the specific value of ZERO_CHUNK_THRESHOLD in config.py"
    print(f"--- 🧬 Testing HYBRID Retrieval for: '{query}' ---")
    print(f"Hybrid Search Status: {ENABLE_HYBRID_SEARCH}")
    
    # 1. Load DB
    from langchain_chroma import Chroma
    from backend import get_embedding_model, CHROMA_DB_DIR
    embeddings = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="default"
    )
    
    # 2. Build chain (this handles the routing and search logic)
    chain = build_rag_chain(db=db)
    
    # Simulate the inputs seen in app.py
    inputs = {
        "input": query,
        "chat_history": [],
        "last_query": None,
        "last_query_embedding": None,
        "force_retrieval": False
    }
    
    # We'll use the 'search_only' logic or just run the chain and check context
    # In rag_chain.py, the final context is in 'docs'
    context_docs = []
    
    # Since the chain is normally streamed, we can iterate
    try:
        for chunk in chain.stream(inputs):
            if isinstance(chunk, dict) and "context" in chunk:
                context_docs = chunk["context"]
                break
    except Exception as e:
        print(f"Error running chain: {e}")
        return

    if not context_docs:
        print("❌ No documents retrieved by the chain.")
        return

    print(f"--- 📜 Context Chunks Retrieved ({len(context_docs)}) ---")
    found_config = False
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get("source", "Unknown")
        print(f"[{i}] {source}")
        if "config.py" in source and "ZERO_CHUNK_THRESHOLD" in doc.page_content:
            found_config = True
            print("   ✨ SUCCESS: Found relevant config.py chunk!")
            
    if not found_config:
        print("❌ FAILURE: Even with Hybrid Search, config.py was missed.")
    else:
        print("✅ STAGE 1 RETRIEVAL PASSED.")

if __name__ == "__main__":
    test_hybrid_flow()
