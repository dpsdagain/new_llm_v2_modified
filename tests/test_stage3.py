import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
from langchain_core.messages import HumanMessage
from rag_chain import build_rag_chain
from langchain_chroma import Chroma
from backend import get_embedding_model, CHROMA_DB_DIR

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stage3():
    # Stage 3 question testing multi-file synthesis
    query = "How does the Sentinel determine when to trigger a summary, and what are the specific token and turn thresholds in config.py?"
    print(f"--- 🧬 Stage 3 FINAL Benchmark: '{query}' ---")
    
    # 1. Load DB (using the fresh v2_atomic collection)
    embeddings = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="v2_atomic"
    )
    chain = build_rag_chain(db=db)
    
    inputs = {
        "input": query,
        "chat_history": [],
        "last_query": None,
        "last_query_embedding": None,
        "force_retrieval": False
    }
    
    print("--- 🤖 Result ---")
    full_answer = ""
    for chunk in chain.stream(inputs):
        if isinstance(chunk, dict) and "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            full_answer += chunk["answer"]
        elif isinstance(chunk, dict) and "context" in chunk:
             print(f"\n[Context sources: {[d.metadata.get('source') for d in chunk['context']]}]\n")
            
    print("\n-----------------")
    
    # Verification
    # Expected: SENTINEL_INTERVAL=5, SENTINEL_TOKEN_THRESHOLD=2000
    if "2000" in full_answer and "5" in full_answer:
        print("✅ STAGE 3 PASSED.")
    else:
        print("❌ STAGE 3 FAILED (Inaccurate synthesis).")

if __name__ == "__main__":
    test_stage3()
