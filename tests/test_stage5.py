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

def test_stage5():
    # Stage 5 question testing agentic reasoning about architecture
    query = "If I change RETRIEVER_K in config.py to 12, how does that affect the RRF fusion logic and the reranker candidates in rag_chain.py?"
    print(f"--- 🧬 Stage 5 FINAL Benchmark: Cross-File Reasoning ---")
    
    # 1. Load DB (v2_atomic) and chain
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
    # The answer should mention that it increases the pool of candidates for RRF/Reranking.
    # Qwen Turbo is excellent at this.
    if "12" in full_answer and ("reranker" in full_answer.lower() or "rerank" in full_answer.lower()):
        print("✅ STAGE 5 PASSED.")
    else:
        print("❌ STAGE 5 FAILED (Insufficient architectural reasoning).")

if __name__ == "__main__":
    test_stage5()
