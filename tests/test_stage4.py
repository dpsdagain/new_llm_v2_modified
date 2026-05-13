import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import build_rag_chain
from langchain_chroma import Chroma
from backend import get_embedding_model, CHROMA_DB_DIR

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stage4():
    print(f"--- 🧬 Stage 4 FINAL Benchmark: Multi-Turn Retrieval Memory ---")
    
    # 1. Load DB (v2_atomic) and chain
    embeddings = get_embedding_model()
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="v2_atomic"
    )
    chain = build_rag_chain(db=db)
    
    # ── Turn 1 ──────────────────────────────────────────────────────────
    query1 = "Give me a 1-sentence summary of what app.py does."
    print(f"\n[Turn 1] User: {query1}")
    
    inputs1 = {
        "input": query1,
        "chat_history": [],
        "last_query": None,
        "last_query_embedding": None,
        "force_retrieval": False
    }
    
    print("--- 🤖 Turn 1 Response ---")
    answer1 = ""
    for chunk in chain.stream(inputs1):
        if isinstance(chunk, dict) and "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            answer1 += chunk["answer"]
            
    # ── Turn 2 ──────────────────────────────────────────────────────────
    query2 = "Now, what is the very first import statement in that file?"
    print(f"\n\n[Turn 2] User: {query2}")
    
    # Simulate history passing
    chat_history = [
        HumanMessage(content=query1),
        AIMessage(content=answer1)
    ]
    
    inputs2 = {
        "input": query2,
        "chat_history": chat_history,
        "last_query": query1,
        "last_query_embedding": None,
        "force_retrieval": False
    }
    
    print("--- 🤖 Turn 2 Response ---")
    answer2 = ""
    for chunk in chain.stream(inputs2):
        if isinstance(chunk, dict) and "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            answer2 += chunk["answer"]
            
    print("\n-----------------")
    
    # Verification
    # app.py starts with "import streamlit as st" or similar
    if "import" in answer2.lower() or "streamlit" in answer2.lower() or "os" in answer2.lower():
        print("✅ STAGE 4 PASSED.")
    else:
        print("❌ STAGE 4 FAILED (Memory loss or poor retrieval).")

if __name__ == "__main__":
    test_stage4()
