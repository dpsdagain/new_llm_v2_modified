import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import traceback
from langchain_core.messages import HumanMessage
from rag_chain import build_rag_chain
from langchain_chroma import Chroma
from backend import get_embedding_model, CHROMA_DB_DIR

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stage2():
    query = "Which function in backend.py is called by app.py to handle background ingestion?"
    print(f"--- 🧬 Stage 2 Benchmark: '{query}' ---")
    
    try:
        # 1. Load DB & Chain
        embeddings = get_embedding_model()
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name="default"
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
        if "ingest_into_chroma" in full_answer or "AsyncIngestionTask" in full_answer:
            print("✅ STAGE 2 PASSED.")
        else:
            print("❌ STAGE 2 FAILED (Insufficient reasoning).")
            
    except Exception as e:
        print("\n❌ CRITICAL ERROR DURING BENCHMARK:")
        print(str(e))
        traceback.print_exc()

if __name__ == "__main__":
    test_stage2()
