import sys
import os
import io

# Write output to a UTF-8 log file
log_file = io.open('shadow_test_report.txt', 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

# Allow imports from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend import load_and_chunk_codebase, ingest_into_chroma, load_existing_chroma
from rag_chain import build_rag_chain
from langchain_core.messages import HumanMessage

async def main():
    test_dir = r'f:\Gemini_anti\new_llm_v3\new_llm_v2_modified\test_shadow_env'
    coll_name = "shadow_test_coll"
    model_id = "openai/gpt-oss-120b:free"
    
    print(f"--- 1. Ingesting {test_dir} ---")
    chunks = load_and_chunk_codebase(test_dir)
    print(f"Found {len(chunks)} chunks.")
    
    db, added = ingest_into_chroma(chunks, collection_name=coll_name)
    print(f"Ingested {added} chunks into {coll_name}.")
    
    print(f"--- 2. Initializing RAG Chain with {model_id} ---")
    chain = build_rag_chain(db, model=model_id)
    
    query = "I'm seeing immediate failures in network_manager.py. Why isn't the retry logic working?"
    print(f"Query: {query}")
    
    print("--- 3. Running Query ---")
    # Wrap in a loop to handle streaming output
    full_answer = ""
    # We use invoke/stream logic from rag_chain. 
    # build_rag_chain returns a RunnableLambda.
    
    # We need to simulate the inputs passed by app.py
    inputs = {
        "input": query,
        "chat_history": [],
        "full_history": [],
        "global_turn_count": 1,
        "full_source_context": "None pinned.",
        "collection_name": coll_name,
        "sentinel_state": "No summary generated yet."
    }
    
    # We'll use the stream to see the reasoning
    for chunk in chain.stream(inputs):
        if "answer" in chunk:
            full_answer += chunk["answer"]
            # Print answer in real-time
            print(chunk["answer"], end="", flush=True)
        if "context" in chunk:
            print("\n--- RETRIEVED CHUNKS ---")
            for i, d in enumerate(chunk["context"]):
                print(f"[{i}] {d.metadata.get('source')}")
            print("--- END CONTEXT ---\n")

    print("\n--- FINAL ANALYSIS ---")
    if "FAIL_FAST" in full_answer and "settings.py" in full_answer.lower():
        print("✅ SUCCESS: The system successfully linked the bug to the hidden configuration!")
    else:
        print("❌ FAILURE: The system missed the link to the hidden configuration.")

if __name__ == "__main__":
    asyncio.run(main())
