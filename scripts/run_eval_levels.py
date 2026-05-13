import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import asyncio
import time
import config
import backend
import rag_chain

async def run_eval():
    coll_name = f"eval_levels_col_{int(time.time())}"
    model_id = "google/gemini-2.0-flash-001"
    
    # Aggressive exclusion of the test script itself
    script_name = os.path.basename(__file__)
    config.EXCLUDED_FILE_PATTERNS.append(script_name)
    backend.EXCLUDED_FILE_PATTERNS.append(script_name)
    
    print(f"--- 🚀 RUNNING EVALUATION LEVELS WITH {model_id} ---")
    
    # 1. SETUP: Ingest codebase
    backend.delete_collection(coll_name)
    print("Ingesting codebase...")
    docs = backend.load_and_chunk_codebase(os.path.dirname(__file__))
    db, added = backend.ingest_into_chroma(docs, coll_name)
    print(f"Ingested {added} documents into {coll_name}.")

    queries = [
        {
            "level": "Level 1: The Execution Flow Test (2-Hop)",
            "query": "Trace the exact execution path of a user's chat message from the moment they hit enter in app.py until the final LLM token is streamed back to the UI. Name every specific custom function and file the text payload passes through."
        },
        {
            "level": "Level 2: The Blast Radius Test (Reverse Dependency)",
            "query": "If I were to rename the _content_hash function in backend.py to generate_sha256_hash and change its return type, which other specific files and functions would break, and why?"
        },
        {
            "level": "Level 3: The Scattered Logic Test (Concept Aggregation)",
            "query": "List every distinct place and method in this entire codebase where token counting or token estimation is performed. Explain how the logic differs in each location."
        },
        {
            "level": "Level 4: The Invisible Bridge Test (3-Hop Config Propagation)",
            "query": "Explain step-by-step how changing ENABLE_PROMPT_CACHING from True to False in config.py ultimately alters the final JSON payload sent to the Claude API. Mention the intermediate functions that evaluate this flag."
        }
    ]

    chain = rag_chain.build_rag_chain(db, model=model_id)

    for q in queries:
        print(f"\n\n{'='*80}")
        print(f"📌 {q['level']}")
        print(f"Query: {q['query']}")
        print(f"{'-'*80}")
        
        full_answer = ""
        inputs = {
            "input": q['query'],
            "chat_history": [],
            "collection_name": coll_name,
            "auto_specialist": False 
        }
        
        try:
            # Using synchronous stream() as per the chain requirement
            for chunk in chain.stream(inputs):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
                    full_answer += chunk["answer"]
        except Exception as e:
            print(f"\nERROR: {e}")

    # Cleanup
    print("\n\nCleaning up...")
    backend.delete_collection(coll_name)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(run_eval())
