import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import asyncio
import time
import hashlib
import config
import backend
import rag_chain
from langchain_core.documents import Document

async def run_eval_restricted():
    coll_name = f"eval_restricted_{int(time.time())}"
    model_id = "google/gemini-2.0-flash-001"
    
    # Files requested by user
    target_files = ["app.py", "rag_chain.py", "backend.py", "config.py", "requirements.txt"]
    
    print(f"--- 🚀 RUNNING EVALUATION (RESTRICTED INGESTION) WITH {model_id} ---")
    print(f"Target Files: {target_files}")
    
    # 1. SETUP: Delete and Re-ingest specific files
    backend.delete_collection(coll_name)
    print("Ingesting restricted file set...")
    
    all_chunks = []
    chunker = backend.CodeASTChunker()
    
    for filename in target_files:
        if not os.path.exists(filename):
            print(f"⚠️ Warning: {filename} not found.")
            continue
            
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        ext = os.path.splitext(filename)[1]
        # Use the AST chunker logic from backend
        chunks = chunker.chunk_file(content, filename, ext)
        
        # Add metadata consistent with backend.py
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": os.path.abspath(filename),
                "source_type": "code",
                "file_extension": ext,
                "chunk_index": i
            })
            chunk.metadata["content_hash"] = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            all_chunks.append(chunk)

    db, added = backend.ingest_into_chroma(all_chunks, coll_name)
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
        
        inputs = {
            "input": q['query'],
            "chat_history": [],
            "collection_name": coll_name,
            "auto_specialist": False,
            "force_retrieval": True
        }
        
        try:
            for chunk in chain.stream(inputs):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
        except Exception as e:
            print(f"\nERROR: {e}")

    # Cleanup
    print("\n\nCleaning up...")
    backend.delete_collection(coll_name)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(run_eval_restricted())
