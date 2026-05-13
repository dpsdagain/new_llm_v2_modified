import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import asyncio
import time
import config
import backend
import rag_chain
import shutil
import hashlib
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import TextLoader

# --- CONFIGURATION ---
TARGET_FILES = ["app.py", "rag_chain.py", "backend.py", "config.py", "requirements.txt"]
MODEL_ID = "ollama-cloud:gpt-oss:120b-cloud"
COLLECTION_NAME = "multi_hop_stress_test"

async def run_gpt_120b_test():
    print(f"--- 🚀 STARTING GPT-120B MULTI-HOP STRESS TEST ---")
    print(f"Model: {MODEL_ID}")
    
    # 1. HARD CLEANUP
    print(f"Deleting existing collection '{COLLECTION_NAME}'...")
    try:
        backend.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Backend delete failed: {e}")
        
    # Attempt to clear FTS5 files
    for p in Path('.').rglob("*_fts5.db"):
        try:
            p.unlink()
        except:
            pass

    # 2. SELECTIVE INGESTION
    print(f"Ingesting specific files for context: {TARGET_FILES}")
    all_chunks = []
    
    for fname in TARGET_FILES:
        fpath = os.path.abspath(fname)
        if not os.path.exists(fpath):
            continue
            
        ext = Path(fpath).suffix.lower()
        try:
            loader = TextLoader(fpath, autodetect_encoding=True)
            raw_docs = loader.load()
            content = raw_docs[0].page_content
        except:
            continue

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Ingest Logic (using the most robust path identified in previous turns)
        if len(content) < config.ZERO_CHUNK_THRESHOLD:
            all_chunks.append(Document(
                page_content=content,
                metadata={"source": fpath, "zero_chunk": True, "content_hash": h, "file_extension": ext}
            ))
        else:
            from backend import CodeASTChunker, _content_hash
            ast_chunker = CodeASTChunker(chunk_size=config.CODE_CHUNK_SIZE)
            ast_chunks = ast_chunker.chunk_file(content, fpath, ext)
            if ast_chunks:
                for i, c in enumerate(ast_chunks):
                    c.metadata.update({"source_type": "code", "file_extension": ext, "chunk_index": i, "content_hash": _content_hash(c)})
                all_chunks.extend(ast_chunks)

    db, added = backend.ingest_into_chroma(all_chunks, COLLECTION_NAME)
    print(f"Successfully ingested {added} chunks into '{COLLECTION_NAME}'.")

    # 3. RUN TEST PROMPTS
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
        },
        {
            "level": "Level 8: The Security Injection Test (Vulnerability Synthesis)",
            "query": "Identify a potential 'Prompt Injection' vulnerability in the _background_summarize function. Explain how a user could craft a message that forces the Sentinel to include the OPENROUTER_API_KEY in its summary. Trace the path from the user's input in app.py to the background LLM call in rag_chain.py and identify the missing validation check."
        }
    ]

    chain = rag_chain.build_rag_chain(db, model=MODEL_ID)
    
    report_content = f"# GPT-120B Multi-Hop Stress Test Report\n\n"
    report_content += f"**Model:** {MODEL_ID}\n"
    report_content += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += f"**Files Ingested:** {', '.join(TARGET_FILES)}\n\n"

    for q in queries:
        print(f"\n\n{'='*80}")
        print(f"📌 {q['level']}")
        print(f"Query: {q['query']}")
        print(f"{'-'*80}")
        
        report_content += f"## {q['level']}\n\n"
        report_content += f"**Query:** {q['query']}\n\n"
        report_content += f"**Answer:**\n\n"
        
        full_answer = ""
        inputs = {
            "input": q['query'],
            "chat_history": [],
            "collection_name": COLLECTION_NAME,
            "auto_specialist": False 
        }
        
        try:
            for chunk in chain.stream(inputs):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
                    full_answer += chunk["answer"]
            
            report_content += full_answer + "\n\n"
            report_content += "---\n\n"
            
        except Exception as e:
            print(f"\nERROR: {e}")
            report_content += f"**ERROR:** {e}\n\n"

    # 4. SAVE REPORT
    with open("multi_hop_stress_test_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n\n--- ✅ TEST COMPLETE. Report saved to 'multi_hop_stress_test_report.md' ---")

if __name__ == "__main__":
    asyncio.run(run_gpt_120b_test())
