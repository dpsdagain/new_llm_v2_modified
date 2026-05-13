import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import time
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

import backend
import rag_chain
from config import CHROMA_DB_DIR, OLLAMA_CLOUD_PREFIX

def run_tests():
    print("=== Level 3 Stress Test Suite ===")
    
    # 1. Fresh Ingestion
    print("\n--- Phase 1: Fresh Ingestion ---")
    target_files = ["app.py", "backend.py", "config.py", "rag_chain.py", "requirements.txt"]
    abs_targets = [os.path.abspath(f) for f in target_files]
    
    all_chunks = []
    for fpath in abs_targets:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            continue
            
        print(f"Chunking {os.path.basename(fpath)}...")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        ext = Path(fpath).suffix.lower()
        
        # Use AST Chunker for .py files
        if ext == ".py":
            ast_chunker = backend.CodeASTChunker(chunk_size=backend.CODE_CHUNK_SIZE)
            chunks = ast_chunker.chunk_file(content, fpath, ext)
            if chunks:
                for i, chunk in enumerate(chunks):
                    chunk.metadata["source_type"] = "code"
                    chunk.metadata["file_extension"] = ext
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["content_hash"] = backend._content_hash(chunk)
                all_chunks.extend(chunks)
            else:
                # Fallback to legacy splitter
                print(f"AST chunking failed for {os.path.basename(fpath)}, falling back to legacy splitter.")
                from langchain_core.documents import Document
                from backend import _get_splitter
                splitter = _get_splitter(ext, chunk_size_override=backend.CODE_CHUNK_SIZE)
                sub_chunks = splitter.split_text(content)
                for i, text in enumerate(sub_chunks):
                    chunk = Document(
                        page_content=text,
                        metadata={
                            "source": fpath,
                            "source_type": "code",
                            "file_extension": ext,
                            "chunk_index": i,
                            "content_hash": backend.hashlib.sha256(text.encode("utf-8")).hexdigest()
                        }
                    )
                    all_chunks.append(chunk)
        else:
            # Fallback for requirements.txt
            from langchain_core.documents import Document
            chunk = Document(
                page_content=content,
                metadata={
                    "source": fpath,
                    "source_type": "code",
                    "file_extension": ext,
                    "zero_chunk": True,
                    "chunk_index": 0,
                    "content_hash": backend.hashlib.sha256(content.encode("utf-8")).hexdigest()
                }
            )
            all_chunks.append(chunk)

    print(f"Total chunks created: {len(all_chunks)}")
    collection_name = "level3_stress_test"
    db, added = backend.ingest_into_chroma(all_chunks, collection_name)
    print(f"Successfully ingested {added} chunks into '{collection_name}'.")

    # 2. Setup RAG Chain
    print("\n--- Phase 2: Setup RAG Chain ---")
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    print(f"Using model: {model_id}")
    chain = rag_chain.build_rag_chain(db, model=model_id)

    # 3. Test Cases
    test_cases = [
        {
            "id": "3.1",
            "name": "Aggregation Override",
            "prompt": "List every place where SQLiteFTS5BM25 is initialized or queried across all files."
        },
        {
            "id": "3.2",
            "name": "3-Hop Constant Propagation",
            "prompt": "Trace step-by-step how the ENABLE_PROMPT_CACHING flag ultimately affects the system prompt formatting."
        },
        {
            "id": "3.3",
            "name": "AST Context Awareness",
            "prompt": "Look at the function _extract_called_functions. What class does it belong to, and what file is it in?"
        }
    ]

    results = []

    for tc in test_cases:
        print(f"\n--- Running Test {tc['id']}: {tc['name']} ---")
        print(f"Prompt: {tc['prompt']}")
        
        inputs = {
            "input": tc["prompt"],
            "chat_history": [],
            "collection_name": collection_name,
            "auto_specialist": False # Explicitly use the selected model
        }
        
        full_answer = ""
        metadata = {}
        
        try:
            for chunk in chain.stream(inputs):
                if isinstance(chunk, dict):
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
                        print(chunk["answer"], end="", flush=True)
                    else:
                        # Other metadata like context, intent, etc.
                        for k, v in chunk.items():
                            if k != "query_embedding": # Don't print embedding
                                metadata[k] = v
            
            print("\n")
            results.append({
                "test_id": tc["id"],
                "name": tc["name"],
                "prompt": tc["prompt"],
                "answer": full_answer,
                "metadata": {k: str(v) for k, v in metadata.items()}
            })
            
        except Exception as e:
            print(f"\nERROR running test {tc['id']}: {e}")
            results.append({
                "test_id": tc["id"],
                "name": tc["name"],
                "prompt": tc["prompt"],
                "error": str(e)
            })

    # 4. Generate Report
    print("\n--- Phase 3: Generating Report ---")
    with open("level3_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    report = f"# Level 3 Stress Test Report\n\n"
    report += f"**Model:** {model_id}\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Files Ingested:** {', '.join(target_files)}\n\n"
    
    for res in results:
        report += f"## Test {res['test_id']}: {res['name']}\n\n"
        report += f"**Prompt:** {res['prompt']}\n\n"
        if "error" in res:
            report += f"**Status:** FAILED\n\n"
            report += f"**Error:** {res['error']}\n\n"
        else:
            report += f"**Answer:**\n\n{res['answer']}\n\n"
            # Logic for Pass/Fail reasoning (to be filled after analysis)
            report += f"**Analysis:** [To be completed after manual review]\n\n"
        report += "---\n\n"
    
    with open("level3_stress_test_report.md", "w") as f:
        f.write(report)
    
    print("Done. Results saved to level3_test_results.json and level3_stress_test_report.md")

if __name__ == "__main__":
    run_tests()
