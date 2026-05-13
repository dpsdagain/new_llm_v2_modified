import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import backend
import rag_chain

def run_scenario_test():
    print("--- 🧪 RUNNING SCENARIO TEST: Cross-File Bug Discovery ---")
    
    # 1. SETUP: Ingest the specific scenario directory
    scenario_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_file_bug")
    coll_name = "scenario_test_coll"
    
    print(f"Ingesting from: {scenario_dir}")
    backend.delete_collection(coll_name)
    docs = backend.load_and_chunk_codebase(scenario_dir)
    db, added = backend.ingest_into_chroma(docs, coll_name)
    print(f"Ingested {added} documents.")

    # 2. DEFINE THE QUERY
    query = "I'm running `run_task` in `file_a.py` but it's returning `None`. Why is this happening and where is the root cause? Trace the logic across all relevant files."
    
    # 3. BUILD THE CHAIN
    # Use Ollama Cloud GPT-OSS (120B) Free Tier
    model_id = "ollama-cloud:gpt-oss:120b-cloud"
    print(f"Using model: {model_id}")
    
    chain = rag_chain.build_rag_chain(db, model=model_id)
    
    inputs = {
        "input": query,
        "chat_history": [],
        "collection_name": coll_name,
        "auto_specialist": False # Disable auto-specialist to force gpt-oss-120b
    }

    print("\n--- 🔍 RETRIEVAL RESULTS ---")
    
    full_answer = ""
    retrieved_docs = []
    
    try:
        # Use .stream() instead of .astream() since the underlying function is a sync generator
        iterator = chain.stream(inputs)
        for chunk in iterator:
            if "context" in chunk:
                retrieved_docs = chunk["context"]
                print(f"Retrieved {len(retrieved_docs)} chunks.")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  [{i+1}] {os.path.basename(source)}")
            
            if "answer" in chunk:
                full_answer += chunk["answer"]
                # Print answer chunks as they come
                print(chunk["answer"], end="", flush=True)
                
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n--- 📊 TEST REPORT ---")
    
    # Reasoning and Analysis
    files_retrieved = {os.path.basename(doc.metadata.get("source", "")) for doc in retrieved_docs}
    expected_files = {"file_a.py", "utils.py", "settings.py"}
    missing_files = expected_files - files_retrieved
    
    report = {
        "scenario": "Cross-File Bug Discovery",
        "query": query,
        "model": model_id,
        "files_retrieved": list(files_retrieved),
        "expected_files": list(expected_files),
        "missing_files": list(missing_files),
        "success": len(missing_files) == 0 and ("GLOBAL_LIMIT" in full_answer and ("0" in full_answer or "zero" in full_answer.lower())),
        "reasoning": ""
    }
    
    if report["success"]:
        report["reasoning"] = "The system successfully traced the dependency from file_a.py -> utils.py -> settings.py and identified GLOBAL_LIMIT = 0 as the root cause."
    elif len(missing_files) > 0:
        report["reasoning"] = f"The system failed to retrieve all relevant files. Missing: {missing_files}. This confirms the risk of incomplete context in RAG systems for cross-file bugs."
    else:
        report["reasoning"] = "All files were retrieved, but the LLM failed to correctly identify the root cause."

    print(json.dumps(report, indent=2))
    
    # Save report to file
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "scenario_test_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Scenario Test Report: Cross-File Bug Discovery\n\n")
        f.write(f"## Objective\n")
        f.write(f"Test the system's ability to retrieve and reason across multiple files where a bug in one file is caused by a configuration in a seemingly 'irrelevant' or distant file.\n\n")
        f.write(f"## Setup\n")
        f.write(f"- **File A (`file_a.py`)**: Contains `run_task()` which calls `get_limit()` and fails if result < 1.\n")
        f.write(f"- **Utils (`utils.py`)**: Contains `get_limit()` which returns `settings.GLOBAL_LIMIT`.\n")
        f.write(f"- **Settings (`settings.py`)**: Contains `GLOBAL_LIMIT = 0`.\n\n")
        f.write(f"## Execution\n")
        f.write(f"- **Model**: `{model_id}`\n")
        f.write(f"- **Query**: \"{query}\"\n\n")
        f.write(f"## Results\n")
        f.write(f"- **Files Retrieved**: {', '.join(report['files_retrieved'])}\n")
        f.write(f"- **Success**: {'✅ Yes' if report['success'] else '❌ No'}\n\n")
        f.write(f"### Reasoning\n{report['reasoning']}\n\n")
        f.write(f"### Full LLM Answer\n\n{full_answer}\n")

    print(f"\nReport saved to: {report_path}")

    # Cleanup
    backend.delete_collection(coll_name)

if __name__ == "__main__":
    run_scenario_test()
