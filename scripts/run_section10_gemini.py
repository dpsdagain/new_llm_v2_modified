import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import asyncio
import time
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
import backend
import rag_chain

# Override config for testing
config.GEMINI_MODEL = "google/gemini-2.0-flash-001"
config.DEFAULT_MODEL = "google/gemini-2.0-flash-001"
# Disable auto-specialist for this test to focus on the requested Gemini model
config.ENABLE_AUTO_SPECIALIST = False

async def run_section10_tests():
    print(f"--- 🚀 RUNNING SECTION 10 TESTS: {config.GEMINI_MODEL} ---")
    
    # 1. SETUP: Ingest codebase into "self_test"
    coll_name = "self_test"
    print(f"Cleaning and ingesting into '{coll_name}'...")
    backend.delete_collection(coll_name)
    docs = backend.load_and_chunk_codebase(os.path.dirname(__file__))
    db, added = backend.ingest_into_chroma(docs, coll_name)
    print(f"Ingested {added} documents.")

    # 2. QA LIST (Section 10)
    # We'll handle QA-05 (multi-turn) and QA-12 (pinned) specially.
    
    qa_list = [
        {"id": "QA-01", "query": "Explain the end-to-end flow when a user submits a query. What are the main stages?", 
         "expected": ["history", "cache", "intent", "hybrid", "rerank", "prompt", "llm"]},
        {"id": "QA-02", "query": "What does the hybrid_search function do and what algorithm does it use to merge results?", 
         "expected": ["RRF", "Reciprocal Rank Fusion", "vector", "BM25"]},
        {"id": "QA-03", "query": "What is the current value of RERANK_CANDIDATES and why was it recently changed?", 
         "expected": ["15", "token efficiency"]},
        {"id": "QA-04", "query": "Write a Python function that estimates the token count of a list of LangChain messages using the _est_tokens logic from this codebase.", 
         "expected": ["def", "len", "// 3"]},
        # QA-05 is handled below
        {"id": "QA-06", "query": "What happens to chat history when it grows large? Explain the Ghost History mechanism.", 
         "expected": ["GHOST_HISTORY_MAX", "200", "800", "sentinel"]},
        {"id": "QA-07", "query": "What is a zero-chunk and when is it created during ingestion?", 
         "expected": ["ZERO_CHUNK_THRESHOLD", "100,000", "single document"]},
        {"id": "QA-08", "query": "Why are large zero-chunk documents now filtered out from retrieval results?", 
         "expected": ["context explosions", "33k tokens", "signal-to-noise"]},
        {"id": "QA-09", "query": "How does provider-side prefix caching work in this system? Which models support it?", 
         "expected": ["Claude", "Gemini", "deterministic", "5 blocks"]},
        {"id": "QA-10", "query": "Analyze the architectural trade-offs between using a Sentinel summary vs. keeping full chat history for long conversations.", 
         "expected": ["sentinel", "history", "compression", "tokens"]},
        {"id": "QA-11", "query": "What is the GDP of France in 2024?", 
         "expected": ["not in context", "cannot find", "no information"]},
        # QA-12 is handled below
        {"id": "QA-13", "query": "What are the current values for SENTINEL_INTERVAL and SENTINEL_TOKEN_THRESHOLD?", 
         "expected": ["3", "1500"]},
        {"id": "QA-14", "query": "What model is used for reranking retrieved documents and what does it optimize for?", 
         "expected": ["MiniLM", "cross-encoder", "relevance"]},
        {"id": "QA-15", "query": "If I ask the same question twice, what happens on the second ask?", 
         "expected": ["semantic cache", "0.85", "skip retrieval"]}
    ]

    report = []
    report.append(f"# Section 10 QA Test Report\n")
    report.append(f"**Model:** {config.GEMINI_MODEL}\n")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("| ID | Question | Accuracy | Result |\n")
    report.append("|---|---|---|---|\n")

    chain = rag_chain.build_rag_chain(db, model=config.GEMINI_MODEL)

    async def run_query(query, chat_history=[], full_source_context="None pinned.", collection_name=coll_name):
        full_answer = ""
        inputs = {
            "input": query,
            "chat_history": chat_history,
            "full_source_context": full_source_context,
            "collection_name": collection_name,
            "auto_specialist": False
        }
        await asyncio.sleep(2) # Prevent rate limiting
        for chunk in chain.stream(inputs):
            if "answer" in chunk:
                full_answer += chunk["answer"]
        return full_answer

    # Run standard QA
    for qa in qa_list:
        print(f"Running {qa['id']}...", flush=True)
        answer = await run_query(qa['query'])
        found = [term for term in qa['expected'] if term.lower() in answer.lower()]
        score = len(found) / len(qa['expected'])
        status = "✅ PASS" if score >= 0.5 else "❌ FAIL"
        report.append(f"| {qa['id']} | {qa['query'][:50]}... | {score*100:.0f}% | {status} |\n")
        print(f"  Result: {status} ({score*100:.0f}%)")

    # QA-05: Follow-up chain (Multi-turn)
    print("Running QA-05 (Multi-turn)...", flush=True)
    history = []
    q5_1 = "What is the SemanticCache class?"
    a5_1 = await run_query(q5_1, history)
    history.append(HumanMessage(content=q5_1))
    history.append(AIMessage(content=a5_1))
    
    q5_2 = "How does its lookup method decide whether to return a cached answer?"
    a5_2 = await run_query(q5_2, history)
    history.append(HumanMessage(content=q5_2))
    history.append(AIMessage(content=a5_2))
    
    q5_3 = "What threshold is used and where is it configured?"
    a5_3 = await run_query(q5_3, history)
    
    q5_expected = ["0.85", "config.py", "SEMANTIC_CACHE_THRESHOLD"]
    found_q5 = [term for term in q5_expected if term.lower() in a5_3.lower()]
    score_q5 = len(found_q5) / len(q5_expected)
    status_q5 = "✅ PASS" if score_q5 >= 0.5 else "❌ FAIL"
    report.append(f"| QA-05 | Follow-up Chain (Multi-turn) | {score_q5*100:.0f}% | {status_q5} |\n")
    print(f"  Result: {status_q5} ({score_q5*100:.0f}%)")

    # QA-12: Pinned file test
    print("Running QA-12 (Pinned file)...", flush=True)
    with open("rag_chain.py", "r", encoding="utf-8") as f:
        rag_chain_content = f.read()
    
    q12 = "In the pinned file, what line does the _get_max_tokens function start on and what does it return for a GENERAL query shorter than 200 characters?"
    a12 = await run_query(q12, full_source_context=rag_chain_content)
    
    # We know it's line 114 (or near it) and returns 1024
    q12_expected = ["114", "1024"]
    found_q12 = [term for term in q12_expected if term.lower() in a12.lower()]
    score_q12 = len(found_q12) / len(q12_expected)
    status_q12 = "✅ PASS" if score_q12 >= 0.5 else "❌ FAIL"
    report.append(f"| QA-12 | Pinned File Test (rag_chain.py) | {score_q12*100:.0f}% | {status_q12} |\n")
    print(f"  Result: {status_q12} ({score_q12*100:.0f}%)")

    # Write report to file
    with open("section10_test_report.md", "w", encoding="utf-8") as f:
        f.writelines(report)
    
    print(f"\n✅ Tests complete. Report written to section10_test_report.md")

    # Cleanup
    backend.delete_collection(coll_name)

if __name__ == "__main__":
    asyncio.run(run_section10_tests())
