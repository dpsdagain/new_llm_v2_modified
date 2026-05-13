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

async def run_exhaustive_qa_ranking():
    # Extract ALL free models from config
    models_to_test = set()
    for category in config.CLOUDROUTER_MODELS.values():
        for model_id in category.values():
            if ":free" in model_id:
                models_to_test.add(model_id)
    
    models_to_test = list(models_to_test)
    print(f"--- 🚀 EXHAUSTIVE RAG RANKING: {len(models_to_test)} FREE MODELS ---")
    
    # 1. SETUP: Ingest codebase
    coll_name = "ranking_test_col"
    backend.delete_collection(coll_name)
    docs = backend.load_and_chunk_codebase(os.path.dirname(__file__))
    db, added = backend.ingest_into_chroma(docs, coll_name)
    print(f"Ingested {added} documents.")

    # 2. FULL QA LIST (Section 10)
    qa_list = [
        {"id": "QA-01", "query": "Explain the end-to-end flow when a user submits a query. What are the main stages?", "expected": ["history", "semantic cache", "intent", "hybrid", "rerank", "llm"]},
        {"id": "QA-02", "query": "What does the hybrid_search function do and what algorithm does it use to merge results?", "expected": ["RRF", "Reciprocal Rank Fusion", "vector", "BM25"]},
        {"id": "QA-03", "query": "What is the current value of RERANK_CANDIDATES and why was it recently changed?", "expected": ["15", "token efficiency"]},
        {"id": "QA-04", "query": "Write a Python function that estimates the token count of a list of LangChain messages using the _est_tokens logic from this codebase.", "expected": ["def", "len", "// 3"]},
        {"id": "QA-06", "query": "What happens to chat history when it grows large? Explain the Ghost History mechanism.", "expected": ["GHOST_HISTORY_MAX", "truncate", "200", "800"]},
        {"id": "QA-07", "query": "What is a zero-chunk and when is it created during ingestion?", "expected": ["ZERO_CHUNK_THRESHOLD", "100,000", "single document"]},
        {"id": "QA-08", "query": "Why are large zero-chunk documents now filtered out from retrieval results?", "expected": ["context explosions", "33k tokens", "signal-to-noise"]},
        {"id": "QA-09", "query": "How does provider-side prefix caching work in this system? Which models support it?", "expected": ["Claude", "Gemini", "deterministic", "5 blocks"]},
        {"id": "QA-10", "query": "Analyze the architectural trade-offs between using a Sentinel summary vs. keeping full chat history for long conversations.", "expected": ["sentinel", "history", "compression", "tokens"]},
        {"id": "QA-11", "query": "What is the GDP of France in 2024?", "expected": ["not in context", "cannot find", "no information"]},
        {"id": "QA-13", "query": "What are the current values for SENTINEL_INTERVAL and SENTINEL_TOKEN_THRESHOLD?", "expected": ["3", "1500"]},
        {"id": "QA-14", "query": "What model is used for reranking retrieved documents and what does it optimize for?", "expected": ["MiniLM", "cross-encoder", "relevance"]},
        {"id": "QA-15", "query": "If I ask the same question twice, what happens on the second ask?", "expected": ["semantic cache", "0.85", "skip retrieval"]}
    ]

    final_results = {}

    for model_id in models_to_test:
        print(f"\n\n{'='*80}")
        print(f"🧪 TESTING MODEL: {model_id}")
        print(f"{'='*80}")
        
        chain = rag_chain.build_rag_chain(db, model=model_id)
        model_scores = []

        for qa in qa_list:
            print(f"\n[{qa['id']}] ", end="", flush=True)
            full_answer = ""
            inputs = {
                "input": qa['query'],
                "chat_history": [],
                "collection_name": coll_name,
                "auto_specialist": True # TEST Specialist Routing detection
            }
            
            try:
                # Delay to mitigate 429s across multiple models
                await asyncio.sleep(3) 
                
                for chunk in chain.stream(inputs):
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
            except Exception as e:
                full_answer = f"ERROR: {e}"

            # Scoring
            found = [term for term in qa['expected'] if term.lower() in full_answer.lower()]
            score = len(found) / len(qa['expected'])
            model_scores.append(score)
            print(f"{'✅' if score >= 0.5 else '❌'} {score*100:.0f}%", end=" ", flush=True)

        avg_score = sum(model_scores) / len(model_scores)
        final_results[model_id] = avg_score
        print(f"\nAVG SCORE for {model_id}: {avg_score*100:.1f}%")

    # 4. FINAL RANKING REPORT
    print("\n\n" + "#" * 40)
    print("🏆 FINAL FREE MODEL RANKING (Section 10)")
    print("#" * 40)
    
    ranked = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'RANK':<5} | {'MODEL ID':<45} | {'ACCURACY'}")
    print("-" * 65)
    for i, (model, score) in enumerate(ranked, 1):
        print(f"{i:<5} | {model:<45} | {score*100:>7.1f}%")
    
    # Final Summary for User
    best_model = ranked[0][0]
    print(f"\n🥇 BEST PERFORMING MODEL: {best_model}")
    print("########################################")

    # Cleanup
    backend.delete_collection(coll_name)

if __name__ == "__main__":
    asyncio.run(run_exhaustive_qa_ranking())
