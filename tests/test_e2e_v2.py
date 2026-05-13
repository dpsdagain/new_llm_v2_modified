import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
import backend
import rag_chain

async def test_e2e_suite_logic_only():
    print("--- 🏁 STARTING SECTION 9: END-TO-END (T-E2E) LOGIC VERIFICATION ---")
    
    # SETUP
    coll_name = "e2e_test_col_logic"
    backend.delete_collection(coll_name)
    docs = [
        backend.Document(page_content="The project is named 'Private AI'.", metadata={"source": "README.md", "content_hash": "h1"}),
    ]
    db, _ = backend.ingest_into_chroma(docs, coll_name)
    
    # We will manually trigger the _full_context_cache_chain logic 
    # to verify metadata transitions (Intents, Caches, Sentinels)
    chain_lambda = rag_chain.build_rag_chain(db)
    
    # --- T-E2E-1: Single-Turn ---
    print("\n[T-E2E-1] Testing Single-Turn Metadata...")
    inputs = {"input": "What is the name?", "chat_history": [], "collection_name": coll_name}
    
    gen = chain_lambda.stream(inputs)
    metadata = next(gen)
    print(f"  Result: Intent={metadata.get('intent')}, Docs={len(metadata.get('context', []))}")
    if metadata.get('intent') == "NEW" and len(metadata.get('context', [])) > 0:
        print("  ✅ T-E2E-1 Logic Passed.")
    else:
        print("  ❌ T-E2E-1 Logic Failed.")

    # --- T-E2E-4: Semantic Cache Hit ---
    print("\n[T-E2E-4] Testing Semantic Cache Metadata...")
    sem_cache = rag_chain.get_semantic_cache()
    sem_cache.upsert("What is the name?", "The project is named Private AI.")
    
    gen = chain_lambda.stream(inputs)
    metadata = next(gen)
    print(f"  Result: Intent={metadata.get('intent')}")
    if metadata.get('intent') == "CACHE_HIT":
        print("  ✅ T-E2E-4 Logic Passed.")
    else:
        print("  ❌ T-E2E-4 Logic Failed.")

    # --- T-E2E-2: Follow-Up Intent ---
    print("\n[T-E2E-2] Testing Follow-Up Intent Detection...")
    history = [HumanMessage(content="What is the name?"), AIMessage(content="Private AI.")]
    inputs_followup = {
        "input": "Where is it stored?", 
        "chat_history": history, 
        "collection_name": coll_name,
        "last_query": "What is the name?",
        "force_retrieval": True 
    }
    gen = chain_lambda.stream(inputs_followup)
    metadata = next(gen)
    print(f"  Result: Intent={metadata.get('intent')}")
    if metadata.get('intent') == "FOLLOW-UP":
        print("  ✅ T-E2E-2 Logic Passed.")
    else:
        print(f"  ❌ T-E2E-2 Logic Failed (Intent: {metadata.get('intent')}).")

    # --- T-E2E-5: Sentinel Trigger ---
    print("\n[T-E2E-5] Testing Sentinel Trigger Logic...")
    heavy_content = "word " * 2300 
    history_long = [
        HumanMessage(content=heavy_content), AIMessage(content="OK"),
        HumanMessage(content=heavy_content), AIMessage(content="OK")
    ]
    inputs_sentinel = {
        "input": "Next question", 
        "chat_history": history_long, 
        "collection_name": coll_name,
        "sentinel_future_active": False,
        "force_retrieval": True
    }
    gen = chain_lambda.stream(inputs_sentinel)
    metadata = next(gen)
    has_future = metadata.get("sentinel_future") is not None
    print(f"  Result: Sentinel Future Triggered={has_future}")
    if has_future:
        print("  ✅ T-E2E-5 Logic Passed.")
    else:
        print("  ❌ T-E2E-5 Logic Failed.")

    # CLEANUP
    backend.delete_collection(coll_name)
    print("\n--- E2E LOGIC SUITE COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(test_e2e_suite_logic_only())
