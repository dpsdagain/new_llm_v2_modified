import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import asyncio
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
import backend
import rag_chain

async def check_gui_failure_reason():
    print("--- 🔍 DIAGNOSING GUI VS SCRIPT DISCREPANCY ---")
    
    # 1. Check if Ollama is responsive (used by Auto-Specialist)
    from langchain_community.chat_models import ChatOllama
    try:
        ollama = ChatOllama(base_url=config.OLLAMA_BASE_URL, model="llama3.2:1b", timeout=2)
        # Quick check
        print("Checking Ollama connectivity...")
        resp = ollama.invoke("hi")
        print("✅ Ollama is ONLINE.")
    except Exception as e:
        print(f"❌ Ollama is OFFLINE or model not found: {e}")
        print("   (This explains why GUI fails if 'Auto-Specialist' is ON)")

    # 2. Check collections
    cols = backend.list_collections()
    print(f"Available collections: {cols}")
    if "self_test" not in cols:
        print("❌ 'self_test' collection missing. Did you ingest in the GUI?")
    else:
        info = backend.get_collection_info("self_test")
        print(f"✅ 'self_test' exists with {info['count']} chunks.")

    # 3. Test a Specialist Routing Query (QA-04 logic)
    print("\nTesting QA-04 with auto_specialist=True...")
    db = backend.load_existing_chroma("self_test")
    if db:
        chain = rag_chain.build_rag_chain(db, model="google/gemini-2.0-flash-001")
        inputs = {
            "input": "Write a Python function that estimates tokens using _est_tokens.",
            "chat_history": [],
            "collection_name": "self_test",
            "auto_specialist": True
        }
        try:
            full_answer = ""
            # This will attempt to route to Ollama for CODE specialty
            for chunk in chain.stream(inputs):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                if "specialist_active" in chunk:
                    print(f"   Routed to specialist: {chunk['specialist_active']}")
            
            if full_answer:
                print("✅ Query succeeded with auto_specialist.")
            else:
                print("❌ Query returned empty answer with auto_specialist.")
        except Exception as e:
            print(f"❌ Query FAILED with auto_specialist: {e}")

if __name__ == "__main__":
    asyncio.run(check_gui_failure_reason())
