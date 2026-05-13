import sys
import os

# Mock the pieces if needed, but let's try importing get_llm first
try:
    print("Importing get_llm...")
    from rag_chain import get_llm
    from config import OLLAMA_PREFIX
    print("Import successful.")

    def test_routing():
        print(f"OLLAMA_PREFIX is: '{OLLAMA_PREFIX}'")
        
        # 1. Test local model string
        local_id = f"{OLLAMA_PREFIX}llama3.1"
        print(f"Testing local ID: {local_id}")
        llm = get_llm(model=local_id, streaming=False)
        print(f"Resulting LLM type: {type(llm)}")
        assert "ChatOllama" in str(type(llm))
        
        # 2. Test cloud model string
        cloud_id = "google/gemma-4-31b-it:free"
        print(f"Testing cloud ID: {cloud_id}")
        llm_cloud = get_llm(model=cloud_id, streaming=False)
        print(f"Resulting LLM type: {type(llm_cloud)}")
        assert "ChatOpenAI" in str(type(llm_cloud))
        
        print("✅ ROUTING LOGIC VERIFIED")

    test_routing()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
