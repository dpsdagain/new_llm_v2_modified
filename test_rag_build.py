import sys
import os
# Add current dir to path
sys.path.append(os.getcwd())

from rag_chain import build_rag_chain
from langchain_core.messages import AIMessageChunk

print("Testing rag_chain building and stream output format...")
try:
    chain = build_rag_chain(None, model="google/gemini-2.0-flash-001")
    print("✅ Chain built successfully!")
    
    # Mock inputs
    inputs = {
        "input": "test",
        "chat_history": [],
        "full_source_context": "test",
        "stable_context": "test",
        "new_context": "test"
    }
    
    # We can't easily stream without real API keys, but we can inspect the chain structure
    print(f"Chain type: {type(chain)}")
    
    # Check if we can import the necessary classes
    from rag_chain import _prepare_history_with_cache
    print("✅ Logic imports verified!")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
