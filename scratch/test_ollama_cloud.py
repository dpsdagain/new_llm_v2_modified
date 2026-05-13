import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag_chain import get_llm
from config import OLLAMA_CLOUD_PREFIX, OLLAMA_CLOUD_BASE_URL

def test_ollama_cloud_routing():
    print(f"Testing Ollama Cloud Routing with prefix: {OLLAMA_CLOUD_PREFIX}")
    
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    print(f"Model ID: {model_id}")
    
    try:
        llm = get_llm(model=model_id, streaming=False)
        print(f"LLM Type: {type(llm)}")
        
        # Verify it's ChatOpenAI (as we use it for cloud endpoints)
        from langchain_openai import ChatOpenAI
        assert isinstance(llm, ChatOpenAI)
        
        # Verify base_url
        assert llm.openai_api_base == OLLAMA_CLOUD_BASE_URL
        print(f"SUCCESS: Base URL correctly set to: {llm.openai_api_base}")
        
        # Verify model name (prefix stripped)
        assert llm.model_name == "gemma4:31b-cloud"
        print(f"SUCCESS: Model name correctly set to: {llm.model_name}")
        
        print("DONE: OLLAMA CLOUD ROUTING VERIFIED")
        
    except Exception as e:
        print(f"FAILED Verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_cloud_routing()
