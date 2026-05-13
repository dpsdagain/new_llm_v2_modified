from rag_chain import get_llm
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from config import OLLAMA_PREFIX

def test_routing():
    print(f"Testing with OLLAMA_PREFIX: {OLLAMA_PREFIX}")
    
    # Test local model
    local_model_id = f"{OLLAMA_PREFIX}llama3.1"
    print(f"Testing model ID: {local_model_id}")
    llm_local = get_llm(model=local_model_id, streaming=False)
    print(f"Type of LLM for local: {type(llm_local)}")
    assert isinstance(llm_local, ChatOllama), f"Expected ChatOllama, got {type(llm_local)}"
    
    # Test cloud model
    cloud_model_id = "google/gemma-4-31b-it:free"
    print(f"Testing model ID: {cloud_model_id}")
    llm_cloud = get_llm(model=cloud_model_id, streaming=False)
    print(f"Type of LLM for cloud: {type(llm_cloud)}")
    assert isinstance(llm_cloud, ChatOpenAI), f"Expected ChatOpenAI, got {type(llm_cloud)}"
    
    print("✅ Routing verification passed!")

if __name__ == "__main__":
    try:
        test_routing()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
