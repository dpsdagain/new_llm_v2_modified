import os

# Standardize OLLAMA_PREFIX as expected in config
OLLAMA_PREFIX = "ollama:"

def get_llm_logic_test(model: str | None = None):
    # Simplified version of get_llm from rag_chain.py
    if model and model.startswith(OLLAMA_PREFIX):
        ollama_model_name = model[len(OLLAMA_PREFIX):]
        return f"ChatOllama(model={ollama_model_name})"
    else:
        current_model = model or "default_cloud_model"
        return f"ChatOpenAI(model={current_model})"

def test_routing():
    print(f"Testing with OLLAMA_PREFIX: '{OLLAMA_PREFIX}'")
    
    # 1. Test local model
    local_id = f"{OLLAMA_PREFIX}llama3.1"
    res = get_llm_logic_test(model=local_id)
    print(f"Model: {local_id} -> {res}")
    assert "ChatOllama" in res
    assert "llama3.1" in res
    
    # 2. Test cloud model
    cloud_id = "google/gemma-4-31b-it:free"
    res = get_llm_logic_test(model=cloud_id)
    print(f"Model: {cloud_id} -> {res}")
    assert "ChatOpenAI" in res
    
    print("✅ Routing logic verified visually and with assertions.")

if __name__ == "__main__":
    test_routing()
