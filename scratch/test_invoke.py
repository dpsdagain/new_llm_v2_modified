import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import get_llm
from config import OLLAMA_CLOUD_PREFIX

def simulate_chat():
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    llm = get_llm(model=model_id, streaming=False)
    
    try:
        response = llm.invoke("Say the word 'Hello' and nothing else.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error invoking model: {e}")

if __name__ == "__main__":
    simulate_chat()
