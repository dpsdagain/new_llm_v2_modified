import os
import sys
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import get_llm
from config import OLLAMA_CLOUD_PREFIX

def check_instance_leaking():
    print("--- LLM Instance Leak Diagnostic ---")
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    
    instances = []
    print("\nCreating 20 LLM instances...")
    for i in range(20):
        llm = get_llm(model=model_id, streaming=False)
        instances.append(id(llm))
    
    for i in range(min(5, len(instances))):
        print(f"Instance {i}: {instances[i]}")
        
    unique_instances = len(set(instances))
    print(f"Total instances created: 20")
    print(f"Unique instance IDs: {unique_instances}")
    
    if unique_instances > 1:
        print("\n[CONFIRMED] System creates distinct LLM instances for every call.")
        print("This is the likely cause of socket accumulation if many components call get_llm().")
    else:
        print("\n[REJECTED] get_llm() is already returning cached instances.")

    # Check for TIME_WAIT or connection pressure
    print("\nSystem-wide TIME_WAIT scan (port 443):")
    import subprocess
    try:
        res = subprocess.check_output("netstat -ano | findstr :443 | findstr TIME_WAIT", shell=True)
        lines = res.decode().strip().split("\n")
        print(f"Active TIME_WAIT connections to 443: {len(lines)}")
    except Exception:
        print("No TIME_WAIT connections found for 443.")

if __name__ == "__main__":
    check_instance_leaking()
