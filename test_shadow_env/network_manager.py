# network_manager.py - Handles networking
from settings import RETRY_STRATEGY_ID, MAX_RETRIES

def perform_api_call(endpoint, payload):
    print(f"Calling {endpoint} using {RETRY_STRATEGY_ID} strategy.")
    
    for i in range(MAX_RETRIES + 1):
        try:
            return _mock_call(endpoint, payload)
        except Exception as e:
            if RETRY_STRATEGY_ID == "FAIL_FAST":
                print("FAIL_FAST strategy: no retries allowed.")
                raise RuntimeError("Immediate failure per strategy.") from e
            print(f"Attempt {i+1} failed, retrying...")
    return None

def _mock_call(e, p):
    raise ConnectionError("Timeout")
