from rag_chain import VectorRouter
from config import SPECIALIST_MAPPING

def test_specialist_routing():
    router = VectorRouter()
    
    test_cases = [
        ("Write a Python function to sort a list of dictionaries.", "CODE"),
        ("Refactor this C++ code for better performance.", "CODE"),
        ("Analyze the logic behind this mathematical proof.", "REASONING"),
        ("Step by step, explain how a transformer model works.", "REASONING"),
        ("What is the capital of France?", "GENERAL"),
        ("Compare these two architectural designs.", "REASONING"),
    ]
    
    print("--- 🧪 Phase 4 Specialist Routing Verification ---")
    
    all_passed = True
    for query, expected_label in test_cases:
        actual_label = router.detect_specialty(query)
        model = SPECIALIST_MAPPING.get(actual_label)
        status = "✅ PASS" if actual_label == expected_label else "❌ FAIL"
        print(f"Query: '{query[:40]}...'")
        print(f"  -> Detected: {actual_label} | Model: {model} | {status}")
        if actual_label != expected_label:
            all_passed = False
            
    if all_passed:
        print("\n🎉 SUCCESS: All intent categories correctly identified and routed.")
    else:
        print("\n⚠️ WARNING: Some intents were misclassified. Consider tuning triggers in rag_chain.py.")

if __name__ == "__main__":
    test_specialist_routing()
