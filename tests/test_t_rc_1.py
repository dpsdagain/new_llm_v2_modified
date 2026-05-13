import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from rag_chain import is_cache_capable, get_cache_profile

class TestRagChainCacheLogic(unittest.TestCase):
    """
    T-RC-1: Test cache capability detection and profile routing.
    Validated against test.md requirements.
    """

    def test_cache_logic(self):
        test_cases = [
            {
                "model": "anthropic/claude-3.5-sonnet",
                "exp_capable": True,
                "exp_profile": (4, 1024)
            },
            {
                "model": "google/gemini-2.0-flash",
                "exp_capable": True,
                "exp_profile": (8, 1028)
            },
            {
                "model": "google/gemma-4-26b",
                "exp_capable": True,
                "exp_profile": (8, 1028)
            },
            {
                "model": "deepseek/deepseek-chat",
                "exp_capable": False,
                "exp_profile": (4, 1024)
            },
            {
                "model": "qwen/qwen-turbo",
                "exp_capable": False,
                "exp_profile": (4, 1024)
            },
            {
                "model": "ollama/llama3.1",
                "exp_capable": False,
                "exp_profile": (4, 1024) # Fallback to default
            }
        ]

        for case in test_cases:
            model = case["model"]
            capable = is_cache_capable(model)
            profile = get_cache_profile(model)
            
            with self.subTest(model=model):
                self.assertEqual(capable, case["exp_capable"], f"Capability mismatch for {model}")
                self.assertEqual(profile, case["exp_profile"], f"Profile mismatch for {model}")

if __name__ == "__main__":
    unittest.main()
