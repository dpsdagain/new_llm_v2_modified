import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import unittest

class TestProviderCacheProfiles(unittest.TestCase):
    """
    T-CFG-2: Unit test for PROVIDER_CACHE_PROFILES structure.
    Validated against test.md requirements.
    """

    def test_keys_lowercase_and_completeness(self):
        expected_keys = {
            "claude", "gemini", "gemma", "deepseek", "qwen", 
            "nemotron", "glm", "gpt-5", "reka", "mistral"
        }
        actual_keys = set(config.PROVIDER_CACHE_PROFILES.keys())
        
        # Check all expected keys are present
        for key in expected_keys:
            self.assertIn(key, actual_keys, f"Missing expected provider key: {key}")
            
        # Check all keys are lowercase
        for key in actual_keys:
            self.assertEqual(key, key.lower(), f"Provider key should be lowercase: {key}")

    def test_value_types(self):
        for key, value in config.PROVIDER_CACHE_PROFILES.items():
            self.assertIsInstance(value, tuple, f"Value for {key} should be a tuple")
            self.assertEqual(len(value), 2, f"Value for {key} should be a tuple of length 2")
            self.assertIsInstance(value[0], int, f"First element of tuple for {key} should be int")
            self.assertIsInstance(value[1], int, f"Second element of tuple for {key} should be int")

    def test_specific_values(self):
        # Claude: (4, 1024)
        self.assertEqual(config.PROVIDER_CACHE_PROFILES["claude"], (4, 1024))
        # Gemini: (8, 1028)
        self.assertEqual(config.PROVIDER_CACHE_PROFILES["gemini"], (8, 1028))
        # Gemma: (8, 1028)
        self.assertEqual(config.PROVIDER_CACHE_PROFILES["gemma"], (8, 1028))

if __name__ == "__main__":
    unittest.main()
