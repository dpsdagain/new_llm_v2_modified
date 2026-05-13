import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import re
from rag_chain import _normalize_query

class TestQueryNormalization(unittest.TestCase):
    """
    T-RC-2: Test query normalization consistency.
    Validated against test.md requirements.
    """

    def test_normalize_query(self):
        test_cases = [
            ("What is Python?", "what is python"),
            ("  Hello, World!  ", "hello world"),
            ("test", "test"),
            ("C++ vs Rust", "c vs rust")
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                actual = _normalize_query(input_text)
                # Note: re.sub(r'[^\w\s]', '', "  Hello, World!  ") becomes "  Hello World  "
                # strip() makes it "hello world". 
                # However, if there are multiple spaces inside, they might stay. 
                # Let's verify if the actual implementation handles internal double-spaces.
                # Current implementation: re.sub(r'[^\w\s]', '', query).lower().strip()
                # For "  Hello, World!  ", the comma is replaced by nothing, which might leave 
                # two spaces: "  Hello  World  " -> "hello  world".
                # Let's see if the test plan expected "hello world" (single space).
                self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
