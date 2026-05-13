import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import unittest

class TestConfigConstants(unittest.TestCase):
    """
    T-CFG-1: Unit test for config constants types and ranges.
    Validated against test.md requirements.
    """

    def test_zero_chunk_threshold(self):
        self.assertIsInstance(config.ZERO_CHUNK_THRESHOLD, int)
        self.assertGreater(config.ZERO_CHUNK_THRESHOLD, 0)
        self.assertEqual(config.ZERO_CHUNK_THRESHOLD, 100000)

    def test_retriever_k(self):
        self.assertIsInstance(config.RETRIEVER_K, int)
        self.assertGreater(config.RETRIEVER_K, 0)
        self.assertEqual(config.RETRIEVER_K, 6)

    def test_rerank_top_k(self):
        self.assertIsInstance(config.RERANK_TOP_K, int)
        self.assertGreater(config.RERANK_TOP_K, 0)
        self.assertEqual(config.RERANK_TOP_K, 6)

    def test_rerank_candidates(self):
        self.assertIsInstance(config.RERANK_CANDIDATES, int)
        self.assertGreaterEqual(config.RERANK_CANDIDATES, config.RERANK_TOP_K)
        self.assertEqual(config.RERANK_CANDIDATES, 25)

    def test_semantic_cache_threshold(self):
        self.assertIsInstance(config.SEMANTIC_CACHE_THRESHOLD, float)
        self.assertGreater(config.SEMANTIC_CACHE_THRESHOLD, 0)
        self.assertLessEqual(config.SEMANTIC_CACHE_THRESHOLD, 1.0)
        self.assertEqual(config.SEMANTIC_CACHE_THRESHOLD, 0.85)

    def test_bm25_weight(self):
        self.assertIsInstance(config.BM25_WEIGHT, float)
        self.assertGreaterEqual(config.BM25_WEIGHT, 0)
        self.assertLessEqual(config.BM25_WEIGHT, 1.0)
        self.assertEqual(config.BM25_WEIGHT, 0.5)

    def test_vector_weight(self):
        self.assertIsInstance(config.VECTOR_WEIGHT, float)
        self.assertGreaterEqual(config.VECTOR_WEIGHT, 0)
        self.assertLessEqual(config.VECTOR_WEIGHT, 1.0)
        self.assertEqual(config.VECTOR_WEIGHT, 0.5)

    def test_ghost_history_window(self):
        self.assertIsInstance(config.GHOST_HISTORY_WINDOW, int)
        self.assertGreater(config.GHOST_HISTORY_WINDOW, 0)
        self.assertEqual(config.GHOST_HISTORY_WINDOW, 8)

    def test_ghost_history_max(self):
        self.assertIsInstance(config.GHOST_HISTORY_MAX, int)
        self.assertGreaterEqual(config.GHOST_HISTORY_MAX, config.GHOST_HISTORY_WINDOW)
        self.assertEqual(config.GHOST_HISTORY_MAX, 10)

    def test_sentinel_interval(self):
        self.assertIsInstance(config.SENTINEL_INTERVAL, int)
        self.assertGreater(config.SENTINEL_INTERVAL, 0)
        self.assertEqual(config.SENTINEL_INTERVAL, 5)

    def test_sentinel_token_threshold(self):
        self.assertIsInstance(config.SENTINEL_TOKEN_THRESHOLD, int)
        self.assertGreater(config.SENTINEL_TOKEN_THRESHOLD, 0)
        self.assertEqual(config.SENTINEL_TOKEN_THRESHOLD, 2000)

    def test_sentinel_max_tokens(self):
        self.assertIsInstance(config.SENTINEL_MAX_TOKENS, int)
        self.assertGreater(config.SENTINEL_MAX_TOKENS, 0)
        self.assertEqual(config.SENTINEL_MAX_TOKENS, 500)

    def test_max_history_tokens(self):
        self.assertIsInstance(config.MAX_HISTORY_TOKENS, int)
        self.assertGreater(config.MAX_HISTORY_TOKENS, 0)
        self.assertEqual(config.MAX_HISTORY_TOKENS, 2000)

    def test_ai_response_max_chars(self):
        self.assertIsInstance(config.AI_RESPONSE_MAX_CHARS, int)
        self.assertGreater(config.AI_RESPONSE_MAX_CHARS, 0)
        self.assertEqual(config.AI_RESPONSE_MAX_CHARS, 800)

    def test_ghost_ai_chars(self):
        self.assertIsInstance(config.GHOST_AI_CHARS, int)
        self.assertGreater(config.GHOST_AI_CHARS, 0)
        self.assertEqual(config.GHOST_AI_CHARS, 200)

    def test_default_model(self):
        self.assertIsInstance(config.DEFAULT_MODEL, str)
        self.assertNotEqual(config.DEFAULT_MODEL, "")
        self.assertEqual(config.DEFAULT_MODEL, "qwen/qwen-turbo")

    def test_pinned_relevance_threshold(self):
        self.assertIsInstance(config.PINNED_RELEVANCE_THRESHOLD, float)
        self.assertGreater(config.PINNED_RELEVANCE_THRESHOLD, 0)
        self.assertLessEqual(config.PINNED_RELEVANCE_THRESHOLD, 1.0)
        self.assertEqual(config.PINNED_RELEVANCE_THRESHOLD, 0.40)

    def test_sticky_pinned_context(self):
        self.assertIsInstance(config.STICKY_PINNED_CONTEXT, bool)
        self.assertEqual(config.STICKY_PINNED_CONTEXT, True)

    def test_min_prev_query_length(self):
        self.assertIsInstance(config.MIN_PREV_QUERY_LENGTH, int)
        self.assertGreater(config.MIN_PREV_QUERY_LENGTH, 0)
        self.assertEqual(config.MIN_PREV_QUERY_LENGTH, 15)

    def test_min_current_query_length(self):
        self.assertIsInstance(config.MIN_CURRENT_QUERY_LENGTH, int)
        self.assertGreater(config.MIN_CURRENT_QUERY_LENGTH, 0)
        self.assertEqual(config.MIN_CURRENT_QUERY_LENGTH, 10)

if __name__ == "__main__":
    unittest.main()
