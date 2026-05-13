import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config
import backend
import rag_chain

class TestMasterV2(unittest.TestCase):

    # --- 1. Parameter Validation Tests (T-CFG) ---

    def test_t_cfg_1_numeric(self):
        print("Running T-CFG-1...")
        self.assertEqual(config.MAX_TOKENS, 4096)
        self.assertEqual(config.LLM_TEMPERATURE, 0.0)
        self.assertEqual(config.CHUNK_SIZE, 1500)
        self.assertEqual(config.CHUNK_OVERLAP, 200)
        self.assertEqual(config.ZERO_CHUNK_THRESHOLD, 100000)
        self.assertEqual(config.MAX_ZERO_CHUNK_CHARS, 9000)
        self.assertEqual(config.RETRIEVER_K, 6)
        self.assertEqual(config.SEMANTIC_CACHE_THRESHOLD, 0.85)
        self.assertEqual(config.SENTINEL_INTERVAL, 3)
        self.assertEqual(config.SENTINEL_TOKEN_THRESHOLD, 1500)
        self.assertEqual(config.RERANK_CANDIDATES, 15)
        self.assertAlmostEqual(config.BM25_WEIGHT + config.VECTOR_WEIGHT, 1.0)
        print("T-CFG-1 Passed.")

    def test_t_cfg_2_bool(self):
        print("Running T-CFG-2...")
        self.assertTrue(config.ENABLE_PROMPT_CACHING)
        self.assertTrue(config.TRUST_NATIVE_CACHE)
        self.assertTrue(config.STICKY_PINNED_CONTEXT)
        self.assertTrue(config.ENABLE_HYBRID_SEARCH)
        self.assertTrue(config.USE_RERANKER)
        print("T-CFG-2 Passed.")

    def test_t_cfg_3_profiles(self):
        print("Running T-CFG-3...")
        for provider, (max_bp, min_tok) in config.PROVIDER_CACHE_PROFILES.items():
            self.assertIn(max_bp, (4, 8))
            self.assertGreaterEqual(min_tok, 1024)
        self.assertEqual(config.PROVIDER_CACHE_PROFILES["claude"], (4, 1024))
        self.assertEqual(config.PROVIDER_CACHE_PROFILES["gemini"], (8, 1028))
        print("T-CFG-3 Passed.")

    # --- 2. Backend Logic (T-BE) ---

    def test_t_be_2_content_hash(self):
        print("Running T-BE-2...")
        doc_a = Document(page_content="hello world", metadata={"source": "a.py"})
        doc_b = Document(page_content="hello world", metadata={"source": "b.py"})
        hash_a = backend._content_hash(doc_a)
        hash_b = backend._content_hash(doc_b)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(len(hash_a), 64)
        print("T-BE-2 Passed.")

    def test_t_be_7_exclusions(self):
        print("Running T-BE-7...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "app.py"), "w") as f: f.write("print('hello')")
            with open(os.path.join(tmp_dir, ".env"), "w") as f: f.write("SECRET=xxx")
            os.mkdir(os.path.join(tmp_dir, "archive"))
            with open(os.path.join(tmp_dir, "archive", "old.py"), "w") as f: f.write("pass")
            
            files = backend._collect_code_files(tmp_dir)
            filenames = [os.path.basename(f) for f in files]
            self.assertIn("app.py", filenames)
            self.assertNotIn(".env", filenames)
            # archive is in EXCLUDED_FILE_PATTERNS as "*archive*"
            self.assertNotIn("old.py", filenames)
        print("T-BE-7 Passed.")

    # --- 3. RAG Chain Logic (T-RC) ---

    def test_t_rc_1_cache_detection(self):
        print("Running T-RC-1...")
        self.assertTrue(rag_chain.is_cache_capable("claude-3.5-sonnet"))
        self.assertTrue(rag_chain.is_cache_capable("google/gemini-2.0-flash"))
        self.assertFalse(rag_chain.is_cache_capable("qwen/qwen3-coder:free"))
        print("T-RC-1 Passed.")

    def test_t_rc_2_normalization(self):
        print("Running T-RC-2...")
        self.assertEqual(rag_chain._normalize_query("What is Python?"), "what is python")
        self.assertEqual(rag_chain._normalize_query("  Hello, World!  "), "hello world")
        print("T-RC-2 Passed.")

    def test_t_rc_4_cosine(self):
        print("Running T-RC-4...")
        self.assertEqual(rag_chain.calculate_cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertEqual(rag_chain.calculate_cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(rag_chain.calculate_cosine_similarity([0, 0], [1, 0]), 0.0)
        print("T-RC-4 Passed.")

    def test_t_rc_12_max_tokens(self):
        print("Running T-RC-12...")
        self.assertEqual(rag_chain._get_max_tokens("CODE", "short"), config.MAX_TOKENS)
        self.assertEqual(rag_chain._get_max_tokens("GENERAL", "short"), 1024)
        self.assertEqual(rag_chain._get_max_tokens("GENERAL", "x" * 201), config.MAX_TOKENS)
        print("T-RC-12 Passed.")

    # --- 4. History (T-HIST) ---

    def test_t_hist_5_sentinel_trigger(self):
        print("Running T-HIST-5...")
        history = [HumanMessage(content="w" * 2250), AIMessage(content="w" * 2250)]
        self.assertGreaterEqual(rag_chain._est_tokens(history), config.SENTINEL_TOKEN_THRESHOLD)
        print("T-HIST-5 Passed.")

    # --- 5. Token Optimization (T-TOK) ---

    def test_t_tok_1_zero_chunk_filter(self):
        print("Running T-TOK-1...")
        oversized = Document(
            page_content="x" * 10001,
            metadata={"source": "big.py", "zero_chunk": True}
        )
        normal = Document(
            page_content="def foo(): pass",
            metadata={"source": "small.py", "zero_chunk": False}
        )
        docs = [oversized, normal]
        filtered = [
            d for d in docs
            if not (d.metadata.get("zero_chunk") and len(d.page_content) > config.MAX_ZERO_CHUNK_CHARS)
        ]
        self.assertNotIn(oversized, filtered)
        self.assertIn(normal, filtered)
        print("T-TOK-1 Passed.")

    # --- 6. Integration (T-INGEST) ---

    def test_t_ingest_smoke(self):
        print("Running T-INGEST smoke...")
        coll_name = "test_smoke_v2"
        backend.delete_collection(coll_name)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "test.py"), "w") as f: f.write("def test_func(): pass")
            
            docs = backend.load_and_chunk_codebase(tmp_dir)
            db, added = backend.ingest_into_chroma(docs, coll_name)
            
            self.assertGreaterEqual(added, 1)
        
        backend.delete_collection(coll_name)
        print("T-INGEST Passed.")

    # --- 7. Router Specialty Detection (T-RC-7) ---

    def test_t_rc_7_specialty(self):
        print("Running T-RC-7...")
        router = rag_chain.get_router()
        self.assertEqual(router.detect_specialty("write a python function"), "CODE")
        self.assertEqual(router.detect_specialty("analyze architectural trade-offs"), "REASONING")
        self.assertEqual(router.detect_specialty("what is the capital of France"), "GENERAL")
        print("T-RC-7 Passed.")

if __name__ == "__main__":
    unittest.main()
