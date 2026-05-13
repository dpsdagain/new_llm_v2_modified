import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import os
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Mocking config and constants before imports if necessary, 
# but usually rag_chain imports them.
import rag_chain
import config

class TestRagFull(unittest.TestCase):
    """
    Comprehensive suite for RAG Chain (Section 3: T-RC-3 to T-RC-11).
    """

    @classmethod
    def setUpClass(cls):
        # Clear lru_cache for deterministic testing of T-RC-3
        rag_chain._get_pinned_embedding.cache_clear()

    # --- T-RC-3: Pinned Embedding LRU Cache ---
    @patch("backend.get_embedding_model")
    def test_pinned_embedding_caching(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1, 0.2]
        mock_get_model.return_value = mock_model
        
        prefix = "stable_prefix"
        # 1st call: Cache miss
        emb1 = rag_chain._get_pinned_embedding(prefix)
        # 2nd call: Cache hit
        emb2 = rag_chain._get_pinned_embedding(prefix)
        
        info = rag_chain._get_pinned_embedding.cache_info()
        self.assertEqual(info.hits, 1)
        self.assertEqual(emb1, emb2)
        mock_model.embed_query.assert_called_once()

    # --- T-RC-4: Cosine Similarity ---
    def test_cosine_similarity(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        v3 = [0.0, 1.0]
        v4 = [-1.0, 0.0]
        v_zero = [0.0, 0.0]
        
        self.assertAlmostEqual(rag_chain.calculate_cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(rag_chain.calculate_cosine_similarity(v1, v3), 0.0)
        self.assertAlmostEqual(rag_chain.calculate_cosine_similarity(v1, v4), -1.0)
        self.assertEqual(rag_chain.calculate_cosine_similarity(v_zero, v1), 0.0)

    # --- T-RC-5: Deterministic Sort ---
    def test_deterministic_sort(self):
        docA = Document(page_content="A", metadata={"source": "A.py", "chunk_index": 0, "content_hash": "hA"})
        docB = Document(page_content="B", metadata={"source": "B.py", "chunk_index": 0, "content_hash": "hB"})
        docC = Document(page_content="C", metadata={"source": "C.py", "chunk_index": 0, "content_hash": "hC"})
        
        # Test 1: No stable hashes (straight sort by source)
        raw_list = [docC, docA, docB]
        sorted_list = rag_chain._sort_docs_deterministically(raw_list)
        self.assertEqual([d.page_content for d in sorted_list], ["A", "B", "C"])
        
        # Test 2: Stable hashes (Prefix Preservation)
        # Docs A and C were in the previous prompt. B is new.
        # Expected: [A, C] followed by [B] (if A < C alphabetically)
        stable = {"hA", "hC"}
        raw_list = [docB, docA, docC]
        sorted_list = rag_chain._sort_docs_deterministically(raw_list, stable_hashes=stable)
        self.assertEqual([d.page_content for d in sorted_list], ["A", "C", "B"])

    # --- T-RC-6: Singletons via Module Init ---
    def test_module_singletons(self):
        # Router
        r1 = rag_chain.get_router()
        r2 = rag_chain.get_router()
        self.assertIs(r1, r2)
        
        # Semantic Cache
        with patch("rag_chain.SemanticCache") as mock_sc:
            sc1 = rag_chain.get_semantic_cache()
            sc2 = rag_chain.get_semantic_cache()
            self.assertIs(sc1, sc2)

    # --- T-RC-7: VectorRouter classification & detection ---
    @patch("rag_chain.get_llm")
    def test_vector_router(self, mock_get_llm):
        router = rag_chain.get_router()
        
        # Specialty detection (Regex-based)
        self.assertEqual(router.detect_specialty("Write a python function"), "CODE")
        self.assertEqual(router.detect_specialty("Explain the architectural logic"), "REASONING")
        self.assertEqual(router.detect_specialty("Analyze this diagram"), "VISION")
        self.assertEqual(router.detect_specialty("What is the capital of France?"), "GENERAL")
        
        # Intent classification (LLM-based)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="FOLLOW-UP")
        mock_get_llm.return_value = mock_llm
        
        intent = router.classify_intent("Tell me more", [HumanMessage(content="Explain X")])
        self.assertEqual(intent, "FOLLOW-UP")

    # --- T-RC-8: LocalReRanker Cross-Encoder scoring ---
    @patch("rag_chain.CrossEncoder")
    def test_local_reranker(self, mock_ce_class):
        mock_ce = MagicMock()
        # Doc 1 has low relevance (0.1), Doc 2 has high (0.9)
        mock_ce.predict.return_value = [0.1, 0.9]
        mock_ce_class.return_value = mock_ce
        
        with patch("rag_chain.USE_RERANKER", True):
            reranker = rag_chain.LocalReRanker()
            docs = [
                Document(page_content="Low relevance", metadata={"id": 1}),
                Document(page_content="High relevance", metadata={"id": 2})
            ]
            ranked = reranker.rerank("Query", docs, top_k=1)
            
            self.assertEqual(len(ranked), 1)
            self.assertEqual(ranked[0].page_content, "High relevance")

    # --- T-RC-9: SemanticCache lookup/upsert ---
    @patch("rag_chain.Chroma")
    def test_semantic_cache(self, mock_chroma_class):
        mock_db = MagicMock()
        # Similarity score 0.98 (> 0.95 threshold)
        mock_db.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Query", metadata={"answer": "Cached Answer"}), 0.98)
        ]
        mock_chroma_class.return_value = mock_db
        
        cache = rag_chain.SemanticCache()
        ans = cache.lookup("Query", threshold=0.95)
        self.assertEqual(ans, "Cached Answer")
        
        # Test miss
        mock_db.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Query", metadata={"answer": "Cached Answer"}), 0.80)
        ]
        ans = cache.lookup("Query", threshold=0.95)
        self.assertIsNone(ans)

    # --- T-RC-10: Hybrid Search RRF Fusion ---
    @patch("rag_chain.SQLiteFTS5BM25")
    @patch("rag_chain.ENABLE_HYBRID_SEARCH", True)
    def test_hybrid_search_rrf(self, mock_fts_class):
        mock_db = MagicMock()
        # Vector results: [D1, D2]
        d1 = Document(page_content="D1", metadata={"source": "a.py", "chunk_index": 0, "content_hash": "h1"})
        d2 = Document(page_content="D2", metadata={"source": "b.py", "chunk_index": 0, "content_hash": "h2"})
        mock_db.similarity_search.return_value = [d1, d2]
        
        # FTS results: [D2, D3]
        d3 = Document(page_content="D3", metadata={"source": "c.py", "chunk_index": 0, "content_hash": "h3"})
        mock_fts = MagicMock()
        mock_fts.search.return_value = [d2, d3]
        mock_fts_class.return_value = mock_fts
        
        results = rag_chain.hybrid_search(mock_db, "Query", k=3)
        # D2 should be top because it's in both
        self.assertEqual(results[0].page_content, "D2")
        self.assertEqual(len(results), 3)

    # --- T-RC-11: Exact-Match Embedding Cache ---
    @patch("backend.get_embedding_model")
    @patch("rag_chain.get_router")
    @patch("rag_chain.get_llm")
    def test_exact_match_embedding_cache(self, mock_llm, mock_router, mock_get_model):
        chain = rag_chain.build_rag_chain(None, model="claude")
        
        # Input 1: Fresh query
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [1, 2, 3]
        mock_get_model.return_value = mock_model
        
        # Trigger the chain
        it = chain.stream({"input": "What is Python?", "full_source_context": ""})
        # Need to consume the iterator to trigger the logic
        for _ in it: pass 
        
        initial_call_count = mock_model.embed_query.call_count
        
        # Input 2: Identical query with last_query_embedding passed back
        it2 = chain.stream({
            "input": "What is Python?", 
            "last_query": "What is Python?",
            "last_query_embedding": [1, 2, 3],
            "full_source_context": ""
        })
        for _ in it2: pass
        
        # In Iteration 2:
        # 1. SemanticCache.lookup still calls embed_query (1 call)
        # 2. Main retrieval bypasses embed_query due to exact match logic (0 calls)
        # Total calls in It2 = 1.
        # Total overall = initial_call_count (2) + 1 = 3.
        self.assertEqual(mock_model.embed_query.call_count, initial_call_count + 1)

if __name__ == "__main__":
    unittest.main()
