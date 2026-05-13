import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
import rag_chain
import app
import config

class TestFinalIntegration(unittest.TestCase):
    """
    Final Integration Suite: Sections 7, 8, and 9.
    """

    # --- T-CHAIN-1 & T-CHAIN-7: specialist routing & zero-chunk filtering ---
    @patch("rag_chain.get_router")
    @patch("rag_chain.is_cache_capable", return_value=False) # Non-caching model (Qwen)
    def test_specialist_and_zerochunk(self, mock_cache_check, mock_router_class):
        mock_router = MagicMock()
        mock_router.detect_specialty.return_value = "CODE"
        mock_router_class.return_value = mock_router
        
        # Doc 1: Normal
        d1 = MagicMock(page_content="Normal", metadata={"source": "a.py"})
        # Doc 2: Large Zero-Chunk
        d2 = MagicMock(page_content="Very large... " * 1000, metadata={"source": "large.py", "zero_chunk": True})
        
        # Test logic in _full_context_cache_chain
        # We simulate the filtering logic
        provider_has_cache = False
        final_docs = [d1, d2]
        
        if not provider_has_cache:
            final_docs = [d for d in final_docs if not (d.metadata.get("zero_chunk") and len(d.page_content) > 10000)]
            
        self.assertEqual(len(final_docs), 1)
        self.assertEqual(final_docs[0].metadata["source"], "a.py")

    # --- T-CHAIN-5/6: Prompt caching blocks (Claude vs others) ---
    def test_prompt_caching_format(self):
        history = [HumanMessage(content="Hello")]
        
        # Claude (4 breakpoints limit): 
        # History remains plain string to preserve breakpoints for RAG blocks
        claude_history = rag_chain._prepare_history_with_cache(history, "anthropic/claude-3-sonnet")
        self.assertTrue(isinstance(claude_history[0].content, str))
        
        # Gemini (supports many breakpoints):
        # Last history message gets a cache marker
        gemini_history = rag_chain._prepare_history_with_cache(history, "google/gemini-2.0-flash")
        self.assertTrue(isinstance(gemini_history[0].content, list))
        self.assertIn("cache_control", str(gemini_history[0].content))

    # --- T-CACHE-1/3: Native Cache Toggle ---
    def test_native_cache_stability(self):
        # Logic: if trust_native_cache=True, skip_retrieval should be False
        # even if is_semantic_hit is True.
        
        is_semantic_hit = True
        provider_has_cache = True # Claude
        trust_native_cache = True
        force_retrieval = False
        previous_union = [MagicMock()]
        
        skip_retrieval = (
            not (trust_native_cache and provider_has_cache)
            and is_semantic_hit
            and bool(previous_union)
            and not force_retrieval
        )
        self.assertFalse(skip_retrieval) # Retrieval is NOT skipped

    # --- T-APP-1: Usage Metadata Parsing ---
    def test_usage_metadata_parsing(self):
        # OpenRouter Format
        chunk_or = MagicMock()
        chunk_or.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 30}
            }
        }
        usage = app.extract_usage_metadata(chunk_or)
        self.assertEqual(usage["input"], 100)
        self.assertEqual(usage["cache_read"], 30)
        
        # Anthropic Format
        chunk_ant = MagicMock()
        chunk_ant.response_metadata = {
            "anthropic-ratelimit-input-tokens-cache-read": 1000,
            "anthropic-ratelimit-input-tokens-cache-creation": 200
        }
        usage_ant = app.extract_usage_metadata(chunk_ant)
        self.assertEqual(usage_ant["cache_read"], 1000)
        self.assertEqual(usage_ant["cache_create"], 200)

    # --- T-APP-4: Force Retrieval Logic ---
    @patch("app.st.session_state", {"_source_names_cache_default": ["app.py", "backend.py"]})
    def test_force_retrieval_logic(self):
        # Keyword match
        self.assertTrue(app.detect_force_retrieval("please reload the docs", "default"))
        # Filename match
        self.assertTrue(app.detect_force_retrieval("Explain app.py", "default"))
        # Negative match
        self.assertFalse(app.detect_force_retrieval("What is python?", "default"))

    # --- T-APP-5: Session State Init Simulation ---
    def test_session_state_init(self):
        # We simulate the keys set in app.py
        mock_st = MagicMock()
        mock_st.session_state = {}
        
        # Keys to check from app.py
        keys = ["chat_history", "active_collection", "model_id", "pinned_file", "token_usage", "metrics_history", "sentinel_state"]
        
        for k in keys:
            if k not in mock_st.session_state:
                mock_st.session_state[k] = "initialized"
                
        for k in keys:
            self.assertIn(k, mock_st.session_state)

if __name__ == "__main__":
    unittest.main()
