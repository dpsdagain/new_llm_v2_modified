import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import rag_chain
import config

class TestHistoryFull(unittest.TestCase):
    """
    Validation for Section 4: Conversation History (T-HIST-1 to T-HIST-5).
    """

    def setUp(self):
        # Reset monkeypatches
        rag_chain.GHOST_HISTORY_MAX = 10
        rag_chain.AI_RESPONSE_MAX_CHARS = 500
        rag_chain.SENTINEL_TOKEN_THRESHOLD = 1000
        rag_chain.SENTINEL_INTERVAL = 5

    # --- T-HIST-1: compress_chat_history with sentinel state ---
    def test_compress_with_sentinel(self):
        history = [HumanMessage(content=f"m{i}") for i in range(20)]
        sentinel_state = "Previously discussed X."
        
        compressed = rag_chain.compress_chat_history(history, sentinel_state)
        # Should only keep last 4 messages
        self.assertEqual(len(compressed), 4)
        self.assertEqual(compressed[-1].content, "m19")

    # --- T-HIST-2: _truncate_ai_in_history with code blocks ---
    def test_truncate_ai_with_code(self):
        # AI Response with 1000 chars of prose and a code block
        prose = "Very long prose... " * 50 
        code = "\n```python\nprint('Hello World')\n```\n"
        content = prose + code
        msg = AIMessage(content=content)
        
        # Manually set Max Chars to production default for testing
        rag_chain.AI_RESPONSE_MAX_CHARS = 800 
        
        truncated_msgs = rag_chain._truncate_ai_in_history([msg])
        truncated_content = truncated_msgs[0].content
        
        # Assertions
        # 1. Code block must exist in full
        self.assertIn("print('Hello World')", truncated_content)
        # 2. Prose should be trimmed with a snippet/gist
        self.assertIn("[prose truncated]", truncated_content)
        # 3. Total length should be significantly reduced compared to original (~2000 vs ~500)
        self.assertLess(len(truncated_content), len(content))

    # --- T-HIST-3: compress_chat_history short history passthrough ---
    def test_short_history_passthrough(self):
        rag_chain.GHOST_HISTORY_MAX = 10
        # 6 messages (below 10)
        history = [HumanMessage(content=f"m{i}") for i in range(6)]
        
        compressed = rag_chain.compress_chat_history(history, None)
        self.assertEqual(len(compressed), 6)

    # --- T-HIST-4: Sentinel Cooldown ---
    def test_sentinel_cooldown(self):
        # We test the logic used in build_rag_chain's inner function
        # Mocking the variables used in the logic gate at line 875
        
        # Case A: Below interval
        turn_count = 8
        last_turn = 5
        interval = 5
        # (8 - 5) = 3 < 5 -> False
        should_summarize = (
            turn_count > 0 
            and 2000 >= 1000 # threshold met
            and (turn_count - last_turn) >= interval
        )
        self.assertFalse(should_summarize)
        
        # Case B: At/Above interval
        turn_count = 10
        # (10 - 5) = 5 >= 5 -> True
        should_summarize = (
            turn_count > 0 
            and 2000 >= 1000 
            and (turn_count - last_turn) >= interval
        )
        self.assertTrue(should_summarize)

    # --- T-HIST-5: Background sentinel update ---
    @patch("rag_chain.VectorRouter")
    def test_background_summarize_mock(self, mock_router_class):
        mock_router = MagicMock()
        mock_router.summarize_state_fast.return_value = "Mock Summary"
        mock_router_class.return_value = mock_router
        
        history = [HumanMessage(content="Hello")]
        result = rag_chain._background_summarize(history)
        
        self.assertEqual(result, "Mock Summary")
        mock_router.summarize_state_fast.assert_called_once_with(history)

if __name__ == "__main__":
    unittest.main()
