import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import unittest
import fnmatch

class TestConfigMappingsAndExtensions(unittest.TestCase):
    """
    T-CFG-3: SPECIALIST_MAPPING completeness.
    T-CFG-4: CODE_EXTENSIONS and EXCLUDED_FILE_PATTERNS validation.
    """

    # --- T-CFG-3 Tests ---
    def test_specialist_mapping_completeness(self):
        expected_keys = {"CODE", "REASONING", "VISION", "GENERAL"}
        self.assertEqual(set(config.SPECIALIST_MAPPING.keys()), expected_keys)

    def test_specialist_mapping_model_strings(self):
        # CODE & REASONING map to google/gemma-4-...
        self.assertTrue(config.SPECIALIST_MAPPING["CODE"].startswith("google/gemma-4"))
        self.assertTrue(config.SPECIALIST_MAPPING["REASONING"].startswith("google/gemma-4"))
        
        # VISION maps to rekaai/reka-edge
        self.assertEqual(config.SPECIALIST_MAPPING["VISION"], "rekaai/reka-edge")
        
        # GENERAL maps to nvidia/nemotron-3...
        self.assertTrue(config.SPECIALIST_MAPPING["GENERAL"].startswith("nvidia/nemotron-3"))

    # --- T-CFG-4 Tests ---
    def test_code_extensions_inclusion(self):
        required_exts = {".py", ".js", ".ts", ".java", ".cpp", ".v", ".sv", ".html", ".sql"}
        for ext in required_exts:
            self.assertIn(ext, config.CODE_EXTENSIONS, f"Missing required extension: {ext}")

    def test_excluded_file_patterns_inclusion(self):
        required_patterns = {"*.env", "*node_modules*", "*chroma_db*", "*.lock"}
        # Note: config.py has *-lock.json and *.lock which covers *.lock conceptually if we consider wildcard expansion, 
        # but the test asks for these specific patterns to be present in the logic.
        
        # We check if any pattern in EXCLUDED_FILE_PATTERNS matches the required pattern or covers it.
        # However, for the test we just check if they are explicitly mentioned or similar.
        actual_patterns = set(config.EXCLUDED_FILE_PATTERNS)
        self.assertIn("*.env", actual_patterns)
        self.assertIn("*node_modules*", actual_patterns)
        self.assertIn("*chroma_db*", actual_patterns)
        self.assertIn("*.lock", actual_patterns)

    def test_no_overlap_between_extensions_and_exclusions(self):
        # "No overlap: no extension in CODE_EXTENSIONS matches any pattern in EXCLUDED_FILE_PATTERNS"
        for ext in config.CODE_EXTENSIONS:
            # We treat the extension as a filename (e.g. "test.py") to see if it's excluded
            filename = f"test{ext}"
            for pattern in config.EXCLUDED_FILE_PATTERNS:
                self.assertFalse(fnmatch.fnmatch(filename, pattern), 
                                f"Extension {ext} (test-file: {filename}) is excluded by pattern {pattern}")

if __name__ == "__main__":
    unittest.main()
