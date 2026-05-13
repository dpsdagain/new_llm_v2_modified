import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import os
import shutil
import time
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mocking environment before imports
os.environ["CHROMA_DB_DIR"] = "test_ingest_db"
import backend
import config

class TestIngestIntegration(unittest.TestCase):
    """
    Integration Tests for Ingestion Pipeline (Section 5: T-ING-1 to T-ING-5).
    """

    @classmethod
    def setUpClass(cls):
        if os.path.exists("test_ingest_db"):
            shutil.rmtree("test_ingest_db")
        os.makedirs("test_ingest_db", exist_ok=True)
        # Monkeypatch backend constants
        backend.CHROMA_DB_DIR = "test_ingest_db"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test_ingest_db"):
            try:
                shutil.rmtree("test_ingest_db")
            except:
                pass

    def setUp(self):
        # Mock embedding model for performance and cost
        self.mock_emb = MagicMock()
        self.mock_emb.embed_query.return_value = [0.1] * 384
        self.mock_emb.embed_documents.return_value = [[0.1] * 384] * 10
        
        self.patcher = patch("backend.get_embedding_model", return_value=self.mock_emb)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    # --- T-ING-1: End-to-end single file ingestion ---
    def test_end_to_end_file(self):
        coll = "single_file_test"
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "test.py")
            with open(fpath, "w") as f:
                f.write("def hello():\n    print('world')")
            
            chunks = backend.load_and_chunk_codebase(td)
            db, added = backend.ingest_into_chroma(chunks, coll)
            
            self.assertGreaterEqual(added, 1)
            # Check FTS5
            fts_path = os.path.join("test_ingest_db", f"{coll}_fts5.db")
            self.assertTrue(os.path.exists(fts_path))

    # --- T-ING-2: End-to-end directory ingestion ---
    def test_directory_ingestion(self):
        coll = "dir_test"
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            with open(os.path.join(td, "src", "app.py"), "w") as f: f.write("class App: pass")
            with open(os.path.join(td, ".env"), "w") as f: f.write("SECRET=123")
            
            chunks = backend.load_and_chunk_codebase(td)
            sources = [c.metadata["source"] for c in chunks]
            
            # .env should be excluded
            self.assertTrue(any("app.py" in s for s in sources))
            self.assertFalse(any(".env" in s for s in sources))

    # --- T-ING-3: Re-ingestion after file modification ---
    def test_incremental_ingestion(self):
        coll = "inc_test"
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "mod.py")
            with open(fpath, "w") as f: f.write("v1 content")
            
            chunks1 = backend.load_and_chunk_codebase(td)
            _, added1 = backend.ingest_into_chroma(chunks1, coll)
            
            # Modify
            with open(fpath, "w") as f: f.write("v2 content - totally different")
            chunks2 = backend.load_and_chunk_codebase(td)
            _, added2 = backend.ingest_into_chroma(chunks2, coll)
            
            # Should have added more chunks (or at least 1 for the new content)
            self.assertGreaterEqual(added2, 1)

    # --- T-ING-4: AsyncIngestionTask Lifecycle ---
    def test_async_task_lifecycle(self):
        coll = "async_test"
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "run.py"), "w") as f: f.write("print('run')")
            
            task = backend.AsyncIngestionTask(td, coll)
            self.assertEqual(task.status, "pending")
            
            task.start()
            
            # Poll with timeout
            start_time = time.time()
            while not task.is_done and time.time() - start_time < 10:
                time.sleep(0.1)
            
            self.assertEqual(task.status, "done")
            self.assertGreater(task.progress, 0.9)
            self.assertIsNotNone(task.result)

    # --- T-ING-5: summarize_document_for_pin ---
    def test_document_summarization(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write('"""Module doc"""\nclass MyClass:\n    """Class doc"""\n    def my_method(self):\n        pass\n\ndef my_func():\n    # code\n    pass')
            tf_path = tf.name
        
        try:
            summary = backend.summarize_document_for_pin(tf_path, max_chars=500)
            self.assertIn("class MyClass", summary)
            self.assertIn("def my_method", summary)
            self.assertIn("def my_func", summary)
            # It should not contain the internal logic if it's long, 
            # but here it's short so it might. 
            # The focus is on signatures.
        finally:
            os.remove(tf_path)

if __name__ == "__main__":
    unittest.main()
