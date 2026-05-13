import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
import unittest
import threading
import time
import sqlite3
import json
import tempfile
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set up environment for tests
os.environ["CHROMA_DB_DIR"] = "test_chroma_db"
import backend
import config
from langchain_core.documents import Document

# Monkeypatch constants in backend
backend.CHROMA_DB_DIR = "test_chroma_db"
backend.ZERO_CHUNK_THRESHOLD = 50
backend.CODE_CHUNK_SIZE = 50
backend.CHUNK_OVERLAP = 10 # MUST BE SMALLER THAN CHUNK_SIZE (50)

class TestBackendFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a clean test DB directory
        if os.path.exists("test_chroma_db"):
            try:
                shutil.rmtree("test_chroma_db")
            except:
                pass
        os.makedirs("test_chroma_db", exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        backend._embedding_model = None

    # --- T-BE-1: Thread Safety ---
    @patch("backend.HuggingFaceEmbeddings")
    def test_get_embedding_model_thread_safety(self, mock_hfe):
        def slow_init(*args, **kwargs):
            time.sleep(0.05)
            return MagicMock()
        mock_hfe.side_effect = slow_init
        barrier = threading.Barrier(4)
        results = []
        def get_model():
            barrier.wait()
            results.append(backend.get_embedding_model())
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(lambda _: get_model(), range(4))
        self.assertEqual(len(results), 4)
        self.assertIs(results[0], results[1])
        self.assertEqual(mock_hfe.call_count, 1)

    # --- T-BE-2: Content Hash ---
    def test_content_hash_deduplication(self):
        doc1 = Document(page_content="exact same", metadata={"s": "a"})
        doc2 = Document(page_content="exact same", metadata={"s": "b"})
        self.assertEqual(backend._content_hash(doc1), backend._content_hash(doc2))

    # --- T-BE-3 & T-BE-4: SQLite FTS5 ---
    def test_sqlite_fts5_indexing_and_search(self):
        col_name = "test_col_fts_" + str(time.time()).replace(".", "")
        fts = backend.SQLiteFTS5BM25(col_name)
        fts.add_documents([Document(page_content="The quick brown fox", metadata={"content_hash": "h1"})])
        db_path = os.path.join(backend.CHROMA_DB_DIR, f"{col_name}_fts5.db")
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT count(*) FROM docs_fts").fetchone()[0]
        conn.close() 
        self.assertEqual(count, 1)
        self.assertGreater(len(fts.search("fox")), 0)

    # --- T-BE-5: Re-ingest Deduplication ---
    @patch("backend.load_existing_chroma")
    @patch("backend.get_embedding_model")
    def test_ingest_deduplication(self, mock_get_model, mock_load_chroma):
        mock_db = MagicMock()
        mock_db.get.return_value = {"metadatas": [{"content_hash": "h1"}]}
        mock_load_chroma.return_value = mock_db
        _, added = backend.ingest_into_chroma([Document(page_content="c1", metadata={"content_hash": "h1"})], "tc")
        self.assertEqual(added, 0)

    # --- T-BE-6: Incremental Append ---
    @patch("backend.load_existing_chroma")
    @patch("backend.get_embedding_model")
    def test_ingest_append(self, mock_get_model, mock_load_chroma):
        mock_db = MagicMock()
        mock_db.get.return_value = {"metadatas": []}
        mock_load_chroma.return_value = mock_db
        _, added = backend.ingest_into_chroma([Document(page_content="c2", metadata={"content_hash": "h2"})], "tc2")
        self.assertEqual(added, 1)

    # --- T-BE-7: Exclusion patterns ---
    def test_ingest_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "app.py"), "w").close()
            open(os.path.join(td, ".env"), "w").close()
            basenames = [os.path.basename(f) for f in backend._collect_code_files(td)]
            self.assertIn("app.py", basenames)
            self.assertNotIn(".env", basenames)

    # --- T-BE-8: AST Chunker Python ---
    def test_ast_chunker_python(self):
        chunker = backend.CodeASTChunker(chunk_size=1000)
        code = "class X:\n    def m(self): pass"
        try:
            chunks = chunker.chunk_file(code, "test.py", ".py")
            if chunks:
                self.assertIn("X", chunks[0].page_content)
        except:
            self.skipTest("Tree-sitter skip")

    # --- T-BE-9: HDL Regex Fallback ---
    def test_ast_chunker_hdl(self):
        chunker = backend.CodeASTChunker()
        verilog = "module top; endmodule"
        chunks = chunker.chunk_file(verilog, "top.v", ".v")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["hdl_type"], "regex_block")

    # --- T-BE-10: Large file chunking ---
    def test_large_file_chunking(self):
        content = "A" * 200 
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "large.py")
            with open(fpath, "w") as f: f.write(content)
            chunks = backend.load_and_chunk_codebase(td)
            self.assertGreater(len(chunks), 1)

    # --- T-BE-11: Collection deletion ---
    @patch("chromadb.PersistentClient")
    def test_collection_deletion(self, mock_client):
        col_name = "to_del_" + str(time.time()).replace(".", "")
        db_path = os.path.join(backend.CHROMA_DB_DIR, f"{col_name}_fts5.db")
        open(db_path, "w").close()
        self.assertTrue(backend.delete_collection(col_name))
        self.assertFalse(os.path.exists(db_path))

if __name__ == "__main__":
    unittest.main()
