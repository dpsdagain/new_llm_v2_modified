import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import os
import json
from config import CHROMA_DB_DIR

def test_fts5_filename():
    db_path = os.path.join(CHROMA_DB_DIR, "default_fts5.db")
    if not os.path.exists(db_path):
        print(f"Index not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Check schema
    try:
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name='docs_fts'")
        print(f"Schema: {cursor.fetchone()[0]}")
    except Exception as e:
        print(f"Error checking schema: {e}")
        
    # Test queries
    queries = ["rag_chain", "backend", "app", "config"]
    for q in queries:
        print(f"\n--- Searching for: {q} ---")
        try:
            # Note: SQLite FTS5 rank uses BM25
            rows = conn.execute(
                "SELECT source_name, content FROM docs_fts WHERE docs_fts MATCH ? LIMIT 3", 
                (q,)
            ).fetchall()
            for src, content in rows:
                print(f"Found in: {src} | Snippet: {content[:100].replace('\n', ' ')}...")
        except Exception as e:
            print(f"Search failed: {e}")
            
    conn.close()

if __name__ == "__main__":
    test_fts5_filename()
