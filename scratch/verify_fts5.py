import sqlite3
import os

db_path = './chroma_db/v2_atomic_fts5.db'
if not os.path.exists(db_path):
    print(f"ERROR: {db_path} does not exist.")
    exit(1)

conn = sqlite3.connect(db_path)
cur = conn.execute('SELECT calls FROM docs_fts WHERE calls != "" LIMIT 10')
rows = cur.fetchall()

print(f"Found {len(rows)} samples with metadata:")
for r in rows:
    print(f"- {r[0]}")
