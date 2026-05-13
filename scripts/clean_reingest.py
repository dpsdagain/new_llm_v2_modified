import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from backend import ingest_into_chroma, load_and_chunk_codebase, CHROMA_DB_DIR, delete_collection
import chromadb
import shutil

def clean_reingest():
    print("--- Performing Clean Re-Ingest (v2_atomic) ---")
    
    # 1. Target current folder
    target_path = os.getcwd()
    print(f"Collecting and chunking {target_path}...")
    
    # Using the primary collection used by app.py
    collection_name = "default"
    print(f"Wiping collection '{collection_name}' for a clean start...")
    delete_collection(collection_name)
    
    # Loading
    # config.py EXCLUDED_FILE_PATTERNS will automatically skip the 'archive' folder
    chunks = load_and_chunk_codebase(target_path)
    if not chunks:
        print("FAIL: No chunks found to ingest.")
        return
        
    print(f"Ingesting {len(chunks)} chunks into '{collection_name}'...")
    db, added = ingest_into_chroma(chunks, collection_name)
    
    print("DONE: Clean Re-ingest complete.")
    print(f"SUCCESS: COLLECTION REBUILT: '{collection_name}' with {added} chunks.")

if __name__ == "__main__":
    clean_reingest()
