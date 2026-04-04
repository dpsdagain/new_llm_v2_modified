import os
import hashlib
from langchain_core.documents import Document
from backend import load_and_chunk_codebase
from config import ZERO_CHUNK_THRESHOLD

def test_zero_chunking_logic():
    print("--- Verifying Zero-Chunking Logic ---")
    test_dir = "test_logic_dir"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a small file
    test_path = os.path.join(test_dir, "small.py")
    content = "print('hello world')"
    with open(test_path, "w") as f:
        f.write(content)
        
    chunks = load_and_chunk_codebase(test_dir)
    print(f"File size: {len(content)}")
    print(f"Threshold: {ZERO_CHUNK_THRESHOLD}")
    print(f"Number of chunks: {len(chunks)}")
    
    if len(chunks) == 1 and chunks[0].metadata.get("zero_chunk"):
        print("SUCCESS: Zero-chunking works correctly.")
    else:
        print("FAILURE: Zero-chunking did not trigger.")
        
    # Clean up
    os.remove(test_path)
    os.rmdir(test_dir)

if __name__ == "__main__":
    test_zero_chunking_logic()
