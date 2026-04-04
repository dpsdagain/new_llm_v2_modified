import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend import load_and_chunk_codebase
from config import ZERO_CHUNK_THRESHOLD, CODE_CHUNK_SIZE

def run_verification():
    print("--- 🔬 SYSTEM INTEGRITY VERIFICATION ---")
    test_dir = "test_verification"
    os.makedirs(test_dir, exist_ok=True)

    # 1. 📂 Stage 1: Zero-Chunking Verification (< 500k)
    print("\n--- STAGE 1: Small File Proof ---")
    small_path = os.path.join(test_dir, "utils.py")
    small_content = """import os\n\ndef my_small_function():\n    return "I am visible to the AI in full." """
    with open(small_path, "w") as f:
        f.write(small_content)
    
    chunks = load_and_chunk_codebase(test_dir)
    print(f"File Size: {len(small_content)} chars")
    print(f"Zero Chunking Threshold: {ZERO_CHUNK_THRESHOLD} chars")
    print(f"Chunks generated: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} Metadata: Zero-Chunking={c.metadata.get('zero_chunk', False)}")
        print(f"Chunk {i} Content: {c.page_content[:50]}...")
    
    # 2. 📂 Stage 2: Syntax-Aware Splitting Verification (> 500k)
    print("\n--- STAGE 2: Massive File Syntax Splitting ---")
    # Clean up small file
    os.remove(small_path)
    
    large_path = os.path.join(test_dir, "mega_logic.py")
    # Generate a massive file (~600k) that exceeds threshold
    # Each function is ~2k chars. We need ~300 of them.
    functions = []
    for i in range(400):
        functions.append(f"def massive_function_{i}():\n    # This is a large repeated function to bloat the file\n" + "    print('padding' * 50)\n" * 5)
    
    large_content = "\n".join(functions)
    with open(large_path, "w") as f:
        f.write(large_content)
        
    print(f"Large File Size: {len(large_content)} chars")
    chunks = load_and_chunk_codebase(test_dir)
    print(f"Chunks generated for massive file: {len(chunks)}")
    
    # Verify split point of chunk 1
    if len(chunks) > 1:
        # Check the start and end of a middle chunk
        m_chunk = chunks[1]
        print(f"Chunk 1 Metadata: {m_chunk.metadata.get('source')} [index {m_chunk.metadata.get('chunk_index')}]")
        print(f"Chunk 1 Starting characters: {m_chunk.page_content[:50].strip()}...")
        # Check for clean function break
        if m_chunk.page_content.startswith("def "):
            print("✅ SUCCESS: Chunk starts with a clean function definition.")
        else:
            print("⚠️ WARNING: Chunk starts mid-line (check syntax splitter).")
            
    # Clean up
    os.remove(large_path)
    os.rmdir(test_dir)
    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    run_verification()
