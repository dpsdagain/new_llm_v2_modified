import os
import hashlib
from langchain_core.documents import Document
from backend import load_and_chunk_codebase, ingest_into_chroma, load_existing_chroma, delete_collection
from rag_chain import calculate_cosine_similarity, build_rag_chain
from config import CHROMA_DB_DIR

def test_zero_chunking():
    print("--- Verifying Zero-Chunking ---")
    test_file = "small_test.txt"
    test_dir = "test_ingest_dir"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a small file
    test_path = os.path.join(test_dir, test_file)
    content = "Hello world! This is a small file for zero chunking test."
    with open(test_path, "w") as f:
        f.write(content)
        
    # Ingest
    coll_name = "test_zero_chunk"
    delete_collection(coll_name)
    
    chunks = load_and_chunk_codebase(test_dir)
    print(f"Number of chunks created: {len(chunks)}")
    for c in chunks:
        print(f"Chunk Metadata: {c.metadata}")
        if c.metadata.get("zero_chunk"):
            print("Successfully identified as zero-chunk.")
            
    db, added = ingest_into_chroma(chunks, coll_name)
    print(f"Added to DB: {added}")
    
    # Clean up
    os.remove(test_path)
    os.rmdir(test_dir)
    delete_collection(coll_name)

def test_semantic_similarity():
    print("\n--- Verifying Semantic Similarity calculation ---")
    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.1, 0.2, 0.35]
    sim = calculate_cosine_similarity(vec1, vec2)
    print(f"Similarity (0.1,0.2,0.3) vs (0.1,0.2,0.35): {sim:.4f}")
    
    vec3 = [-0.1, -0.2, -0.3]
    sim_neg = calculate_cosine_similarity(vec1, vec3)
    print(f"Similarity (positive) vs (negative): {sim_neg:.4f}")

if __name__ == "__main__":
    test_zero_chunking()
    test_semantic_similarity()
