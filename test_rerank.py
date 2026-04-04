import os
import sys
from langchain_core.documents import Document
from rag_chain import LocalReRanker

def test_reranking():
    print("--- 🎯 Phase 3 Re-ranking Verification ---")
    reranker = LocalReRanker()
    
    query = "How does the zero-chunking threshold work in this code?"
    
    # Simulate a mix of relevant and irrelevant chunks
    docs = [
        Document(page_content="The ZERO_CHUNK_THRESHOLD is set to 500k chars in config.py. Files under this size are kept as one chunk.", metadata={"source": "relevant_file.py"}),
        Document(page_content="ChromaDB is a vector database used for storing document embeddings and metadata.", metadata={"source": "irrelevant_file.py"}),
        Document(page_content="The ingest_into_chroma function handles adding documents to the database and checking for existing hashes.", metadata={"source": "semi_relevant.py"}),
        Document(page_content="This is a completely random piece of text about weather in London.", metadata={"source": "random.txt"}),
    ]
    
    print(f"Original order: {[d.metadata['source'] for d in docs]}")
    
    reranked = reranker.rerank(query, docs, top_k=2)
    
    print(f"Reranked order (Top 2): {[d.metadata['source'] for d in reranked]}")
    
    if reranked[0].metadata['source'] == "relevant_file.py":
        print("✅ SUCCESS: Re-ranker correctly prioritized the most relevant document.")
    else:
        print("❌ FAILURE: Re-ranker did not prioritize correctly.")

if __name__ == "__main__":
    test_reranking()
