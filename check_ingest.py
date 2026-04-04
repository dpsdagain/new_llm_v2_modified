import os
import sys
from backend import load_and_chunk_codebase, _is_excluded
from config import CODE_EXTENSIONS

def check_ingest():
    print(f"Current Directory: {os.getcwd()}")
    files = [f for f in os.listdir(".") if os.path.isfile(f)]
    print(f"Files in dir: {files}")
    
    for f in ["config.py", "backend.py", "app.py", "rag_chain.py", "verify_logic.py"]:
        if os.path.exists(f):
            ext = os.path.splitext(f)[1].lower()
            excluded = _is_excluded(f)
            in_ext = ext in CODE_EXTENSIONS
            print(f"File: {f} | Ext: {ext} | In CODE_EXTENSIONS: {in_ext} | Excluded: {excluded}")
        else:
            print(f"File {f} DOES NOT EXIST at {os.path.abspath(f)}")

    print("\nRunning load_and_chunk_codebase('.')")
    chunks = load_and_chunk_codebase(".")
    sources = {c.metadata.get("source") for c in chunks}
    print(f"Total chunks: {len(chunks)}")
    print(f"Core files in chunks: {[f for f in sources if os.path.basename(f) in ['config.py', 'backend.py', 'app.py', 'rag_chain.py']]}")

if __name__ == "__main__":
    check_ingest()
