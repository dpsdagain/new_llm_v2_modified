import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from backend import _collect_code_files

def test_file_collection():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"--- 📂 Testing File Collection in: {root} ---")
    
    files = _collect_code_files(root)
    
    found_config = False
    for f in files:
        if "config.py" in f:
            found_config = True
            print(f"✅ FOUND: {f}")
    
    if not found_config:
        print("❌ MISSING: config.py was NOT collected.")
        # Check why
        from backend import _is_excluded
        config_path = os.path.join(root, "config.py")
        is_ex = _is_excluded(config_path)
        print(f"   _is_excluded('{config_path}') -> {is_ex}")
        
        from config import EXCLUDED_FILE_PATTERNS
        import fnmatch
        name = os.path.basename(config_path)
        full = config_path.replace("\\", "/")
        print(f"   Patterns check for '{name}':")
        for p in EXCLUDED_FILE_PATTERNS:
            if fnmatch.fnmatch(name, p) or fnmatch.fnmatch(full, p):
                print(f"   MATCHED PATTERN: {p}")

if __name__ == "__main__":
    test_file_collection()
