import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend import list_collections, get_collection_info
    
    print("--- 🔍 Collections on Disk ---")
    colls = list_collections()
    if not colls:
        print("No collections found.")
    else:
        for c in colls:
            info = get_collection_info(c)
            print(f"- {c}: {info['count']} chunks, {len(info['sources'])} sources")
    print("------------------------------")
except Exception as e:
    print(f"Error checking collections: {e}")
