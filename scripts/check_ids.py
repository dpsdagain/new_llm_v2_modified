import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

def test_model_id(model_id):
    print(f"Testing {model_id}...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=5)
        if resp.status_code == 200:
            print(f"✅ Success: {model_id}")
            return True
        else:
            print(f"❌ Failed ({resp.status_code}): {resp.json()}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Candidates based on user's list and OpenRouter conventions
candidates = [
    "qwen/qwen3.6-plus:free",
    "qwen/qwen-3.6-plus:free",
    "qwen/qwen-turbo",
    "qwen/qwen-plus",
    "google/gemini-2.0-flash-001" 
]

for c in candidates:
    if test_model_id(c):
        print(f"\n✨ WINNER: {c}")
        break
