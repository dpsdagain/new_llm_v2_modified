import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OLLAMA_CLOUD_API_KEY")

endpoints = [
    "https://api.ollama.cloud/v1",
    "https://api.ollama.com/v1",
    "https://ollama.com/api",
    "https://api.deepinfra.com/v1/openai",
    "https://api.groq.com/openai/v1",
]

def test_endpoints():
    print(f"Testing with API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")
    
    for url in endpoints:
        print(f"\nTarget: {url}")
        try:
            # Try a simple models list or version check
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(f"{url}/models", headers=headers, timeout=5)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"SUCCESS! Found models.")
                # print(resp.json())
            else:
                print(f"Response: {resp.text[:100]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_endpoints()
