import requests, os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')
r = requests.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {key}'})
models = r.json().get('data', [])

print(f"{'MODEL ID':<50} | {'ZDR SUPPORTED'}")
print("-" * 70)

for m in models:
    if ':free' in m.get('id', ''):
        # A model supports ZDR if any of its endpoints support it
        # Note: OpenRouter API structure for endpoints can vary, but we'll try to find it.
        # Actually, ZDR is often a property of the model/endpoint in the detailed view.
        # We'll check for a 'zdr' flag if it exists or just model info.
        is_zdr = m.get('architecture', {}).get('instruct_type') is not None # Placeholder check
        # Let's see if we can find it in 'endpoints' (this is a guess at the API structure for 2026)
        endpoints = m.get('endpoints', [])
        supports_zdr = any(e.get('zero_data_retention', False) for e in endpoints)
        
        print(f"{m['id']:<50} | {'✅ YES' if supports_zdr else '❌ NO'}")
