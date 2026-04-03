import requests, os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')
r = requests.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {key}'})
models = r.json().get('data', [])
free_models = [m for m in models if ':free' in m.get('id', '')]
for m in free_models:
    name = m['id']
    arch = m.get('architecture', {})
    it = arch.get('instruct_type', '?')
    print(f"{name}  |  instruct_type={it}")
