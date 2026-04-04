"""
diagnose_tokens.py — Verify what OpenRouter actually returns for token usage.

Run with:
    venv/Scripts/python diagnose_tokens.py
"""
import os, json
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.0-flash-001"

headers = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Token Diagnostic",
}

# ══════════════════════════════════════════════════════════════
#  TEST 1: Non-streaming (should reliably return usage)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: NON-STREAMING call")
print("=" * 60)

llm_sync = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
    streaming=False,
    max_tokens=50,
    default_headers=headers,
)

result = llm_sync.invoke("Say hello in one word.")
print(f"\nAnswer: {result.content}")
print(f"\nresponse_metadata keys: {list(result.response_metadata.keys())}")
print(f"\nFull response_metadata:")
print(json.dumps(result.response_metadata, indent=2, default=str))

# ══════════════════════════════════════════════════════════════
#  TEST 2: Streaming WITH stream_options (your current setup)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: STREAMING with stream_options={'include_usage': True}")
print("=" * 60)

llm_stream = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
    streaming=True,
    max_tokens=50,
    default_headers=headers,
    model_kwargs={"stream_options": {"include_usage": True}},
)

last_chunk = None
chunk_count = 0
for chunk in llm_stream.stream("Say hello in one word."):
    chunk_count += 1
    last_chunk = chunk

print(f"\nTotal chunks received: {chunk_count}")
if last_chunk:
    print(f"\nLast chunk content: '{last_chunk.content}'")
    print(f"Last chunk has response_metadata: {bool(last_chunk.response_metadata)}")
    if last_chunk.response_metadata:
        print(f"response_metadata keys: {list(last_chunk.response_metadata.keys())}")
        print(f"\nFull response_metadata:")
        print(json.dumps(last_chunk.response_metadata, indent=2, default=str))
    else:
        print(">>> NO response_metadata on last chunk!")

# ══════════════════════════════════════════════════════════════
#  TEST 3: Streaming WITHOUT stream_options
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: STREAMING without stream_options")
print("=" * 60)

llm_plain = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
    streaming=True,
    max_tokens=50,
    default_headers=headers,
)

last_chunk2 = None
chunk_count2 = 0
for chunk in llm_plain.stream("Say hello in one word."):
    chunk_count2 += 1
    last_chunk2 = chunk

print(f"\nTotal chunks received: {chunk_count2}")
if last_chunk2:
    print(f"\nLast chunk content: '{last_chunk2.content}'")
    print(f"Last chunk has response_metadata: {bool(last_chunk2.response_metadata)}")
    if last_chunk2.response_metadata:
        print(f"response_metadata keys: {list(last_chunk2.response_metadata.keys())}")
        print(f"\nFull response_metadata:")
        print(json.dumps(last_chunk2.response_metadata, indent=2, default=str))
    else:
        print(">>> NO response_metadata on last chunk!")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
