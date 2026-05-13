# End-to-End Query Test Report (T-E2E)
**Date:** 2026-04-11 14:35:30
**Model:** `google/gemma-4-31b-it:free`
**Collection:** `e2e_test_collection_final`
**Total Runtime:** 50.0s

## Summary

| Metric | Count |
|--------|-------|
| ✅ Passed | 1 |
| ❌ Failed | 1 |
| 💥 Errors | 3 |
| ⏭️ Skipped | 0 |
| **Total** | **5** |

## Results Overview

| Test ID | Name | Status | Duration |
|---------|------|--------|----------|
| T-E2E-1 | Single-Turn Query Returns Answer | ❌ FAIL | 3.5s |
| T-E2E-2 | Follow-Up Intent Detected and Context Reused | 💥 ERROR | 5.0s |
| T-E2E-3 | Model Switching Clears Cache State | 💥 ERROR | 3.9s |
| T-E2E-4 | Semantic Cache Hit on Near-Identical Query | ✅ PASS | 0.8s |
| T-E2E-5 | Long Conversation Triggers Sentinel at Turn 3 | 💥 ERROR | 12.1s |

## Detailed Results

### ❌ T-E2E-1 — Single-Turn Query Returns Answer
**Status:** FAIL | **Duration:** 3.5s

| # | Check | Result | Info |
|---|-------|--------|------|
| 1 | Answer non-empty | ❌ | Answer length: 0 chars |
| 2 | Answer > 50 chars | ❌ | Full answer: ... |
| 3 | Answer mentions retrieval/search concepts | ❌ | Keywords found in answer: [] |
| 4 | Intent is NEW | ❌ | Intent: CACHE_HIT |
| 5 | Context docs retrieved | ❌ | Docs retrieved: 0 |

> **Details:** Answer (0 chars): 

---

### 💥 T-E2E-2 — Follow-Up Intent Detected and Context Reused
**Status:** ERROR | **Duration:** 5.0s

> [!CAUTION]
> **Error:** `RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3BkpLpjDTXTtDhAlchyJPaSaRcX'}
Traceback (most recent call last):
  File "F:\Gemini_anti\new_llm_v3\new_llm_v2_modified\test_`

---

### 💥 T-E2E-3 — Model Switching Clears Cache State
**Status:** ERROR | **Duration:** 3.9s

| # | Check | Result | Info |
|---|-------|--------|------|
| 1 | Chains are distinct objects | ✅ | gemma id=2894200684464, qwen id=2894200335696 |

> [!CAUTION]
> **Error:** `RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3BkpLpjDTXTtDhAlchyJPaSaRcX'}
Traceback (most recent call last):
  File "F:\Gemini_anti\new_llm_v3\new_llm_v2_modified\test_`

---

### ✅ T-E2E-4 — Semantic Cache Hit on Near-Identical Query
**Status:** PASS | **Duration:** 0.8s

| # | Check | Result | Info |
|---|-------|--------|------|
| 1 | Intent is CACHE_HIT | ✅ | Intents found: ['CACHE_HIT'] |
| 2 | Cached answer returned | ✅ | Answer length: 281 |
| 3 | Cached answer matches seed | ✅ | Match check: seed_len=281, got_len=281 |

> **Details:** Cache hit with answer: BM25 is a probabilistic ranking function used in information retrieval. It scores documents based on term frequency and inverse document frequency, adjusted for document length normalization. In this 

---

### 💥 T-E2E-5 — Long Conversation Triggers Sentinel at Turn 3
**Status:** ERROR | **Duration:** 12.1s

| # | Check | Result | Info |
|---|-------|--------|------|
| 1 | History tokens (3858) >= SENTINEL_TOKEN_THRESHOLD (1500) | ✅ | Estimated tokens: 3858 |

> [!CAUTION]
> **Error:** `RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Google AI Studio', 'is_byok': False}}, 'user_id': 'user_3BkpLpjDTXTtDhAlchyJPaSaRcX'}
Traceback (most recent call last):
  File "F:\Gemini_anti\new_llm_v3\new_llm_v2_modified\test_`

---

## Configuration Snapshot

| Parameter | Value |
|-----------|-------|
| DEFAULT_MODEL | `google/gemma-4-31b-it:free` |
| SENTINEL_INTERVAL | `3` |
| SENTINEL_TOKEN_THRESHOLD | `1500` |
| RERANK_CANDIDATES | `15` |
| MAX_ZERO_CHUNK_CHARS | `9000` |
| SEMANTIC_CACHE_THRESHOLD | `0.85` |
| ENABLE_HYBRID_SEARCH | `True` |
| USE_RERANKER | `True` |
| MAX_TOKENS | `4096` |
