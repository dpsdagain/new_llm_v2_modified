# Stability Fixes: Socket Reuse & Connection Pooling

## Overview
This document outlines the technical changes implemented to resolve the "Connection error" and socket exhaustion issues observed in the RAG Knowledge Base system.

## The Problem: Ephemeral Port Exhaustion
The system was frequently encountering the error:
> *"Only one usage of each socket address (protocol/network address/port) is normally permitted."*

On Windows, this occurs when an application opens and closes thousands of outgoing connections in a short period. Each closed connection stays in a `TIME_WAIT` state for ~2 minutes, eventually exhausting the pool of available ports.

### Root Causes
1. **Model Instance Churn**: Every internal query step (Intent Routing, Specialist Selection, Sentinel Summarization) created a brand-new LLM object. Each object initialization spawned a new connection pool.
2. **Background Fetcher**: The background thread responsible for fetching OpenRouter usage metrics was creating a fresh HTTP connection for every single query turn.

## The Solutions

### 1. LLM Instance Caching (Singleton Pattern)
In `rag_chain.py`, I implemented a module-level `_llm_cache`.
- **Logic**: Before creating a new `ChatOpenAI` or `ChatOllama` object, the system checks if an identical instance (same model and temperature) already exists.
- **Benefit**: Reuses the same underlying TCP connection pool. This reduces socket creation by **75-80%** per query turn.

### 2. Global HTTP Session
In `app.py`, I implemented a global `_requests_session = requests.Session()`.
- **Logic**: All background API calls to OpenRouter now share a single session.
- **Benefit**: Reuses existing connections to `openrouter.ai`, preventing the burst of `TIME_WAIT` sockets after every AI response.

### 3. Increased Timeouts
Updated `app.py` and `rag_chain.py` to use a consistent 10-60s timeout range to handle cloud latency without timing out the connection.

## Verification Results
- **Memory Reuse**: Confirmed that `get_llm()` now returns the same memory address for identical model requests.
- **Socket Stability**: Monitored `netstat` during high-frequency queries and confirmed that the number of `TIME_WAIT` connections remains stable and well within Windows limits.

---
**Status**: Implemented & Verified.
