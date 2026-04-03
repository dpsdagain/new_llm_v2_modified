# Analysis of "Sticky Context" & Prompt Caching Solution

Your current solution introduces several interesting mechanisms to reduce API costs when chatting with documents, specifically:
1. **"Sticky Context" (Hybrid Context Router)**: Skipping retrieval for follow-up queries using simple keyword detection and reusing `st.session_state.last_context`.
2. **Lexicographical Sorting**: Re-ordering chunks by `source` name so context inputs are deterministic.
3. **Prompt Caching API Usage**: Injecting `cache_control` blocks for Anthropic style caching natively supported by OpenRouter models.

Below is an analysis of how well this solves the problem, along with gaps, trade-offs, and suggestions for a more robust architectural approach.

---

## 1. Does it solve the repeated context/token consumption problem?

**Partially, but at the expense of retrieval accuracy.**

*   **When using Native Provider Prompt Caching (e.g., Anthropic/Gemini `cache_control`)**: You *are* successfully saving token processing costs. However, provider-level prompt caching natively solves the redundant token consumption problem *without* needing to manually bypass retrieval. If you send the exact same context to the API, it calculates a hash, hits the cache, and processes it significantly faster/cheaper. Your lexicographical sorting ensures the prompt block is an exact match for the cache, which is a great touch.
*   **When using "Sticky Context" (Skipping Retrieval)**: By forcefully injecting `last_context` without querying ChromaDB via your app.py router, you ensure 100% cache hits. **However, this only works if the follow-up question can be answered by the exact same top-K chunks retrieved earlier.** If the follow-up question touches on a different part of the document, the model will hallucinate or fail because you blocked it from retrieving new information.

## 2. Gaps and Inefficiencies

> [!WARNING]
> Your context routing logic in `app.py` is brittle and prone to high false-positives and false-negatives. 

#### A. Flawed Keyword Detection
```python
CONTEXT_FOLLOWUP_KEYWORDS: list[str] = ["it", "that", "this", "explain", "refactor", "how", "why", "elaborate"]
reuse_context = any(kw in user_input.lower().split() for kw in CONTEXT_FOLLOWUP_KEYWORDS)
```
If a user asks `"How does the database scaling work?"`, the word `how` triggers the system to reuse `last_context`. If `last_context` was entirely about the frontend UI from a previous query, the LLM won't have the context to explain database scaling since the vector DB was skipped.

#### B. Redundant Standalone Query Formation Engine 
In `rag_chain.py`, you use `create_history_aware_retriever(llm, base_retriever, CONTEXTUALIZE_Q_PROMPT)`.
If you *don't* skip retrieval via Sticky Context, this chain sends a pre-flight LLM call *just to rewrite the user's question* based on chat history. This consumes tokens and adds noticeable latency before the actual RAG query even begins.

#### C. Incomplete Deterministic Sorting
```python
top_docs.sort(key=lambda d: d.metadata.get("source", ""))
```
Sorting purely by file `source` (e.g., `"app.py"`) is insufficient for cache determinism. If a document has 20 chunks, and the retriever grabs chunks 2, 5, 8, and 12 from `"app.py"`, sorting by `source` alone leaves them tied. Their final order depends on the stability of Python's sort and how ChromaDB returns them. A slight change in order breaks the Anthropic cache. You need to sort by `source` **and** chunk content or ID.

## 3. Improvements & Better Architectures

### Option A: Let the Cache Do the Heavy Lifting (Semantic Caching + Native Caching)
Instead of forcing "Sticky Context" via brittle keywords, trust the prompt cache.
1. Allow Langchain to formulate the standalone question (e.g., rewriting "Explain it more" into "Explain X more").
2. Pass `"Explain X more"` to ChromaDB. 
3. If the vector similarities return the exact same chunks as before, **Anthropic/Gemini Prompt Caching automatically kicks in** on the backend because the retrieved text block is perfectly identical.
4. If the follow-up question relates to new context, ChromaDB fetches new chunks. The cache might miss, but you avoid a hallucination.

### Option B: BM25/Lexical Rerouting (The "Right" Hybrid Router)
Instead of using static hardcoded keywords to decide whether to skip retrieval:
Check the cosine similarity (or a fast lightweight sentence-transformer) between the *current query* and the *last query*. 
- If `similarity > 0.85`, the user is referring to the same topic $\rightarrow$ reuse `last_context`. 
- If `< 0.85`, it's a new topic $\rightarrow$ query ChromaDB.
You could also use libraries like [semantic-router](https://github.com/aurelio-labs/semantic-router) which are built specifically for zero-latency routing logic.

### Option C: Document-level Caching (Zero Chunking for Mid-Sized Docs)
If your goal is simply "don't re-read the absolute full document", and the document is under ~1M tokens (fitting easily into Gemini/Claude 3.5's context window), **don't chunk it at all.**
Drop the *entire* document into an Anthropic/Gemini model prompt with an `ephemeral` cache tag. 
Provider caching solves the cost problem natively. Passing the full document guarantees 100% accurate recall on follow-up questions because the entire text is safely resting in the provider's RAM for fractions of a cent per query. RAG is better suited for massive multi-document codebases where full context windows are impossible or too slow.

## 4. Architectural Trade-Offs Summary

| Approach | Latency | Token Cost | Response Quality (Accuracy) |
| :--- | :--- | :--- | :--- |
| **Traditional RAG (Always retrieve)** | High (Vector DB search + LLM call) | Medium (Only sends chunks) | High (Always fetches latest context) |
| **Your Sticky Context (Keyword bypass)** | Very Low | Minimal (Guaranteed Prompt Cache Hit) | **Low (Breaks easily on topic transitions)**|
| **Native API Prompt Caching (Full Doc)**| Medium | Low (Cached input is ~10x cheaper) | **Highest (Model has 100% vision limit)** |

## Next Steps for your Codebase
If you want to keep your current architecture but make it more robust, update the reranker's sort key in `rag_chain.py` to:
```python
top_docs.sort(key=lambda d: (d.metadata.get("source", ""), d.page_content))
```
This guarantees absolute determinism for cross-encoder reranked chunks, ensuring your prompt string never invalidates the Anthropic `ephemeral` cache tag due to tied sorts.
