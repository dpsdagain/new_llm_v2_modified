# Level 3 Stress Test Report

**Model:** ollama-cloud:gemma4:31b-cloud
**Date:** 2026-04-13 00:47:44
**Files Ingested:** app.py, backend.py, config.py, rag_chain.py, requirements.txt

## Test 3.1: Aggregation Override

**Prompt:** List every place where SQLiteFTS5BM25 is initialized or queried across all files.

**Answer:**

Based on the provided source code, `SQLiteFTS5BM25` is initialized or queried in the following locations:

### **1. `backend.py`**
This file contains the class definition and several internal methods that initialize it:
*   **`_update_bm25_index(new_docs, collection_name)`**: Initializes the class to perform incremental on-disk indexing:
    ```python
    fts = SQLiteFTS5BM25(collection_name)
    fts.add_documents(new_docs)
    ```
*   **`SQLiteFTS5BM25` Class Methods**: The class itself contains the logic for querying the database via the following methods:
    *   `search(self, query, k=10)`: Performs a general keyword search using the `docs_fts` table.
    *   `search_by_call(self, func_name, k=10)`: Queries the `calls` index for specific function call sites.
    *   `search_by_calls_batch(self, terms, k=10)`: Performs a batch "OR" query on the `calls` index.
    *   `search_by_constant(self, const_name, k=10)`: Queries the `constants` index for specific constants.
    *   `search_by_constants_batch(self, terms, k=10)`: Performs a batch "OR" query on the `constants` index.

### **2. `rag_chain.py`**
This file uses `SQLiteFTS5BM25` to perform hybrid searches and propagation queries:
*   **`hybrid_search` logic**: Initializes the class to retrieve keyword-based documents:
    ```python
    fts = SQLiteFTS5BM25(collection_name)
    bm25_docs = fts.search(query, k=k*3)
    ```
*   **Propagation Query logic**: Initializes the class to handle extra retrieval passes when a propagation query is detected:
    ```python
    if is_propagation_query:
        fts = SQLiteFTS5BM25(coll_name)
        top_anchors = retrieval_anchors[:3]
    ```

**Analysis:** [To be completed after manual review]

---

## Test 3.2: 3-Hop Constant Propagation

**Prompt:** Trace step-by-step how the ENABLE_PROMPT_CACHING flag ultimately affects the system prompt formatting.

**Answer:**

Based on the provided source code, here is the step-by-step trace of how the `ENABLE_PROMPT_CACHING` flag affects system prompt formatting:

### 1. Configuration Definition
The process begins in `config.py`, where the global flag is defined:
```python
ENABLE_PROMPT_CACHING = True
```

### 2. Capability Verification
Before formatting, the system checks if the selected model is actually capable of handling cache markers. In `rag_chain.py`, the function `is_cache_capable(model)` is called. It returns `True` if the model name contains specific brands:
*   **Cache-capable brands:** `"claude"`, `"gemini"`, `"gemma"`, or `"glm-5"`.
*   **Exclusions:** If the model is `None` or starts with the `OLLAMA_PREFIX`, it returns `False`.

### 3. Content Formatting Logic
The actual transformation of the prompt text occurs in the `format_message_content` function in `rag_chain.py`. The logic follows this conditional path:

*   **Condition Check:** The function checks three criteria:
    `if not ENABLE_PROMPT_CACHING or not use_cache or not is_cache_capable(model):`
*   **Path A (Disabled/Unsupported):** If `ENABLE_PROMPT_CACHING` is `False` (or the model is unsupported), the function simply returns the `text` as a **plain string**.
*   **Path B (Enabled/Supported):** If the flag is `True` and the model is capable, the plain string is converted into an **Anthropic-style block list**:
    ```python
    return [
        {
            "type": "text", 
            "text": text, 
            "cache_control": {"type": "ephemeral"}
        }
    ]
    ```

### 4. Application in the RAG Chain
This formatting logic is integrated into the `build_rag_chain` function in `rag_chain.py`. The chain determines the formatting strategy by evaluating:
`is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING`

If `is_cc` is true, the system proceeds to use the block-list format (with `cache_control`) for the prompt construction instead of a clean string prompt.

### Summary of Effect
| `ENABLE_PROMPT_CACHING` | Model Capability | Resulting Format |
| :--- | :--- | :--- |
| `False` | Any | Plain String |
| `True` | Not Capable (e.g., Ollama) | Plain String |
| `True` | Capable (e.g., Gemma 4) | JSON Block with `{"cache_control": {"type": "ephemeral"}}` |

**Analysis:** [To be completed after manual review]

---

## Test 3.3: AST Context Awareness

**Prompt:** Look at the function _extract_called_functions. What class does it belong to, and what file is it in?

**Answer:**

The function `_extract_called_functions` belongs to the **`CodeASTChunker`** class and is located in the file **`F:\Gemini_anti\new_llm_v3\new_llm_v2_modified\backend.py`**.

**Analysis:** [To be completed after manual review]

---

