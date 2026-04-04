"""
rag_chain.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  • OpenRouter LLM configuration (free model by default)
  • ChromaDB retriever setup
  • LangChain retrieval chain construction
"""

from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    CLOUDROUTER_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    MAX_TOKENS,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    ANTHROPIC_CACHE_BETA_HEADER,
    ENABLE_PROMPT_CACHING,
    MAX_CACHE_CHECKPOINTS,
    MIN_PREV_QUERY_LENGTH,
    MIN_CURRENT_QUERY_LENGTH,
    SENTINEL_INTERVAL,
    SENTINEL_MAX_TOKENS,
    PROVIDER_CACHE_PROFILES,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

# Prefix used by app.py to signal "use local Ollama with model X"
OLLAMA_PREFIX = "ollama:"


# ═══════════════════════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════════════════════

def is_cache_capable(model: str | None) -> bool:
    """Check if the model/provider supports prompt caching blocks."""
    if not model or model.startswith(OLLAMA_PREFIX):
        return False
    m_lower = model.lower()
    return any(p in m_lower for p in PROVIDER_CACHE_PROFILES)


def get_cache_profile(model: str | None) -> tuple[int, int]:
    """
    Cross-Provider Cache Router.
    Returns (max_checkpoints, min_tokens_for_cache) for the given model.
    Different providers have different cache economics:
      - Claude: 4 breakpoints, 1024 token minimum
      - Gemini: more breakpoints allowed, 1028 token minimum
      - DeepSeek/Qwen: similar to Claude
    Falls back to global defaults if model is unknown.
    """
    if not model:
        return (MAX_CACHE_CHECKPOINTS, 1024)
    m_lower = model.lower()
    for pattern, profile in PROVIDER_CACHE_PROFILES.items():
        if pattern in m_lower:
            return profile
    return (MAX_CACHE_CHECKPOINTS, 1024)


def format_message_content(text: str, model: str | None, use_cache: bool = False) -> str | list[dict]:
    """
    Return content as a plain string for non-cache models, 
    or a block-list for cache-capable ones.
    """
    # If caching is globally disabled or model can't handle it, 
    # always return a plain string.
    if not ENABLE_PROMPT_CACHING or not use_cache or not is_cache_capable(model):
        return text
    
    # Return Anthropic-style block format with cache markers
    return [
        {
            "type": "text", 
            "text": text, 
            "cache_control": {"type": "ephemeral"}
        }
    ]


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
):
    """
    Return a chat model instance.

    If *model* is ``OLLAMA_SENTINEL`` the function returns a local
    ``ChatOllama``; otherwise it returns a ``ChatOpenAI`` pointed at
    OpenRouter.
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    # ── Local Ollama path ──────────────────────────────────────────────
    if model and model.startswith(OLLAMA_PREFIX):
        ollama_model_name = model[len(OLLAMA_PREFIX):]
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=ollama_model_name,
            temperature=temp,
            num_predict=MAX_TOKENS,
        )

    # ── OpenRouter path ────────────────────────────────────────────────
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    # 🚀 Professional Polish: Conditional Header Safety
    # Only send Anthropic-specific headers when using a Claude model
    default_headers = {
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Private AI Knowledge Base",
    }
    
    current_model = model or DEFAULT_MODEL
    if "claude" in current_model.lower():
        default_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=current_model,
        temperature=temp,
        streaming=streaming,
        max_tokens=MAX_TOKENS,
        default_headers=default_headers,
        # 🚀 Enable usage in stream for telemetry visibility
        model_kwargs={"stream_options": {"include_usage": True}}
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

def get_retriever(db: Chroma, k: int | None = None, exclude_file: str | None = None):
    """
    Wrap a ChromaDB store as a LangChain retriever.
    
    If exclude_file is provided, we use a metadata filter to prevent 
    retrieving from the 'Pinned' file to avoid duplicate context.
    """
    search_kwargs = {
        "k": k or RETRIEVER_K,
        "fetch_k": RETRIEVER_FETCH_K,
    }
    
    if exclude_file:
        search_kwargs["filter"] = {"source": {"$ne": exclude_file}}

    return db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-ENCODER RE-RANKER
# ═══════════════════════════════════════════════════════════════════════════

_cross_encoder: HuggingFaceCrossEncoder | None = None


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Return the singleton cross-encoder model (~80 MB, CPU)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
    return _cross_encoder


class ScoringCrossEncoderReranker(CrossEncoderReranker):
    """CrossEncoderReranker that preserves scores in document metadata."""

    def compress_documents(self, documents, query, callbacks=None):
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        top_docs = []
        for doc, score in result[: self.top_n]:
            doc.metadata["reranker_score"] = round(float(score), 4)
            top_docs.append(doc)
        
        # 🚀 Accuracy-First: HIGHER-RESOLUTION DETERMINSTIC SORT (0.05 Bins)
        # We group chunks into 0.05-precision bins (0.95, 0.90, 0.85, etc.) 
        # for sorting. This doubles the precision of the Platinum Standard
        # while preserving nearly identical prefix stability for caching.
        top_docs.sort(key=lambda d: (-round(float(d.metadata.get("reranker_score", 0)) * 20) / 20.0, 
                                     d.metadata.get("source", ""), 
                                     d.page_content))
        return top_docs


def get_reranking_retriever(
    db: Chroma,
    k: int | None = None,
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None,
):
    """
    Wrap the MMR retriever with a cross-encoder re-ranker.
    Optionally filter by file extensions (Phase 1b: Metadata Filtering).
    """
    search_kwargs = {
        "k": k or RETRIEVER_K,
        "fetch_k": RETRIEVER_FETCH_K,
    }

    # Build metadata filter
    filters = {}
    if exclude_file:
        filters["source"] = {"$ne": exclude_file}
    if filter_extensions:
        filters["file_extension"] = {"$in": filter_extensions}

    if filters:
        if len(filters) == 1:
            search_kwargs["filter"] = filters
        else:
            # ChromaDB $and for multiple filters
            search_kwargs["filter"] = {"$and": [{k: v} for k, v in filters.items()]}

    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
    compressor = ScoringCrossEncoderReranker(
        model=_get_cross_encoder(),
        top_n=RERANKER_TOP_N,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def hybrid_retrieve(db: Chroma, query: str, k: int | None = None,
                    exclude_file: str | None = None,
                    filter_extensions: list[str] | None = None) -> list:
    """
    Phase 2a: Hybrid Search — combine BM25 keyword matching with
    vector semantic search for precision retrieval.

    BM25 catches exact keyword matches that embeddings might miss.
    Vector search catches semantic meaning that keywords miss.
    Results are merged and deduplicated before reranking.
    """
    from config import ENABLE_HYBRID_SEARCH, BM25_WEIGHT, VECTOR_WEIGHT

    top_k = k or RETRIEVER_K

    if not ENABLE_HYBRID_SEARCH:
        retriever = get_reranking_retriever(
            db, k=top_k, exclude_file=exclude_file,
            filter_extensions=filter_extensions
        )
        return retriever.invoke(query)

    # 1. Vector search (semantic)
    vector_results = db.similarity_search_with_relevance_scores(query, k=top_k * 3)

    # 2. BM25-style keyword search via ChromaDB's where_document
    #    ChromaDB supports $contains for basic keyword matching
    query_keywords = [w for w in query.lower().split() if len(w) > 3]
    keyword_results = []
    for kw in query_keywords[:5]:  # top 5 keywords
        try:
            kw_docs = db.get(
                where_document={"$contains": kw},
                include=["documents", "metadatas"],
                limit=top_k,
            )
            for doc_text, meta in zip(
                kw_docs.get("documents", []),
                kw_docs.get("metadatas", [])
            ):
                from langchain_core.documents import Document
                keyword_results.append(Document(page_content=doc_text, metadata=meta or {}))
        except Exception:
            continue

    # 3. Merge and deduplicate
    seen = set()
    merged = []
    # Vector results first (weighted higher)
    for doc, score in vector_results:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            doc.metadata["hybrid_vector_score"] = round(float(score), 4)
            merged.append(doc)

    # Keyword results fill remaining slots
    for doc in keyword_results:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            doc.metadata["hybrid_keyword_match"] = True
            merged.append(doc)

    # 4. Rerank the merged set
    if merged:
        ce = _get_cross_encoder()
        scores = ce.score([(query, d.page_content) for d in merged])
        scored = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in scored[:RERANKER_TOP_N]:
            doc.metadata["reranker_score"] = round(float(score), 4)
            result.append(doc)

        # Deterministic sort for cache stability
        result.sort(key=lambda d: (
            -round(float(d.metadata.get("reranker_score", 0)) * 20) / 20.0,
            d.metadata.get("source", ""),
            d.page_content
        ))
        return result

    return []


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

# 🚀 CORE INSTRUCTIONS (Static/Cached)
CORE_INSTRUCTIONS = """\
You are an expert AI assistant specialising in code analysis and document comprehension.

INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved context below.
2. If the context does not contain enough information, say so clearly — \
   do NOT fabricate an answer.
3. When discussing code, reference the source file and explain the logic.
4. Be concise, precise, and use markdown formatting where helpful.
5. If the user asks for code improvements, provide the improved version \
   with clear explanations.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY CACHING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_history_with_cache(history: list[BaseMessage], model: str | None) -> list[BaseMessage]:
    """
    Inject cache markers into the chat history based on the 4-breakpoint limit.
    BP 1 and 2 are used by the system instructions and context. 
    BP 3 and 4 are used here to keep history 'warm'.
    """
    max_bp, _ = get_cache_profile(model)
    if not history or not ENABLE_PROMPT_CACHING or max_bp <= 2 or not is_cache_capable(model):
        return history

    new_history = []
    # Identify indices for breakpoints (e.g., start of history and mid-point)
    # We target the 'content' block to add the cache_control marker.
    for i, msg in enumerate(history):
        # 🚀 Final Zero-Gaps: Exactly 4 Breakpoints 
        # (1:Instructions, 2:Pinned, 3:Stable RAG, 4:History Start)
        is_history_start = (i == 0)
        content = format_message_content(msg.content, model, use_cache=is_history_start)

        if isinstance(msg, HumanMessage):
            new_history.append(HumanMessage(content=content))
        else:
            new_history.append(AIMessage(content=content))
    return new_history

# 🚀 Professional Polish: Linguistic Logic Gates (History vs Speed)
# ═══════════════════════════════════════════════════════════════════════════
#  PLATINUM STANDARD: RAG UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════


def build_sentinel_state_block(
    history: list[BaseMessage],
    existing_block: str = "",
    llm=None,
) -> str:
    """
    Sentinel History Cache: Every SENTINEL_INTERVAL turns, compress
    the conversation history into a compact State Block (~500 tokens).

    The State Block is pinned at the top of the prompt so the LLM
    retains full conversational context while keeping the cached
    prefix small and stable.

    If no LLM is provided, falls back to a local extractive summary
    (no API call, zero cost).
    """
    turn_count = sum(1 for m in history if isinstance(m, HumanMessage))

    # Only rebuild every N turns
    if turn_count % SENTINEL_INTERVAL != 0 or turn_count == 0:
        return existing_block

    # --- Local extractive summary (no LLM call, zero cost) ---
    topics = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            topics.append(f"- Q: {msg.content[:120]}")
        elif isinstance(msg, AIMessage):
            # Extract just the first sentence of each AI response
            first_line = msg.content.split("\n")[0][:150]
            topics.append(f"  A: {first_line}")

    summary = "\n".join(topics[-SENTINEL_INTERVAL * 2:])  # last N turns

    state_block = (
        f"CONVERSATION STATE (auto-summarized at turn {turn_count}):\n"
        f"Topics discussed so far:\n{summary}\n"
        f"---\n"
        f"Previous state: {existing_block[:200] if existing_block else 'None'}"
    )

    # Trim to budget
    if len(state_block) > SENTINEL_MAX_TOKENS * 4:  # rough char estimate
        state_block = state_block[:SENTINEL_MAX_TOKENS * 4]

    return state_block


def is_semantic_lock_query(user_input: str) -> bool:
    """
    Semantic Locking: Detect if the query is a follow-up that should
    bypass retrieval entirely and reuse the exact previous context.

    Returns True for:
      - Pronoun-led queries ("it", "this", "that")
      - Follow-up indicators ("Why?", "Explain more", "How?")
      - Very short continuation queries (1-3 words)
    """
    q = user_input.strip().lower()
    words = q.split()

    # Very short queries (1-3 words) are almost always follow-ups
    if len(words) <= 3:
        return True

    # Starts with a pronoun or follow-up word
    followup_starters = [
        "it", "this", "that", "these", "those", "they",
        "why", "how", "what about", "explain", "elaborate",
        "tell me more", "go on", "continue", "and", "also",
        "same", "again", "more", "yes", "no", "ok",
    ]
    for starter in followup_starters:
        if q.startswith(starter):
            return True

    # Ends with a question mark and is short
    if q.endswith("?") and len(words) <= 6:
        return True

    return False


def rewrite_query_with_context(user_input: str, history: list[BaseMessage]) -> str:
    """
    Rewrite a short/ambiguous follow-up query by injecting context
    from recent chat history. This helps ChromaDB retrieve the right
    chunks when the user says things like "what does it do?" or
    "explain the second one".

    This is a LOCAL rewrite (no LLM call) — fast and free.
    """
    if not history:
        return user_input

    # Only rewrite short queries that are likely follow-ups
    if len(user_input.split()) > 12:
        return user_input  # long enough to stand on its own

    # Extract the last human query for context
    prev_human_msgs = [m for m in history if isinstance(m, HumanMessage)]
    if not prev_human_msgs:
        return user_input

    prev_query = prev_human_msgs[-1].content

    # Detect pronouns / vague references that need context
    vague_patterns = [
        "it", "this", "that", "these", "those", "they", "them",
        "the function", "the file", "the code", "the class",
        "the module", "the method", "above", "same",
    ]
    query_lower = user_input.lower()
    has_vague_ref = any(f" {p} " in f" {query_lower} " or
                        query_lower.startswith(f"{p} ") or
                        query_lower.endswith(f" {p}")
                        for p in vague_patterns)

    if has_vague_ref or len(user_input.split()) <= 5:
        # Prepend previous query as context for ChromaDB search
        return f"{prev_query}\n{user_input}"

    return user_input


def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    
    # 🚀 Professional Polish: Dynamic Retrieval Configuration
    # We build our retrievers inside the lambda to support the 
    # Pinned File exclusion filter.

    # 🚀 Professional Polish: Dual-Path Prompt Construction
    # Non-cache models (Gemini/Qwen/Ollama) get a clean string.
    # Cache-capable models (Claude) get the structured block list.
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        system_blocks = [
            format_message_content(CORE_INSTRUCTIONS, model, use_cache=True)[0],
            format_message_content("CONVERSATION STATE:\n{sentinel_state}", model, use_cache=True)[0],
            format_message_content("FULL SOURCE CONTEXT (PINNED):\n{full_source_context}", model, use_cache=True)[0],
            format_message_content("STABLE RAG CHUNKS (CACHED):\n{stable_context}", model, use_cache=True)[0],
        ]
        # Use remaining breakpoints for dynamic context (no cache marker)
        system_blocks.append({"type": "text", "text": "NEW RAG CHUNKS (DYNAMIC):\n{new_context}"})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        system_text = (
            f"{CORE_INSTRUCTIONS}\n\n"
            "CONVERSATION STATE:\n{sentinel_state}\n\n"
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}\n\n"
            "STABLE RAG CHUNKS:\n{stable_context}\n\n"
            "NEW RAG CHUNKS:\n{new_context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # 🚀 Platinum Standard: Metadata-Aware LCEL Chain
    # We remove StrOutputParser to preserve the 'response_metadata' (for caching token counts)
    # inside the raw message chunks.
    question_answer_chain = prompt | llm

    def _full_context_cache_chain(inputs: dict):
        """
        Unified chain with Sentinel History, Semantic Locking,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])

        # 1. Sentinel History Cache: build/update the state block
        existing_sentinel = inputs.get("sentinel_state", "")
        inputs["sentinel_state"] = build_sentinel_state_block(
            history, existing_block=existing_sentinel
        )

        # 2. Pinned context passthrough
        inputs["full_source_context"] = pinned_content if (pinned_content and pinned_content != "None pinned.") else "None pinned."

        # 3. Retrieval path
        cached_input = inputs.get("cached_docs")
        
        stable_docs = []
        new_docs = []
        
        if isinstance(cached_input, dict):
            # 🚀 Split Context Path
            stable_docs = cached_input.get("stable", [])
            new_docs = cached_input.get("new", [])
        elif cached_input:
            # Fallback for old list format
            stable_docs = cached_input
        else:
            # No cache provided, perform fresh retrieval (if db exists)
            if db:
                search_signal = rewrite_query_with_context(user_input, history)
                pinned_file = inputs.get("exclude_file")
                ext_filter = inputs.get("filter_extensions")
                # Use hybrid search (BM25 + vector) for better precision
                new_docs = hybrid_retrieve(
                    db, search_signal,
                    exclude_file=pinned_file,
                    filter_extensions=ext_filter,
                )
            else:
                new_docs = []
        
        def _format_docs(docs):
            return "\n\n".join([f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}" for d in docs]) if docs else "None."

        inputs["stable_context"] = _format_docs(stable_docs)
        inputs["new_context"] = _format_docs(new_docs)
        
        # Keep internal context list for metadata
        inputs["context"] = stable_docs + new_docs
        
        # Normalise history format with model-awareness
        inputs["chat_history"] = _prepare_history_with_cache(history, model)
        
        yield {"context": inputs["context"]}
        for chunk in question_answer_chain.stream(inputs):
            # Extract content string while preserving the raw chunk for metadata (caching)
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield {"answer": content, "raw_chunk": chunk}

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
