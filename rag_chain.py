"""
rag_chain.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  • OpenRouter LLM configuration (free model by default)
  • ChromaDB retriever setup
  • LangChain retrieval chain construction
"""

from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
    ENABLE_HYBRID_SEARCH,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

import os
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from backend import load_bm25_index

# 🚀 Elite Patterns: LLM Intent Routing
# We use a fast, local model (1B-3B parameters) for orchestration.
AGENT_ROUTER_MODEL = "llama3.2:1b"

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


def hybrid_search(db: Chroma, query: str, collection_name: str = "default", k: int = 10, exclude_file: str | None = None):
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion.
    """
    if not ENABLE_HYBRID_SEARCH:
        # Fallback to vector search
        retriever = get_retriever(db, k=k, exclude_file=exclude_file)
        return retriever.invoke(query)

    # 1. 🔍 Vector Search
    vector_retriever = get_retriever(db, k=k*2, exclude_file=exclude_file)
    vector_docs = vector_retriever.invoke(query)
    
    # 2. 🔍 BM25 Keyword Search
    bm25_data = load_bm25_index(collection_name)
    bm25_docs = []
    if bm25_data:
        bm25_model = bm25_data["bm25"]
        all_docs = bm25_data["docs"]
        
        tokenized_query = word_tokenize(query.lower())
        top_n = bm25_model.get_top_n(tokenized_query, all_docs, n=k*2)
        
        # Apply filter manual since BM25 model doesn't know about metadata filters
        if exclude_file:
            bm25_docs = [d for d in top_n if d.metadata.get("source") != exclude_file]
        else:
            bm25_docs = top_n

    # 3. 🧪 Reciprocal Rank Fusion (RRF)
    # RRF Score(d) = sum(1 / (k + rank))
    RRF_K = 60
    scores = {} # {doc_content: score}
    doc_map = {} # {doc_content: doc_object}
    
    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            content = doc.page_content
            score = 1.0 / (RRF_K + rank + 1) * weight
            scores[content] = scores.get(content, 0) + score
            doc_map[content] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    
    # Sort by RRF score
    sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    final_docs = [doc_map[c] for c in sorted_contents[:k]]
    
    return final_docs


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


def hybrid_search(
    db: Chroma, 
    query: str, 
    collection_name: str = "default", 
    k: int = 10, 
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None
) -> list[Document]:
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF).
    """
    from config import ENABLE_HYBRID_SEARCH, BM25_WEIGHT, VECTOR_WEIGHT
    
    if not ENABLE_HYBRID_SEARCH:
        # Fallback to standard vector search
        search_kwargs = {"k": k, "fetch_k": k*5}
        if exclude_file:
            search_kwargs["filter"] = {"source": {"$ne": exclude_file}}
        return db.similarity_search(query, **search_kwargs)

    # 1. 🔍 Vector Search (Semantic)
    # We fetch a larger candidate pool for RRF to merge
    vector_docs = db.similarity_search(query, k=k*3)
    
    # 2. 🔍 BM25 Keyword Search
    bm25_data = load_bm25_index(collection_name)
    bm25_docs = []
    if bm25_data:
        bm25_model = bm25_data["bm25"]
        all_docs = bm25_data["docs"]
        
        tokenized_query = word_tokenize(query.lower())
        top_n = bm25_model.get_top_n(tokenized_query, all_docs, n=k*3)
        bm25_docs = top_n

    # 3. 🧪 Reciprocal Rank Fusion (RRF)
    # RRF Score(d) = sum(1 / (k + rank))
    RRF_K = 60
    scores = {} # {doc_content: score}
    doc_map = {} # {doc_content: doc_object}
    
    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            if exclude_file and doc.metadata.get("source") == exclude_file:
                continue
            if filter_extensions and doc.metadata.get("file_extension") not in filter_extensions:
                continue
                
            content = doc.page_content
            score = (1.0 / (RRF_K + rank + 1)) * weight
            scores[content] = scores.get(content, 0) + score
            doc_map[content] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    
    # Sort by merged RRF score
    sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[c] for c in sorted_contents[:k]]


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


# 🚀 Local LLM Orchestrator (Agentic Router)
# ═══════════════════════════════════════════════════════════════════════════

class AgenticRouter:
    """
    Decoupled decision engine using a fast local LLM (llama3.2:1b)
    to handle classification, rewriting, and summarization tasks.
    """
    def __init__(self, model_name: str = AGENT_ROUTER_MODEL):
        try:
            self.llm = get_llm(model=f"{OLLAMA_PREFIX}{model_name}", temperature=0.0, streaming=False)
        except Exception:
            self.llm = None

    def classify_intent(self, query: str, history: list[BaseMessage]) -> str:
        """Classify as NEW topic or FOLLOW-UP with context reuse."""
        if not self.llm or not history:
            return "NEW" # conservative fallback
            
        context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}" for m in history[-3:]])
        prompt = (
            f"Given the chat history:\n{context}\n\n"
            f"And the new user prompt: '{query}'\n\n"
            "Is the new prompt a follow-up to the current topic or a completely NEW topic?\n"
            "Reply ONLY with 'FOLLOW-UP' or 'NEW'."
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content.upper()
            return "FOLLOW-UP" if "FOLLOW-UP" in content else "NEW"
        except Exception:
            return "NEW"

    def rewrite_query(self, query: str, history: list[BaseMessage]) -> str:
        """Rewrite a vague follow-up into a standalone search query."""
        if not self.llm or not history:
            return query
            
        context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:300]}" for m in history[-2:]])
        prompt = (
            f"Chat History:\n{context}\n\n"
            f"User just said: '{query}'\n\n"
            "Rewrite this into a standalone search query that includes all necessary context. "
            "Reply ONLY with the rewritten query."
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip().strip('"')
        except Exception:
            return query

    def summarize_state(self, history: list[BaseMessage]) -> str:
        """Create a dense 3-bullet summary of the conversation state."""
        if not self.llm:
            return "None."
            
        context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:500]}" for m in history[-6:]])
        prompt = (
            f"History:\n{context}\n\n"
            "Summarize the conversation state so far in exactly 3 dense bullet points. "
            "Focus on technical topics discussed. Reply ONLY with the bullet points."
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "Error generating summary."


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

    # 🚀 Professional Polish: Instantiate Agentic Router
    router = AgenticRouter()

    def _full_context_cache_chain(inputs: dict):
        """
        Unified chain with Agentic Routing, Hybrid Search,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")

        # 1. 🤖 Parallel Agentic Orchestration ──────────────────────────────
        # We run classification and summarization (if needed) concurrently
        # to reduce pre-flight latency.
        existing_sentinel = inputs.get("sentinel_state", "")
        turn_count = sum(1 for m in history if isinstance(m, HumanMessage))
        should_summarize = (turn_count > 0 and turn_count % SENTINEL_INTERVAL == 0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Always classify intent
            intent_future = executor.submit(router.classify_intent, user_input, history)
            
            # Conditionally summarize state
            summary_future = None
            if should_summarize:
                summary_future = executor.submit(router.summarize_state, history)
            
            # Await results
            intent = intent_future.result()
            if summary_future:
                inputs["sentinel_state"] = summary_future.result()
            else:
                inputs["sentinel_state"] = existing_sentinel
        
        # 3. Pinned context passthrough
        inputs["full_source_context"] = pinned_content if (pinned_content and pinned_content != "None pinned.") else "None pinned."

        # 4. Retrieval path
        cached_input = inputs.get("cached_docs")
        
        stable_docs = []
        new_docs = []
        
        # Use existing cache if intent is FOLLOW-UP and cache exists
        if intent == "FOLLOW-UP" and cached_input:
            if isinstance(cached_input, dict):
                stable_docs = cached_input.get("stable", [])
                new_docs = cached_input.get("new", [])
            else:
                stable_docs = cached_input
        else:
            # Fresh retrieval: Use Agentic Rewriting for precise search signal
            if db:
                search_signal = router.rewrite_query(user_input, history) if intent == "FOLLOW-UP" else user_input
                pinned_file = inputs.get("exclude_file")
                ext_filter = inputs.get("filter_extensions")
                
                # Use hybrid search (BM25 + vector) for better precision
                new_docs = hybrid_search(
                    db, search_signal,
                    collection_name=coll_name,
                    k=RETRIEVER_K,
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
        
        yield {"context": inputs["context"], "intent": intent}
        for chunk in question_answer_chain.stream(inputs):
            # Extract content string while preserving the raw chunk for metadata (caching)
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield {"answer": content, "raw_chunk": chunk}

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
