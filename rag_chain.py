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

import logging
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
    MAX_TOKENS,
    ANTHROPIC_CACHE_BETA_HEADER,
    ENABLE_PROMPT_CACHING,
    ENABLE_AUTO_SPECIALIST,
    MAX_CACHE_CHECKPOINTS,
    SEMANTIC_CACHE_THRESHOLD,
    SENTINEL_MAX_TOKENS,
    SENTINEL_TOKEN_THRESHOLD,
    SENTINEL_INTERVAL,
    TRUST_NATIVE_CACHE,
    PROVIDER_CACHE_PROFILES,
    ENABLE_HYBRID_SEARCH,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    USE_RERANKER,
    RERANK_MODEL,
    RERANK_TOP_K,
    RERANK_CANDIDATES,
    PINNED_RELEVANCE_THRESHOLD,
    STICKY_PINNED_CONTEXT,
    SPECIALIST_MAPPING,
)

logger = logging.getLogger(__name__)

# 🚀 Platinum Scaling: Context Window Limits
MAX_CONTEXT_UNION = 7

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from sentence_transformers import CrossEncoder
import hashlib
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from backend import load_bm25_index

# 🚀 Global Executor for non-blocking background tasks (Sentinel summaries)
# Shut down any stale executor left over from a previous Streamlit hot-reload
# before creating a fresh one.  Without this, each reload leaks 2 threads.
import atexit as _atexit

_background_executor = ThreadPoolExecutor(max_workers=2)
_atexit.register(_background_executor.shutdown, wait=False)

def _background_summarize(history: list[BaseMessage]):
    """Background task to update sentinel state without stalling the main stream."""
    try:
        router = VectorRouter()
        new_state = router.summarize_state_fast(history)
        return new_state
    except Exception as e:
        logger.error(f"❌ Background summary failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  EXACT-MATCH QUERY CACHE
# ═══════════════════════════════════════════════════════════════════════════
# Zero-cost layer: if the user sends the *exact* same query string as the
# previous turn, we can skip the embedding call entirely.
# We normalize the query (lowercase, strip punctuation) to catch "near-identical"
# repeats like "Tell me more" vs "Tell me more.".

_last_query_hash: str | None = None
_last_query_embedding_cache: list[float] | None = None


def _normalize_query(query: str) -> str:
    """Normalize query for cache-hit robustness."""
    import re
    return re.sub(r'[^\w\s]', '', query).lower().strip()


def _exact_match_cache_check(query: str) -> list[float] | None:
    """Return the cached embedding if normalized *query* matches the last one."""
    global _last_query_hash
    norm_q = _normalize_query(query)
    q_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
    if q_hash == _last_query_hash and _last_query_embedding_cache is not None:
        return _last_query_embedding_cache
    return None


def _exact_match_cache_store(query: str, embedding: list[float]):
    """Store the query hash and embedding for exact-match reuse."""
    global _last_query_hash, _last_query_embedding_cache
    norm_q = _normalize_query(query)
    _last_query_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
    _last_query_embedding_cache = embedding


# ═══════════════════════════════════════════════════════════════════════════
#  SENTINEL LOCKING
# ═══════════════════════════════════════════════════════════════════════════
_sentinel_in_progress = False

def _background_summarize_locked(history: list[BaseMessage]):
    """Wrapper to ensure only one sentinel task runs at a time."""
    global _sentinel_in_progress
    if _sentinel_in_progress:
        return None
    _sentinel_in_progress = True
    try:
        return _background_summarize(history)
    finally:
        _sentinel_in_progress = False


# ═══════════════════════════════════════════════════════════════════════════
#  PINNED CONTENT EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════
# The pinned file embedding was previously recomputed from
# pinned_content[:2000] on *every single turn*.  Now we cache it
# and only recompute when the pinned content actually changes.

_pinned_content_hash: str | None = None
_pinned_content_embedding: list[float] | None = None


def _get_pinned_embedding(pinned_content: str) -> list[float]:
    """Return a cached embedding for the pinned content prefix."""
    global _pinned_content_hash, _pinned_content_embedding
    # Hash only the prefix we actually embed
    prefix = pinned_content[:2000]
    p_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    if p_hash == _pinned_content_hash and _pinned_content_embedding is not None:
        return _pinned_content_embedding
    from backend import get_embedding_model
    _pinned_content_embedding = get_embedding_model().embed_query(prefix)
    _pinned_content_hash = p_hash
    return _pinned_content_embedding

# 🚀 Elite Patterns: LLM Intent Routing
# We use a fast, local model (1B-3B parameters) for orchestration.
AGENT_ROUTER_MODEL = "llama3.2:1b"

# Prefix used by app.py to signal "use local Ollama with model X"
OLLAMA_PREFIX = "ollama:"


# ═══════════════════════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════════════════════

def is_cache_capable(model: str | None) -> bool:
    """
    Check if the model/provider supports Anthropic-style prompt caching blocks.

    Claude and Gemini 2.0 (via OpenRouter) both support the 
    ``cache_control: {"type": "ephemeral"}`` block format.  
    DeepSeek and Qwen use implicit prefix caching, so they don't 
    need these markers but still benefit from our deterministic sort.
    """
    if not model or model.startswith(OLLAMA_PREFIX):
        return False
    
    m_lower = model.lower()
    # Claude (Anthropic) and Gemini (Google) both support explicit markers on OpenRouter
    return "claude" in m_lower or "gemini" in m_lower


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








def hybrid_search(
    db: Chroma,
    query: str,
    collection_name: str = "default",
    k: int = 10,
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[Document]:
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF).

    If *query_embedding* is provided, it is reused for the vector search
    via ``similarity_search_by_vector``, avoiding a redundant embedding
    inference that ChromaDB would otherwise perform internally.
    """
    if not ENABLE_HYBRID_SEARCH:
        # Fallback to standard vector search.
        # fetch_k is an MMR-only parameter and is not supported by
        # similarity_search / similarity_search_by_vector — omit it.
        chroma_filter = None
        if exclude_file:
            chroma_filter = {"source": {"$ne": exclude_file}}
        if query_embedding:
            return db.similarity_search_by_vector(query_embedding, k=k, filter=chroma_filter)
        return db.similarity_search(query, k=k, filter=chroma_filter)

    # Build a ChromaDB metadata filter from the caller's exclusion criteria so
    # the vector store never fetches docs that will be thrown away post-fetch.
    # BM25 has no filter API — _rank_docs() still handles that side.
    _conditions: list[dict] = []
    if exclude_file:
        _conditions.append({"source": {"$ne": exclude_file}})
    if filter_extensions:
        _conditions.append({"file_extension": {"$in": filter_extensions}})
    if len(_conditions) == 0:
        _chroma_filter = None
    elif len(_conditions) == 1:
        _chroma_filter = _conditions[0]
    else:
        _chroma_filter = {"$and": _conditions}

    # 1. 🔍 Vector Search (Semantic)
    # We fetch a larger candidate pool for RRF to merge.
    # Reuse pre-computed embedding when available to avoid double-embedding.
    if query_embedding:
        vector_docs = db.similarity_search_by_vector(query_embedding, k=k*3, filter=_chroma_filter)
    else:
        vector_docs = db.similarity_search(query, k=k*3, filter=_chroma_filter)
    
    # 2. 🔍 BM25 Keyword Search
    bm25_data = load_bm25_index(collection_name)
    bm25_docs = []
    if bm25_data:
        bm25_model = bm25_data["bm25"]
        all_docs = bm25_data["docs"]
        
        tokenized_query = word_tokenize(query.lower())
        # 🚀 Fix: Calculate all scores first, apply exact metadata filters, then sort.
        # This prevents the bug where the top-k results are all excluded files, resulting in 0 docs.
        scores = bm25_model.get_scores(tokenized_query)
        doc_scores = []
        for score, doc in zip(scores, all_docs):
            if exclude_file and doc.metadata.get("source") == exclude_file:
                continue
            if filter_extensions and doc.metadata.get("file_extension") not in filter_extensions:
                continue
            doc_scores.append((score, doc))
            
        doc_scores.sort(key=lambda x: x[0], reverse=True)
        bm25_docs = [doc for score, doc in doc_scores[:k*3]]

    # 3. 🧪 Reciprocal Rank Fusion (RRF)
    # RRF Score(d) = sum(1 / (k + rank))
    RRF_K = 60
    scores = {} # {doc_id: score}
    doc_map = {} # {doc_id: doc_object}
    
    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            if exclude_file and doc.metadata.get("source") == exclude_file:
                continue
            if filter_extensions and doc.metadata.get("file_extension") not in filter_extensions:
                continue
                
            # 🚀 Platinum Optimization: Use composite key to avoid boilerplate collision
            doc_id = (doc.metadata.get("source"), doc.metadata.get("chunk_index", 0), doc.metadata.get("content_hash", ""))
            score = (1.0 / (RRF_K + rank + 1)) * weight
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    
    # Sort by merged RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    rrf_results = [doc_map[did] for did in sorted_ids[:k]]

    # Cross-encoder re-ranking is handled by LocalReRanker.rerank() in the
    # caller — applying it here as well would score the same docs twice with
    # the same model for zero quality gain.

    return rrf_results


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

# 🚀 CORE INSTRUCTIONS (Static/Cached)
CORE_INSTRUCTIONS = """\
You are an expert AI assistant specialising in code analysis and document comprehension.

INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved context, pinned source, and conversation state provided below.
2. If the context does not contain enough information, say so clearly — \
   do NOT fabricate an answer.
3. When discussing code, reference the source file and explain the logic.
4. Be concise, precise, and use markdown formatting where helpful.
5. If the user asks for code improvements, provide the improved version \
   with clear explanations.
6. The 'CONVERSATION STATE' section contains a summary of our past discussion. \
   You MUST use it to understand follow-up questions and you MUST report its contents if the user asks what it says.
7. CRITICAL OVERRIDE: If the user asks you to retrieve or read the 'CONVERSATION STATE', do NOT explain the python codebase or how variables like {sentinel_state} work. Look physically below at the text under the heading 'CONVERSATION STATE:' and copy it exactly word-for-word. Even if there are no bullet points and it says "No summary generated yet.", you must reply with exactly that text.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY CACHING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_history_with_cache(history: list[BaseMessage], model: str | None) -> list[BaseMessage]:
    """
    Prepare chat history with optional caching.
    
    For models with >4 breakpoints (Gemini), we add a cache marker to 
    the last message in history to checkpoint the conversation.
    For Claude (4 breakpoints), system blocks already consume the limit.
    """
    if not history:
        return history
    
    max_bp, _ = get_cache_profile(model)
    # If we have spare breakpoints (Gemini supports 8+), use one for history.
    # We use 4 for system blocks, so 5+ is the threshold.
    if max_bp > 4 and is_cache_capable(model) and ENABLE_PROMPT_CACHING:
        new_history = list(history)
        last_msg = new_history[-1]
        # Attach cache control to the last message content
        if isinstance(last_msg.content, str):
            last_msg.content = [
                {
                    "type": "text",
                    "text": last_msg.content,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        return new_history
        
    return list(history)


# 🚀 Professional Polish: Linguistic Logic Gates (History vs Speed)
# ═══════════════════════════════════════════════════════════════════════════
#  PLATINUM STANDARD: RAG UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════


# 🚀 Local LLM Orchestrator (Agentic Router)
# ═══════════════════════════════════════════════════════════════════════════

class LocalReRanker:
    """
    Local Cross-Encoder "Critic" that re-scores retrieved chunks 
    to ensure surgical precision before the context is passed to the LLM.
    """
    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        if USE_RERANKER:
            try:
                # This may take a minute to download on first run (~100MB)
                self.model = CrossEncoder(RERANK_MODEL)
            except Exception as e:
                logger.error(f"❌ Re-ranker failed to load: {e}")
                self.model = None

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Re-score and filter documents using the Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        # Prepare pairs for cross-encoding (Query, Chunk)
        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            
            # Combine scores with docs and sort
            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            
            # 🚀 Phase 5: Store the top score for telemetry
            self.last_top_score = float(scored_docs[0][0]) if scored_docs else 0.0
            
            # Log the top score for telemetry
            if scored_docs:
                logger.info(f"🎯 Top Re-rank Relevance Score: {scored_docs[0][0]:.4f}")
            
            return [doc for score, doc in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"❌ Re-ranking execution failed: {e}")
            return documents[:top_k]
# ═══════════════════════════════════════════════════════════════════════════

class VectorRouter:
    """
    Zero-latency decision engine using vector similarity to handle 
    classification and state management without LLM overhead.
    """
    def __init__(self):
        # We reuse the embedding model already loaded in backend.py
        pass

    def classify_intent(self, current_similarity: float) -> str:
        """
        Classify as NEW topic or FOLLOW-UP using embedding similarity.
        Threshold 0.6 is a balanced middle-ground for semantic continuation.
        """
        if current_similarity >= 0.6:
            return "FOLLOW-UP"
        return "NEW"

    def detect_specialty(self, query: str) -> str:
        """
        Detect the best specialist for the query using robust regex word boundaries.
        Returns one of: ['CODE', 'REASONING', 'VISION', 'GENERAL']
        """
        import re
        q = query.lower()
        
        # 💻 Coding Specialist Triggers
        code_triggers = [
            r"code", r"python", r"javascript", r"verilog", r"function", r"class", 
            r"refactor", r"bug", r"debug", r"compile", r"script", r"hdl", r"rtl",
            r"implement", r"write a", r"how to use", r"api", r"library", r"sql", r"html",
            r"cpp", r"c\+\+", r"rust", r"golang"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in code_triggers) or "```" in q:
            return "CODE"
            
        # 🧠 Reasoning / Math Triggers
        reasoning_triggers = [
            r"analyze", r"logic", r"math", r"derive", r"prove", r"step by step",
            r"complex", r"calculate", r"deepseek", r"reason", r"philosophy", 
            r"compare", r"architecture", r"design pattern", r"explain how"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in reasoning_triggers):
            return "REASONING"
            
        # 👁️ Vision Triggers
        vision_triggers = [r"image", r"plot", r"chart", r"diagram", r"vision", r"see this"]
        if any(re.search(rf"\b{t}\b", q) for t in vision_triggers):
            return "VISION"
            
        # Default
        return "GENERAL"

    @staticmethod
    def _extractive_fallback(history: list[BaseMessage]) -> str:
        """
        Pure-Python extractive summary used when the local LLM (Ollama)
        is unavailable.  Keeps the first sentence of each recent human
        message to preserve topic continuity without any external calls.
        """
        bullets = []
        for m in history[-8:]:
            if not isinstance(m, HumanMessage):
                continue
            text = m.content.strip()
            # Take the first sentence (up to first period, question mark, or 120 chars)
            end = len(text)
            for ch in ".?!":
                idx = text.find(ch)
                if 0 < idx < end:
                    end = idx + 1
            snippet = text[:min(end, 120)].strip()
            if snippet:
                bullets.append(f"- {snippet}")
        return "\n".join(bullets[-3:]) if bullets else "No summary available."

    def summarize_state_fast(self, history: list[BaseMessage]) -> str:
        """
        Summarize conversation state.  Tries the local Ollama model first;
        falls back to a pure-Python extractive summary if Ollama is down
        so history compression is never silently skipped.
        """
        try:
            llm = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
            context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:500]}" for m in history[-6:]])
            prompt = (
                f"History:\n{context}\n\n"
                "Summarize the conversation state so far in exactly 3 dense bullet points. "
                "Focus on technical topics discussed. Reply ONLY with the bullet points."
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            logger.warning("Ollama unavailable for sentinel — using extractive fallback")
            return self._extractive_fallback(history)


def _sort_docs_deterministically(
    docs: list[Document],
    stable_hashes: set[str] | None = None,
) -> list[Document]:
    """
    Prefix-Preserving Deterministic Sort.

    When *stable_hashes* is provided (the content hashes of docs that were
    already in the prompt on the previous turn), those docs sort FIRST
    (``_is_new=0``), preserving the exact byte prefix that the provider
    cache (Anthropic/Gemini) already stored.  New docs sort AFTER
    (``_is_new=1``) so they append to the end of the block without
    breaking the cached prefix.

    Within each group the order is fully deterministic:
    ``(source, chunk_index, content_hash, page_content)``.
    """
    def _sort_key(d):
        is_new = 0 if (
            stable_hashes
            and d.metadata.get("content_hash", "") in stable_hashes
        ) else (1 if stable_hashes else 0)
        return (
            is_new,
            str(d.metadata.get("source", "")),
            int(d.metadata.get("chunk_index", 0)),
            str(d.metadata.get("content_hash", "")),
            str(d.page_content),
        )
    return sorted(docs, key=_sort_key)


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def _content_len(c) -> int:
    """Return the character length of a message content field.

    Content may be a plain string or a list of Anthropic cache-control
    dicts — handles both so token estimates stay accurate.
    """
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(b.get("text", "")) for b in c if isinstance(b, dict))
    return 0


def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    
    # 🚀 Professional Polish: Dynamic Retrieval Configuration
    # We build our retrievers inside the lambda to support the 
    # Pinned File exclusion filter.

    # Dual-Path Prompt Construction
    # Only Claude supports Anthropic-style cache_control blocks via OpenRouter.
    # All other models (Gemini/Qwen/DeepSeek/Ollama) get a clean string prompt.
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        # Dynamic Cache Blocks — Claude only
        # Ordered from most stable to most volatile.  We only attach
        # cache_control markers to the first ``max_bp`` blocks; the
        # rest are plain text (no wasted cache writes).
        # Stable order: Instructions > Pinned > Sentinel > RAG context.
        static_system_text = CORE_INSTRUCTIONS
        block_specs = [
            static_system_text,
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}",
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}",
            "CONVERSATION STATE:\n{sentinel_state}"
        ]
        system_blocks = []
        for idx, text in enumerate(block_specs):
            use_cache_marker = idx < max_bp  # only mark up to max_bp blocks
            formatted = format_message_content(text, model, use_cache=use_cache_marker)
            # format_message_content returns a list for cache-capable models
            if isinstance(formatted, list):
                system_blocks.append(formatted[0])
            else:
                # Plain string — wrap in the Anthropic text-block format
                # so the system_blocks list stays homogeneous.
                system_blocks.append({"type": "text", "text": formatted})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        # For non-cache models, we still keep the same logical order
        system_text = (
            f"{CORE_INSTRUCTIONS}\n\n"
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}\n\n"
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}\n\n"
            "CONVERSATION STATE:\n{sentinel_state}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # 🚀 Platinum Standard: Metadata-Aware LCEL Chain
    # We remove StrOutputParser to preserve the 'response_metadata' (for caching token counts)
    # inside the raw message chunks.
    # 🚀 Professional Polish: Instantiate Vector Router & Re-ranker
    router = VectorRouter()
    reranker = LocalReRanker() if USE_RERANKER else None

    question_answer_chain = prompt | llm
    _specialist_llm_cache: dict[str, object] = {}
    # Cooldown tracker: prevents sentinel from re-firing every turn once the
    # token threshold is crossed.  Stored as a mutable dict so the closure
    # can mutate it without a `nonlocal` declaration.
    _sentinel_cooldown: dict[str, int] = {"last_turn": 0}

    def _full_context_cache_chain(inputs: dict):
        """
        Unified chain with Agentic Routing, Hybrid Search,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")
        
        # 🚀 Fix: Get last query and its embedding from inputs
        last_query_emb = inputs.get("last_query_embedding")
        force_retrieval = inputs.get("force_retrieval", False)

        # 1. Context Awareness (Latency-Free)
        turn_count = sum(1 for m in history if isinstance(m, HumanMessage))

        # Token-aware sentinel trigger: summarize when history is large enough.
        # Rough estimate: 1 token ≈ 4 characters.
        estimated_history_tokens = sum(_content_len(m.content) for m in history) // 4
        # Fire when history exceeds the token budget AND at least SENTINEL_INTERVAL
        # turns have passed since the last sentinel run.  Without the cooldown,
        # once the threshold is crossed it fires every single turn.
        should_summarize = (
            turn_count > 0
            and estimated_history_tokens >= SENTINEL_TOKEN_THRESHOLD
            and (turn_count - _sentinel_cooldown["last_turn"]) >= SENTINEL_INTERVAL
        )

        # Calculate semantic similarity to last query.
        current_similarity = 0.0
        current_emb = None
        if last_query_emb and not force_retrieval:
            # Layer 0: Exact-match cache (zero compute for identical queries)
            exact_hit = _exact_match_cache_check(user_input)
            if exact_hit is not None:
                current_emb = exact_hit
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)
                _exact_match_cache_store(user_input, current_emb)
            current_similarity = calculate_cosine_similarity(current_emb, last_query_emb)

        # Semantic Intent Detection (Latency-Free)

        # Semantic hit drives two behaviours:
        # 1. For Claude (cache-capable): retrieval still happens so the
        #    provider cache can fire on the deterministic prefix.
        # 2. For all other models: retrieval is skipped on a semantic hit
        #    since there's no provider-side cache benefit from re-fetching.
        is_semantic_hit = (
            current_similarity >= SEMANTIC_CACHE_THRESHOLD
        )
        
        # 3. Pinned context passthrough with Relevance Gate
        pinned_eligible = False
        if pinned_content and pinned_content != "None pinned.":
            if STICKY_PINNED_CONTEXT:
                pinned_eligible = True
            elif current_emb:
                pinned_emb = _get_pinned_embedding(pinned_content)
                pinned_sim = calculate_cosine_similarity(current_emb, pinned_emb)
                if pinned_sim >= PINNED_RELEVANCE_THRESHOLD:
                    pinned_eligible = True
            else:
                pinned_eligible = True

        inputs["full_source_context"] = pinned_content if pinned_eligible else "None pinned (irrelevant to current query)."

        # 4. Define Previous Context Union
        cached_input = inputs.get("cached_docs")
        previous_union = []
        if cached_input:
            if isinstance(cached_input, dict):
                previous_union = cached_input.get("stable", []) + cached_input.get("new", [])
            else:
                previous_union = cached_input

        # 5. 🤖 Zero-Latency Vector Routing & Specialist Detection ─────────
        # We classify intent based on semantic similarity to avoid LLM latency.
        intent = router.classify_intent(current_similarity) if history else "NEW"
        
        # Phase 4: Specialist Detection
        enable_auto = inputs.get("auto_specialist", ENABLE_AUTO_SPECIALIST)
        specialty = router.detect_specialty(user_input) if enable_auto else "GENERAL"
        specialist_model = SPECIALIST_MAPPING.get(specialty) if enable_auto else None

        pinned_file = inputs.get("exclude_file")
        ext_filter = inputs.get("filter_extensions")

        # Hybrid search (ChromaDB + BM25)
        # When the model supports provider-side prefix caching (Claude/Gemini/DeepSeek),
        # always retrieve so the deterministic sort can maximise cache hits.
        # For all other models the provider cache doesn't help, so skip
        # retrieval on semantic cache hits to save compute.
        k_fetch = RERANK_CANDIDATES if USE_RERANKER else RETRIEVER_K

        # 🚀 Fix: Include DeepSeek/Qwen as cache-capable for prefix stability, 
        # even if they don't use explicit Anthropic-style markers.
        provider_has_cache = is_cache_capable(model) or any(
            p in (model or "").lower() for p in ["deepseek", "qwen", "mistral"]
        )
        
        skip_retrieval = (
            not (TRUST_NATIVE_CACHE and provider_has_cache)
            and is_semantic_hit
            and bool(previous_union)
            and not force_retrieval
        )

        search_query = user_input
        if intent == "FOLLOW-UP":
            try:
                # Use your existing fast router model to rewrite the query
                llm_rewrite = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
                prompt_rewrite = f"Given chat history: {history[-2:]}\nRewrite this query to be standalone: '{user_input}'"
                search_query = llm_rewrite.invoke(prompt_rewrite).content.strip()
            except Exception:
                pass # Fallback to original

        reranker_score = 0.0
        new_retrievals = []
        if not skip_retrieval and db:
            new_retrievals = hybrid_search(
                db, search_query,
                collection_name=coll_name,
                k=k_fetch,
                exclude_file=pinned_file,
                filter_extensions=ext_filter,
                query_embedding=current_emb if search_query == user_input else None,
            )
        elif skip_retrieval:
            # Semantic cache hit — reuse previous docs as the fresh set.
            new_retrievals = list(previous_union)

        # 3. 🎯 Local Re-ranking (Phase 3) ──────────────────────────────
        if USE_RERANKER and reranker and new_retrievals and not skip_retrieval:
            new_retrievals = reranker.rerank(
                user_input,
                new_retrievals,
                top_k=RERANK_TOP_K
            )
            # Capture the top relevance score for telemetry
            reranker_score = getattr(reranker, 'last_top_score', 0.0)

        # Intent-Aware Union Logic with Context Decay
        if intent == "FOLLOW-UP":
            # 🚀 Fix: Prevent "Knowledge Lock-in" by ensuring fresh retrievals 
            # always have priority. We calculate unique new docs first.
            seen_hashes = set()
            unique_new = []
            for d in new_retrievals:
                h = d.metadata.get("content_hash", d.page_content)
                if h not in seen_hashes:
                    unique_new.append(d)
                    seen_hashes.add(h)

            # Cap the new retrievals at MAX_CONTEXT_UNION
            if current_similarity > 0.85:
                # If it's a tight follow-up, limit the new retrievals to 1 or 2, 
                # relying mostly on the surviving_old context.
                unique_new = unique_new[:2] 
            else:
                unique_new = unique_new[:MAX_CONTEXT_UNION]
            
            # Calculate how many slots are left for the older stable docs
            available_old_slots = MAX_CONTEXT_UNION - len(unique_new)
            
            # Keep older docs up to the available slots. We iterate in order, so we keep 
            # the beginning of the old sequence intact, preserving the longest possible 
            # byte prefix for the provider's prompt cache.
            surviving_old = []
            for d in previous_union:
                h = d.metadata.get("content_hash", d.page_content)
                if h not in seen_hashes and len(surviving_old) < available_old_slots:
                    surviving_old.append(d)
                    seen_hashes.add(h)
                    
            final_docs = surviving_old + unique_new
        else:
            final_docs = new_retrievals[:MAX_CONTEXT_UNION]

        # If the model doesn't support caching, filter out massive zero-chunks to save cost
        if not provider_has_cache:
            final_docs = [d for d in final_docs if not (d.metadata.get("zero_chunk") and len(d.page_content) > 10000)]

        # 🚀 Prefix-Preserving Sort: stable docs stay at top, new docs append.
        # This keeps the stable_context byte prefix identical across turns
        # so the provider-side cache (Anthropic/Gemini) gets a hit.
        stable_hashes = {
            d.metadata.get("content_hash", "")
            for d in previous_union
        } if previous_union and intent == "FOLLOW-UP" else None
        sorted_docs = _sort_docs_deterministically(final_docs, stable_hashes=stable_hashes)
        
        def _format_docs(docs):
            return "\n\n".join([f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}" for d in docs]) if docs else "None."

        inputs["stable_context"] = _format_docs(sorted_docs)
        inputs["context"] = sorted_docs
        inputs["chat_history"] = _prepare_history_with_cache(history, model)
        
        # Dynamic Specialist Swap — cached LLM instances
        active_chain = question_answer_chain
        if enable_auto and specialist_model:
            current_m = getattr(
                active_chain.bound if hasattr(active_chain, "bound") else active_chain,
                "model_name", "",
            )
            if specialist_model != current_m:
                if specialist_model not in _specialist_llm_cache:
                    _specialist_llm_cache[specialist_model] = get_llm(
                        model=specialist_model, streaming=True
                    )
                active_chain = prompt | _specialist_llm_cache[specialist_model]
        
        # Ensure we always have an embedding to pass back for next turn.
        # On the first turn (no last_query_emb), compute it now so the
        # semantic cache can function starting from turn 2.
        if current_emb is None:
            if is_semantic_hit:
                current_emb = last_query_emb
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)
                _exact_match_cache_store(user_input, current_emb)

        # 🚀 ASYNC SENTINEL TRIGGER
        # We trigger the summary in the background. It will be available
        # for the NEXT turn to keep the current turn latency-free.
        background_future = None
        if should_summarize and not _sentinel_in_progress:
            _sentinel_cooldown["last_turn"] = turn_count
            background_future = _background_executor.submit(_background_summarize_locked, history)

        yield {
            "context": inputs["context"], 
            "intent": intent, 
            "query_embedding": current_emb,
            "specialist_active": specialist_model if enable_auto else None,
            "top_relevance_score": reranker_score,
            "sentinel_future": background_future # Pass future to UI for persistence
        }
        
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield {"answer": content, "raw_chunk": chunk}

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
