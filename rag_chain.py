"""
rag_chain.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  • OpenRouter LLM configuration (free model by default)
  • ChromaDB retriever setup
  • LangChain retrieval chain construction
"""

from __future__ import annotations

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GEMINI_MODEL,
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
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)

# Prefix used by app.py to signal "use local Ollama with model X"
OLLAMA_PREFIX = "ollama:"


# ═══════════════════════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════════════════════

# 🚀 Professional Polish: Model Caching Capability Check
def is_cache_capable(model: str | None) -> bool:
    """
    Check if the model/provider supports prompt caching blocks.
    Explicitly returns False for local Ollama models.
    """
    if not model:
        return False
    if model.startswith(OLLAMA_PREFIX):
        return False
    
    m_lower = model.lower()
    # Supports Claude 3.5+, Gemini 2.0+, and DeepSeek on OpenRouter
    return any(p in m_lower for p in ["claude-3-5", "claude-3-haiku", "gemini-2.0", "deepseek"])


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
    
    current_model = model or GEMINI_MODEL
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


def get_reranking_retriever(db: Chroma, k: int | None = None, exclude_file: str | None = None):
    """Wrap the MMR retriever with a cross-encoder re-ranker."""
    base_retriever = get_retriever(db, k=k, exclude_file=exclude_file)
    compressor = ScoringCrossEncoderReranker(
        model=_get_cross_encoder(),
        top_n=RERANKER_TOP_N,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


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
    # 🚀 Professional Polish: Guard against non-cache models
    if not history or not ENABLE_PROMPT_CACHING or MAX_CACHE_CHECKPOINTS <= 2 or not is_cache_capable(model):
        return history

    new_history = []
    # Identify indices for breakpoints (e.g., start of history and mid-point)
    # We target the 'content' block to add the cache_control marker.
    for i, msg in enumerate(history):
        # 🚀 Final Zero-Gaps: Exactly 4 Breakpoints 
        # (1:Instructions, 2:Pinned, 3:RAG, 4:History Start)
        # We keep only the START of the window cached to ensure BP4 limit.
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
    
    if is_cc:
        system_blocks = [
            format_message_content(CORE_INSTRUCTIONS, model, use_cache=True)[0],
            format_message_content("FULL SOURCE CONTEXT (PINNED):\n{full_source_context}", model, use_cache=True)[0],
            # 🚀 Tier 2 Refinement: Context block is now part of the stable System prefix.
            # We place it ABOVE the chat history to ensure the RAG knowledge is cached.
            format_message_content("RETRIEVED RAG CHUNKS (DYNAMIC):\n{context}", model, use_cache=True)[0]
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        # Fallback to plain-string system prompt for maximum reliability
        system_text = f"{CORE_INSTRUCTIONS}\n\nFULL SOURCE CONTEXT (PINNED):\n{{full_source_context}}\n\nRETRIEVED RAG CHUNKS (DYNAMIC):\n{{context}}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    def _full_context_cache_chain(inputs: dict):
        """
        Platinum Standard: Zero-Waste Unified Path.
        Eliminates the 'Pre-flight' LLM call and enforces 100% vision.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        
        # 1. 🚀 Platinum Vision: Absolute Recall
        # Pinned files are always 100% visible (Zero-Chunking principle).
        inputs["full_source_context"] = pinned_content if (pinned_content and pinned_content != "None pinned.") else "None pinned."

        # 2. 🚀 Platinum Logic: Unified Zero-Waste Path

        # 🚀 Zero-Waste Retrieval
        docs = inputs.get("cached_docs")
        if not docs:
            # 🌊 Zero-Token Contextual Search
            # Instead of a pre-flight LLM call, we enrich the search signal
            # by prepending the human's last query to current input.
            search_signal = user_input
            if history:
                prev_human = history[-1].content if hasattr(history[-1], 'content') else ""
                # 🛡️ DILUTION GUARD: Only enrichment the search signal if the 
                # previous message was technical (> 15 chars) or the 
                # current query is a shorthand (< 10 chars).
                if len(prev_human) > 15 or len(user_input) < 10:
                    search_signal = f"{prev_human}\n{user_input}"
            
            pinned_file = inputs.get("exclude_file")
            retriever = get_reranking_retriever(db, exclude_file=pinned_file)
            docs = retriever.invoke(search_signal)
        
        inputs["context"] = docs
        # Normalise history format with model-awareness
        inputs["chat_history"] = _prepare_history_with_cache(history, model)
        
        yield {"context": docs}
        for chunk in question_answer_chain.stream(inputs):
            yield {"answer": chunk}

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
