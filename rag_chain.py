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
        
        # 🚀 ABSOLUTE DETERMINISM For Cache Stability
        # Sorting by (source, content) ensures the prompt string is 
        # mathematically identical even if the DB returns results 
        # in a slightly different order across turns.
        top_docs.sort(key=lambda d: (d.metadata.get("source", ""), d.page_content))
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

# 🚀 CONTEXT PROMPT (Dynamic/Uncached)
CONTEXT_PROMPT = "RETRIEVED CONTEXT:\n{context}"

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY CACHING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_history_with_cache(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Inject cache markers into the chat history based on the 4-breakpoint limit.
    BP 1 and 2 are used by the system instructions and context. 
    BP 3 and 4 are used here to keep history 'warm'.
    """
    if not history or not ENABLE_PROMPT_CACHING or MAX_CACHE_CHECKPOINTS <= 2:
        return history

    new_history = []
    # Identify indices for breakpoints (e.g., start of history and mid-point)
    # We target the 'content' block to add the cache_control marker.
    for i, msg in enumerate(history):
        is_bp3 = (i == 0)
        is_bp4 = (len(history) > 6 and i == len(history) // 2)

        if is_bp3 or is_bp4:
            # Convert to list-of-dicts format for Anthropic/OpenRouter
            content = [
                {
                    "type": "text", 
                    "text": msg.content, 
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        else:
            content = msg.content

        if isinstance(msg, HumanMessage):
            new_history.append(HumanMessage(content=content))
        else:
            new_history.append(AIMessage(content=content))
    return new_history

# ── Prompt used to reformulate follow-up questions into standalone ones ────
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "reformulate the question to be a standalone question that "
     "can be understood without the chat history. "
     "Do NOT answer the question — just reformulate it if needed, "
     "otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ═══════════════════════════════════════════════════════════════════════════
#  RAG CHAIN
# ═══════════════════════════════════════════════════════════════════════════

def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    # 🚀 Professional Polish: Dynamic Retrieval Configuration
    # We build our retrievers inside the lambda to support the 
    # Pinned File exclusion filter.

    # 🚀 CLAUDE-CODE GRADE ARCHITECTURE (OPTIMISED):
    # We partition the system message into frozen static blocks:
    # 1. CORE_INSTRUCTIONS (Static/Cached)
    # 2. FULL_SOURCE_CONTEXT (Static Prefix/Cached) - Pinning full files here.
    # 
    # Chat History follows this (BP 3 & 4)
    # The dynamic RAG context is moved to the VERY BOTTOM (Human Message)
    # to avoid breaking the prefix for the history.
    
    system_blocks = [
        {
            "type": "text",
            "text": CORE_INSTRUCTIONS,
            # BP 1: Instructions (always cached)
            "cache_control": {"type": "ephemeral"} if ENABLE_PROMPT_CACHING else None
        },
        {
            "type": "text",
            "text": "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}",
            # BP 2: The "Heavy Lifter" - Pinned files go here and stay warm!
            "cache_control": {"type": "ephemeral"} if ENABLE_PROMPT_CACHING else None
        }
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_blocks),
        MessagesPlaceholder("chat_history"),
        # ✅ MOVED the dynamic RAG chunks to the very bottom, below the history!
        ("human", "RETRIEVED RAG CHUNKS (DYNAMIC):\n{context}\n\nUser Question:\n{input}"),
    ])

    # Combine documents based on the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    def _full_context_cache_chain(inputs: dict):
        """
        Architecture A Polish: High-Efficiency RAG with Pre-flight Bypass
        and Smart Document Exclusion.
        """
        # Ensure full_source_context is at least a string
        inputs["full_source_context"] = inputs.get("full_source_context", "None pinned.")
        
        # 1. 🚀 Turn 1 Pre-flight Bypass (Refined)
        # Skip reformulated question LLM call if history is empty.
        # This saves ~1.5s of latency on the first turn.
        history = inputs.get("chat_history", [])
        pinned_file = inputs.get("exclude_file") # Passed from app.py
        
        # Build a fresh retriever (incredibly fast) to apply filters
        retriever = get_reranking_retriever(db, exclude_file=pinned_file)
        
        if not history:
            # Bypass! Send raw query to retriever
            docs = retriever.invoke(inputs["input"])
            inputs["context"] = docs
            
            # 🔥 Fix: app.py expects a dict with 'context' and 'answer'
            yield {"context": docs}
            
            # Run the Q&A chain and yield correctly formatted answer chunks
            for chunk in question_answer_chain.stream(inputs):
                yield {"answer": chunk}
        else:
            # 2. History-Aware Retrieval (Follow-up turns)
            # Re-formulate question for accuracy in multi-turn chat
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, CONTEXTUALIZE_Q_PROMPT
            )
            # Combine everything in the full retrieval chain
            full_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Inject history cache markers (BP 3 & 4)
            inputs["chat_history"] = _prepare_history_with_cache(history)
            
            # ✅ YIELD the stream to restore the UI typewriter effect
            yield from full_rag_chain.stream(inputs)

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
