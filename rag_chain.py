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
        
        # 🚀 ABSOLUTE DETERMINISM For Cache Stability
        # We sort by (-round(score, 2), source, content).
        # Rounding to 2 decimal places creates 'bins' of relevance,
        # ensuring minor float jitter doesn't break the cache prefix,
        # while the top-N results stay strictly at the top.
        top_docs.sort(key=lambda d: (-round(float(d.metadata.get("reranker_score", 0)), 2), 
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
        # With our Anchor pattern (Turn 0 is always index 0), 
        # BP 3 at i==0 is incredibly stable.
        is_bp3 = (i == 0)
        is_bp4 = (len(history) > 6 and i == len(history) // 2)

        content = format_message_content(msg.content, model, use_cache=(is_bp3 or is_bp4))

        if isinstance(msg, HumanMessage):
            new_history.append(HumanMessage(content=content))
        else:
            new_history.append(AIMessage(content=content))
    return new_history

# 🚀 Professional Polish: Linguistic Logic Gates (History vs Speed)
import re
STRICT_HISTORY_REGEX = re.compile(r"\b(above|previous|earlier|you said|you mentioned|as before|same as|go back)\b", re.IGNORECASE)
PINNED_REF_REGEX = re.compile(r"\b(this file|the file|pinned file|the code|this code)\b", re.IGNORECASE)

STOPWORDS = {
    "the", "a", "is", "are", "do", "does", "what", "how", "why", "where", "when", 
    "which", "can", "will", "should", "my", "your", "i", "it", "this", "that", 
    "in", "on", "for", "to", "of", "and", "or", "with", "from", "about", "has", 
    "have", "been", "not", "just", "any"
}

def is_pinned_content_relevant(query: str, content: str) -> bool:
    """
    Safety-First Relevance Gate: Determine if the pinned file should be included.
    Uses size threshold, regex matches, and a fast keyword scanner.
    """
    # 1. Size Threshold: Files under ~3k tokens (12k chars) are always included.
    if len(content) < 12000:
        return True
    
    # 2. Explicit Reference: User mentioned 'this file' or similar.
    if PINNED_REF_REGEX.search(query):
        return True
    
    # 3. Keyword Overlap: Fast scanner on the first 100 lines.
    q_tokens = {t.lower() for t in re.split(r"\W+", query) if t.lower() not in STOPWORDS and len(t) > 1}
    # Scan only a representative sample of the file for speed
    sample_content = "\n".join(content.splitlines()[:100])
    f_tokens = {t.lower() for t in re.split(r"\W+", sample_content)}
    
    if q_tokens & f_tokens:
        return True
        
    return False


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
            format_message_content("FULL SOURCE CONTEXT (PINNED):\n{full_source_context}", model, use_cache=True)[0]
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "RETRIEVED RAG CHUNKS (DYNAMIC):\n{context}\n\nUser Question:\n{input}"),
        ])
    else:
        # Fallback to plain-string system prompt for maximum reliability
        system_text = f"{CORE_INSTRUCTIONS}\n\nFULL SOURCE CONTEXT (PINNED):\n{{full_source_context}}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "RETRIEVED RAG CHUNKS (DYNAMIC):\n{context}\n\nUser Question:\n{input}"),
        ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    def _full_context_cache_chain(inputs: dict):
        """
        Architecture Industrial Core: High-Efficiency logic with
        Linguistic Logic Gates and Safety-First Relevance Gating.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        
        # 1. 🛡️ Safety-First Relevance Gate
        # Decide if the pinned file is worth the token cost for this turn.
        if pinned_content and pinned_content != "None pinned.":
            if not is_pinned_content_relevant(user_input, pinned_content):
                inputs["full_source_context"] = "[Large pinned file omitted for token efficiency. Ask about it specifically to re-enable.]"
        else:
            inputs["full_source_context"] = "None pinned."

        # 2. 🚀 Linguistic Logic Gate (Turn 1 & Smart Bypass)
        # Priority: Accuracy (History match) > Speed (Pinned match) > Default Bypass (fresh conversation).
        skip_reformulation = False
        
        if not history:
            skip_reformulation = True
        elif STRICT_HISTORY_REGEX.search(user_input):
            skip_reformulation = False
        elif PINNED_REF_REGEX.search(user_input):
            skip_reformulation = True
        elif len(history) <= 6:
            # 🚀 Tier 1 Refinement: Raised threshold to 6 messages (3 full turns)
            # for better speed in medium-length conversational bursts.
            skip_reformulation = True

        pinned_file = inputs.get("exclude_file")
        retriever = get_reranking_retriever(db, exclude_file=pinned_file)
        
        if skip_reformulation:
            # 🚀 Tier 1 Refinement: Semantic Retrieval Caching support
            # Use pre-fetched docs from app.py if available to skip DB search
            docs = inputs.get("cached_docs")
            if not docs:
                docs = retriever.invoke(user_input)
            
            inputs["context"] = docs
            # Normalise history format with model-awareness
            inputs["chat_history"] = _prepare_history_with_cache(history, model)
            yield {"context": docs}
            for chunk in question_answer_chain.stream(inputs):
                yield {"answer": chunk}
        else:
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, CONTEXTUALIZE_Q_PROMPT
            )
            full_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            inputs["chat_history"] = _prepare_history_with_cache(history, model)
            yield from full_rag_chain.stream(inputs)

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
