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
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=model or GEMINI_MODEL,
        temperature=temp,
        streaming=streaming,
        max_tokens=MAX_TOKENS,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Private AI Knowledge Base",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

def get_retriever(db: Chroma, k: int | None = None):
    """
    Wrap a ChromaDB store as a LangChain retriever.

    Parameters
    ----------
    db : Chroma
        A populated Chroma vector store.
    k  : int, optional
        Number of top-k chunks to return (default from config).
    """
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
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
        return top_docs


def get_reranking_retriever(db: Chroma, k: int | None = None):
    """Wrap the MMR retriever with a cross-encoder re-ranker."""
    base_retriever = get_retriever(db, k=k)
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

SYSTEM_PROMPT = """\
You are an expert AI assistant specialising in code analysis and document comprehension.

INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved context below.
2. If the context does not contain enough information, say so clearly — \
   do NOT fabricate an answer.
3. When discussing code, reference the source file and explain the logic.
4. Be concise, precise, and use markdown formatting where helpful.
5. If the user asks for code improvements, provide the improved version \
   with clear explanations.

RETRIEVED CONTEXT:
{context}
"""

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
    Build and return a LangChain *retrieval chain* with conversation memory.

    The chain:
      1. Reformulates the user question using chat history (if any).
      2. Embeds the standalone question locally.
      3. Retrieves top-k relevant chunks from ChromaDB via MMR.
      4. Stuffs them into the system prompt.
      5. Sends the combined prompt to the OpenRouter LLM.
      6. Returns ``{"answer": str, "context": list[Document]}``.
    """
    llm = get_llm(model=model)
    base_retriever = get_reranking_retriever(db)

    # Wrap retriever to handle conversation history
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, CONTEXTUALIZE_Q_PROMPT
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # "Stuff" all retrieved docs into the {context} placeholder
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Full retrieval chain: retrieve → stuff → generate
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
