"""
app.py — Streamlit Frontend for the Private AI Knowledge Base.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import numpy as np
import streamlit as st

# ── Must be the very first Streamlit call ───────────────────────────────────
st.set_page_config(
    page_title="Private AI Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now safe to import project modules (they may print / use st.cache)
from backend import (
    load_and_chunk_pdf_upload,
    load_and_chunk_codebase,
    ingest_into_chroma,
    load_existing_chroma,
    list_collections,
    get_collection_info,
    delete_collection,
)
from rag_chain import build_rag_chain, OLLAMA_PREFIX
from config import GEMINI_MODEL, QWEN_MODEL, OLLAMA_MODELS
from langchain_core.messages import HumanMessage, AIMessage


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium dark theme
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
    }

    /* ── Status boxes ──────────────────────────────────────────────────── */
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    .status-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    .status-info {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(79, 70, 229, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }

    /* ── Chat bubbles ──────────────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        backdrop-filter: blur(8px);
    }

    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }

    /* ── Expander (source docs) ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_id" not in st.session_state:
    st.session_state.model_id = GEMINI_MODEL
if "pinned_file" not in st.session_state:
    st.session_state.pinned_file = None
if "pinned_content" not in st.session_state:
    st.session_state.pinned_content = "None pinned."
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
# 🚀 Platinum Standard: Semantic Retrieval Cache State
if "last_query_vector" not in st.session_state:
    st.session_state.last_query_vector = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "cache_stats" not in st.session_state:
    st.session_state.cache_stats = None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def detect_force_retrieval(query: str, collection_name: str | None) -> bool:
    """
    Detect if the user is explicitly forcing a refresh or mentioning a specific file.
    """
    force_words = ["refresh", "reload", "force", "update", "latest", "re-retrieve"]
    if any(word in query.lower() for word in force_words):
        return True
    
    if not collection_name:
        return False
        
    # Check for file mentions
    try:
        info = get_collection_info(collection_name)
        sources = [os.path.basename(s).lower() for s in info.get("sources", [])]
        query_lower = query.lower()
        # Look for any filename in the query (simple word match)
        for src in sources:
            if src in query_lower:
                return True
    except Exception:
        pass
        
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Ingestion Controls
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🧠 Knowledge Base")
    
    # ── Model Selection ────────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")
    st.caption("**Cloud (OpenRouter)**")
    model_options = {
        "Gemini 2.0 Flash (Stable)": GEMINI_MODEL,
        "Qwen 3.6 Plus (Free)": QWEN_MODEL,
    }
    # Add each installed Ollama model
    for om in OLLAMA_MODELS:
        model_options[f"Ollama — {om}"] = f"{OLLAMA_PREFIX}{om}"

    # Resolve current index
    _idx_map = {v: i for i, v in enumerate(model_options.values())}
    current_idx = _idx_map.get(st.session_state.model_id, 0)

    selected_label = st.radio(
        "Choose an LLM",
        options=list(model_options.keys()),
        index=current_idx,
        captions=[
            "Cloud", "Cloud (free)",
            *["Local" for _ in OLLAMA_MODELS],
        ],
        help="Switching models will reset the retrieval chain."
    )

    new_model_id = model_options[selected_label]

    # If model changed, reset the chain
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id
        st.session_state.rag_chain = None
        # 🚀 Invalidation: Model change clears the precision cache
        st.session_state.last_query_vector = None
        st.session_state.last_docs = []
        st.toast(f"Switched to {selected_label}", icon="🤖")

    st.divider()

    # ── Pinned Context (Architecture A) ────────────────────────────────
    st.write("---")
    st.markdown("### 📌 Pinned Context")
    st.caption("Pin a full file to the cache for 'Total Vision' and 100% cache hits.")
    
    file_to_pin = st.text_input(
        "Absolute File Path",
        placeholder="C:\\path\\to\\file.py",
        help="Paste the full path to the file you want to analyze deeply."
    )
    
    if st.button("🚀 Pin to Cache"):
        if os.path.exists(file_to_pin) and os.path.isfile(file_to_pin):
            try:
                with open(file_to_pin, "r", encoding="utf-8") as f:
                    content = f.read()
                    st.session_state.pinned_file = file_to_pin
                    st.session_state.pinned_content = f"FILE: {file_to_pin}\n\n{content}"
                st.success(f"Pinned {os.path.basename(file_to_pin)}!")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
        else:
            st.error("Invalid file path or file not found.")

    if st.session_state.pinned_file:
        st.info(f"✅ Active: **{os.path.basename(st.session_state.pinned_file)}**")
        if st.button("❌ Unpin"):
            st.session_state.pinned_file = None
            st.session_state.pinned_content = "None pinned."
            st.rerun()

    st.write("---")
    st.markdown("### 📂 Ingestion")
    collection_name = st.text_input(
        "Collection name",
        value="default",
        help="Give each ingestion a unique name to keep them separate.",
    )

    # ── PDF Upload ─────────────────────────────────────────────────────
    st.markdown("### 📄 Ingest PDF")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
    )
    if st.button("⚡ Ingest PDF", disabled=uploaded_pdf is None, use_container_width=True):
        with st.spinner("Reading & chunking PDF…"):
            chunks = load_and_chunk_pdf_upload(uploaded_pdf, uploaded_pdf.name)
        with st.spinner(f"Embedding {len(chunks)} chunks into ChromaDB…"):
            _, added = ingest_into_chroma(chunks, collection_name=collection_name)
        st.success(f"✅ Ingested **{added}** new chunks from `{uploaded_pdf.name}` ({len(chunks) - added} duplicates skipped)")
        st.session_state.active_collection = collection_name
        st.session_state.rag_chain = None        # force rebuild
        # 🚀 Invalidation: New ingestion clears existing cache
        st.session_state.last_query_vector = None
        st.session_state.last_docs = []

    st.divider()

    # ── Codebase folder ────────────────────────────────────────────────
    st.markdown("### 💻 Ingest Codebase")
    folder_path = st.text_input(
        "Local folder path",
        placeholder=r"C:\Users\HP\my_project",
    )
    if st.button("⚡ Ingest Folder", disabled=not folder_path, use_container_width=True):
        if not os.path.isdir(folder_path):
            st.error("❌ Directory not found. Check the path.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _update_progress(current: int, total: int, filename: str):
                progress_bar.progress(current / total if total else 1.0)
                status_text.text(f"Processing {current}/{total}: {filename}")

            chunks = load_and_chunk_codebase(folder_path, on_progress=_update_progress)
            progress_bar.empty()
            status_text.empty()
            if not chunks:
                st.warning("⚠️ No supported code files found in that directory.")
            else:
                with st.spinner(f"Embedding {len(chunks)} chunks into ChromaDB…"):
                    _, added = ingest_into_chroma(chunks, collection_name=collection_name)
                st.success(f"✅ Ingested **{added}** new chunks from `{folder_path}` ({len(chunks) - added} duplicates skipped)")
                st.session_state.active_collection = collection_name
                st.session_state.rag_chain = None
                # 🚀 Invalidation: Folder ingestion clears existing cache
                st.session_state.last_query_vector = None
                st.session_state.last_docs = []

    st.divider()

    # ── Load existing collection ───────────────────────────────────────
    st.markdown("### 🗃️ Existing Collections")
    existing = list_collections()
    if existing:
        chosen = st.selectbox("Select a collection", existing)

        # Show collection stats
        info = get_collection_info(chosen)
        st.caption(f"📊 {info['count']} chunks | {len(info['sources'])} source files")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 Connect", use_container_width=True):
                st.session_state.active_collection = chosen
                st.session_state.rag_chain = None
                # 🚀 Invalidation: Switching collections clears context
                st.session_state.last_query_vector = None
                st.session_state.last_docs = []
                st.success(f"Connected to **{chosen}**")
        with col2:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state["confirm_delete"] = chosen

        # Confirmation dialog
        if st.session_state.get("confirm_delete") == chosen:
            st.warning(f"Delete collection '{chosen}'? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, delete", use_container_width=True):
                    delete_collection(chosen)
                    if st.session_state.active_collection == chosen:
                        st.session_state.active_collection = None
                        st.session_state.rag_chain = None
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
    else:
        st.info("No collections yet — ingest a PDF or codebase first.")

    st.divider()

    # ── Clear chat ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # ── Developer tools ───────────────────────────────────────────────
    st.markdown("### 🛠️ Developer Tools")
    debug_mode = st.toggle("Debug Mode", value=False, help="Show retrieval details and reranker scores")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN AREA — Chat Interface
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🤖 Private AI Knowledge Base & Code Assistant")

# ── Connection status ──────────────────────────────────────────────────────
if st.session_state.active_collection:
    st.markdown(
        f'<div class="status-box status-success">'
        f'🟢 Connected to collection: <strong>{st.session_state.active_collection}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-box status-info">'
        '💡 Upload a PDF or point to a code folder in the sidebar to get started.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Render chat history ───────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source Documents"):
                for i, src in enumerate(msg["sources"], 1):
                    source_file = src.get("source", "Unknown")
                    st.markdown(f"**Chunk {i}** — `{source_file}`")
                    st.code(src.get("content", ""), language="text")

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your documents or code…")

if user_input:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Build or reuse rag_chain ────────────────────────────────────────
    if st.session_state.rag_chain is None:
        coll = st.session_state.active_collection or "default"
        db = load_existing_chroma(collection_name=coll)
        if db is None:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ No knowledge base found. "
                    "Please ingest a PDF or codebase first using the sidebar."
                )
            st.stop()
        try:
            st.session_state.vector_db = db  # 🚀 Fix: Persist DB for embedding access
            st.session_state.rag_chain = build_rag_chain(db, model=st.session_state.model_id)
        except ValueError as e:
            with st.chat_message("assistant"):
                st.error(f"⚙️ {e}")
            st.stop()

    chain = st.session_state.rag_chain

    # ── Build LangChain-format chat history ───────────────────────────────
    lc_history = []
    for msg in st.session_state.chat_history[:-1]:  # exclude current user msg
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    # ── Stream the response ─────────────────────────────────────────────
    with st.chat_message("assistant"):
        try:
            # We'll use this to capture the final context/answer for history
            full_response = {"answer": "", "context": []}

            def response_generator():
                # 🚀 Accuracy-First: HYBRID INTENT GUARD
                # Detect if we should FORCE a refresh regardless of similarity
                is_forced = detect_force_retrieval(user_input, st.session_state.active_collection)
                
                cached_docs = None
                q_vector = None
                
                if st.session_state.vector_db and not is_forced:
                    # 1. Embed query
                    q_vector = st.session_state.vector_db.embeddings.embed_query(user_input)
                    
                    # 2. 🚀 PLATINUM SLIDING WINDOW: Union Retrieval
                    # We always retrieve fresh docs and merge them with previous context
                    # to maintain a stable cached prefix while adding new info.
                    if st.session_state.last_query_vector is not None and st.session_state.last_docs:
                        sim = np.dot(q_vector, st.session_state.last_query_vector)
                        
                        # Only attempt Union if we have some semantic overlap
                        if sim >= 0.70:
                            from config import RERANKER_TOP_N
                            # Perform new retrieval
                            fresh_docs = st.session_state.vector_db.similarity_search(user_input, k=RERANKER_TOP_N)
                            
                            # Intersection: Docs present in both (Stable Prefix)
                            last_content = [d.page_content for d in st.session_state.last_docs]
                            intersection = [d for d in st.session_state.last_docs if d.page_content in [f.page_content for f in fresh_docs]]
                            
                            # New Chunks: Unique docs from fresh search
                            new_chunks = [f for f in fresh_docs if f.page_content not in last_content]
                            
                            # Construct Union: Intersection first (Cache Stability) + New Chunks
                            stable_docs = intersection
                            new_docs = new_chunks[: (RERANKER_TOP_N * 2) - len(intersection)]
                            
                            cached_docs = {"stable": stable_docs, "new": new_docs}
                            
                            if intersection:
                                st.toast(f"Reusing {len(intersection)} stable chunks", icon="♻️")
                
                # 🚀 Accuracy-First: GHOST HISTORY
                if len(lc_history) <= 10:
                    truncated_history = lc_history
                else:
                    anchor = lc_history[:2]
                    ghosts = [msg for msg in lc_history[2:-8] if isinstance(msg, HumanMessage)]
                    window = lc_history[-8:]
                    truncated_history = anchor + ghosts + window
                
                stream_iter = chain.stream({
                    "input": user_input, 
                    "chat_history": truncated_history,
                    "full_source_context": st.session_state.pinned_content,
                    "exclude_file": st.session_state.pinned_file,
                    "cached_docs": cached_docs
                })

                for chunk in stream_iter:
                    # Extract cache metadata if available (OpenRouter/Anthropic)
                    if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                        meta = chunk.response_metadata
                        cache_read = meta.get("anthropic-ratelimit-input-tokens-cache-read")
                        cache_create = meta.get("anthropic-ratelimit-input-tokens-cache-creation")
                        if cache_read or cache_create:
                            st.session_state.cache_stats = {
                                "read": cache_read,
                                "created": cache_create
                            }

                    if "context" in chunk:
                        full_response["context"] = chunk["context"]
                        st.session_state.last_docs = chunk["context"]
                        st.session_state.last_query_vector = q_vector
                    if "answer" in chunk:
                        full_response["answer"] += chunk["answer"]
                        yield chunk["answer"]
            
            # Execute streaming
            answer = st.write_stream(response_generator())
            result = full_response
            
        except Exception as e:
            st.error(f"❌ LLM error: {e}")
            st.stop()

        # Show source documents and Cache Stats
        source_docs = result.get("context", [])
        sources_meta = []
        
        # 🚀 Cache Usage Transparency
        if st.session_state.get("cache_stats"):
            stats = st.session_state.cache_stats
            st.caption(f"⚡ Cache: **{stats.get('read', 0)}** read | **{stats.get('created', 0)}** created")

        if source_docs:
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(source_docs, 1):
                    src_file = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {i}** — `{src_file}`")
                    st.code(doc.page_content[:500], language="text")
                    sources_meta.append({
                        "source": src_file,
                        "content": doc.page_content[:500],
                    })

        # Debug panel — retrieval details & reranker scores
        if debug_mode and source_docs:
            with st.expander("🛠️ Debug: Retrieval Details", expanded=True):
                st.markdown(f"**Collection:** `{st.session_state.active_collection}`")
                st.markdown(f"**Chunks retrieved:** {len(source_docs)}")
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"---\n**Chunk {i}**")
                    st.json({
                        "source": doc.metadata.get("source", "Unknown"),
                        "source_type": doc.metadata.get("source_type", "unknown"),
                        "file_extension": doc.metadata.get("file_extension", ""),
                        "content_hash": doc.metadata.get("content_hash", ""),
                        "reranker_score": doc.metadata.get("reranker_score", "N/A"),
                        "content_length": len(doc.page_content),
                    })
                    st.code(doc.page_content, language="text")

    # Persist to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_meta,
    })
