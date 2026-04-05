"""
app.py — Streamlit Frontend for the Private AI Knowledge Base.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
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
    load_and_chunk_codebase,
    ingest_into_chroma,
    load_existing_chroma,
    list_collections,
    get_collection_info,
    delete_collection,
)
from rag_chain import build_rag_chain, OLLAMA_PREFIX
from config import (
    DEFAULT_MODEL, CLOUDROUTER_MODELS, OLLAMA_MODELS,
    SEMANTIC_CACHE_THRESHOLD, PINNED_RELEVANCE_THRESHOLD,
    GHOST_HISTORY_WINDOW, GHOST_HISTORY_MAX, AI_RESPONSE_MAX_CHARS,
    GHOST_AI_CHARS, MAX_HISTORY_TOKENS,
    ENABLE_AUTO_SPECIALIST,
)
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
    st.session_state.model_id = DEFAULT_MODEL
if "pinned_file" not in st.session_state:
    st.session_state.pinned_file = None
if "pinned_content" not in st.session_state:
    st.session_state.pinned_content = "None pinned."
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
# 🚀 Platinum Standard: Semantic Retrieval Cache State
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {}
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []
if "specialist_counts" not in st.session_state:
    st.session_state.specialist_counts = {"CODE": 0, "REASONING": 0, "VISION": 0, "GENERAL": 0}
if "last_query_embedding" not in st.session_state:
    st.session_state.last_query_embedding = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "ingestion_task" not in st.session_state:
    st.session_state.ingestion_task = None
if "ingestion_done_processed" not in st.session_state:
    st.session_state.ingestion_done_processed = False
if "sentinel_state" not in st.session_state:
    st.session_state.sentinel_state = "No summary generated yet."
if "sentinel_future" not in st.session_state:
    st.session_state.sentinel_future = None
if "filter_extensions" not in st.session_state:
    st.session_state.filter_extensions = []


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def extract_usage_metadata(raw_chunk) -> dict:
    """
    Intelligently extract token usage across different providers.
    Supports OpenRouter (Cloud), Gemini, Anthropic-style caching, and Ollama.
    """
    if not (raw_chunk and hasattr(raw_chunk, "response_metadata") and raw_chunk.response_metadata):
        return {}
    
    meta = raw_chunk.response_metadata
    usage = {}
    
    # ── 1. Standard OpenAI/OpenRouter 'token_usage' ────────────────────
    if "token_usage" in meta:
        tu = meta["token_usage"]
        usage["input"] = tu.get("prompt_tokens")
        usage["output"] = tu.get("completion_tokens")
        usage["total"] = tu.get("total_tokens")
        # OpenRouter sometimes puts cached tokens here
        if "prompt_tokens_details" in tu:
            details = tu["prompt_tokens_details"]
            if isinstance(details, dict) and "cached_tokens" in details:
                usage["cache_read"] = details["cached_tokens"]
    
    # ── 2. Gemini / Generic 'usage' ────────────────────────────────────
    if "usage" in meta:
        gu = meta["usage"]
        if isinstance(gu, dict):
            usage["input"] = usage.get("input", gu.get("prompt_tokens"))
            usage["output"] = usage.get("output", gu.get("completion_tokens"))
            usage["total"] = usage.get("total", gu.get("total_tokens"))
                
            # Gemini cached_tokens
            if "prompt_tokens_details" in gu:
                details = gu["prompt_tokens_details"]
                if isinstance(details, dict) and "cached_tokens" in details:
                    usage["cache_read"] = details["cached_tokens"]

    # ── 3. Anthropic & OpenRouter Hardware headers ─────────────────────
    # Check both direct metadata and common provider-specific headers
    cache_read = (meta.get("anthropic-ratelimit-input-tokens-cache-read") or 
                  meta.get("cache_read_tokens") or 
                  meta.get("tokens_cached"))
    
    cache_create = (meta.get("anthropic-ratelimit-input-tokens-cache-creation") or 
                    meta.get("cache_creation_tokens"))
                    
    if cache_read is not None:
        usage["cache_read"] = usage.get("cache_read", cache_read)
    if cache_create is not None:
        usage["cache_create"] = usage.get("cache_create", cache_create)

    # ── 4. Local Ollama ────────────────────────────────────────────────
    if not usage.get("input"):
       usage["input"] = meta.get("prompt_eval_count") or meta.get("input_tokens") or 0
    if not usage.get("output"):
       usage["output"] = meta.get("eval_count") or meta.get("output_tokens") or 0

    return {k: (v if v is not None else 0) for k, v in usage.items()}


def _fetch_generation_usage(generation_id: str) -> dict:
    """
    Fetch token usage from OpenRouter's generation endpoint.
    Waits 1 second for OpenRouter to finalise the record, then fetches.
    This runs in a background thread so the UI is never blocked.
    """
    import time
    import requests
    from config import OPENROUTER_API_KEY

    if not OPENROUTER_API_KEY:
        return {}

    time.sleep(1)  # OpenRouter needs a moment to finalize the record
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=5)
        if resp.status_code != 200:
            return {}
        data = resp.json().get("data", {})
        usage = {
            "input": data.get("tokens_prompt", 0),
            "output": data.get("tokens_completion", 0),
            "total": (data.get("tokens_prompt", 0) + data.get("tokens_completion", 0)),
        }
        cached = data.get("tokens_cached", 0)
        if cached:
            usage["cache_read"] = cached
        return usage
    except Exception:
        return {}


def _truncate_ai_in_history(history: list) -> list:
    """
    Cap AI response length in chat history to reduce token waste,
    while aggressively preserving code blocks so the LLM remembers
    the actual code it wrote.
    """
    import re
    truncated = []
    for msg in history:
        if isinstance(msg, AIMessage) and len(msg.content) > AI_RESPONSE_MAX_CHARS:
            code_blocks = re.findall(r"(```.*?```)", msg.content, flags=re.DOTALL)
            if code_blocks:
                gist = msg.content[:400]
                trimmed = f"{gist}\n... [prose truncated]\n\n" + "\n\n".join(code_blocks)
                if len(trimmed) > AI_RESPONSE_MAX_CHARS * 3:
                    trimmed = trimmed[:AI_RESPONSE_MAX_CHARS * 3] + "\n```\n... [code truncated]"
            else:
                trimmed = msg.content[:AI_RESPONSE_MAX_CHARS] + "\n... [truncated for context efficiency]"
            truncated.append(AIMessage(content=trimmed))
        else:
            truncated.append(msg)
    return truncated


def detect_force_retrieval(query: str, collection_name: str | None) -> bool:
    """
    Detect if the user is explicitly forcing a refresh or mentioning a specific file.
    Uses regex for word-boundary matching to avoid false positives on common substrings.

    The source-file list is cached in session state per collection so that
    ChromaDB is not scanned on every single query turn.
    """
    import re
    force_words = ["refresh", "reload", "force", "update", "latest", "re-retrieve"]
    query_lower = query.lower()

    # 1. Check for force keywords
    for word in force_words:
        if re.search(rf"\b{re.escape(word)}\b", query_lower):
            return True

    if not collection_name:
        return False

    # 2. Check for specific file mentions using a cached source list.
    #    The cache is keyed by collection name and is invalidated when
    #    the active collection changes (handled in the connect/ingest flow).
    cache_key = f"_source_names_cache_{collection_name}"
    sources: list[str] = st.session_state.get(cache_key)
    if sources is None:
        try:
            from backend import get_collection_info
            info = get_collection_info(collection_name)
            sources = [os.path.basename(s).lower() for s in info.get("sources", [])]
            st.session_state[cache_key] = sources
        except Exception:
            sources = []

    for src in sources:
        if re.search(rf"\b{re.escape(src)}\b", query_lower):
            return True

    return False


# 🚀 Side Effect: Removed get_dynamic_threshold (Replaced by Backend Union Retrieval)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Ingestion Controls
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🧠 Knowledge Base")
    
    # ── Model Selection ────────────────────────────────────────────────
    # ── Model Selection (Categorised) ──────────────────────────────────
    st.markdown("### 🤖 Model Selection")
    
    # Flatten categories for finding the current model's category
    model_to_category = {}
    for cat, models in CLOUDROUTER_MODELS.items():
        for name, mid in models.items():
            model_to_category[mid] = (cat, name)
    
    # 1. Choose Category
    categories = list(CLOUDROUTER_MODELS.keys()) + ["Local (Ollama)"]
    
    # Determine initial category
    current_mid = st.session_state.model_id
    if current_mid.startswith(OLLAMA_PREFIX):
        initial_cat = "Local (Ollama)"
    else:
        initial_cat = model_to_category.get(current_mid, (categories[0], ""))[0]
    
    selected_cat = st.selectbox(
        "Model Tier / Specialisation",
        options=categories,
        index=categories.index(initial_cat) if initial_cat in categories else 0
    )
    
    # 2. Choose Model within Category
    if selected_cat == "Local (Ollama)":
        if not OLLAMA_MODELS:
            st.error("No local models found in config.")
            new_model_id = DEFAULT_MODEL
        else:
            local_options = {m: f"{OLLAMA_PREFIX}{m}" for m in OLLAMA_MODELS}
            selected_local = st.selectbox("Select local model", options=list(local_options.keys()))
            new_model_id = local_options[selected_local]
    else:
        # Cloud models
        tier_models = CLOUDROUTER_MODELS[selected_cat]
        selected_model_name = st.selectbox(
            f"Select {selected_cat} AI",
            options=list(tier_models.keys()),
            index=list(tier_models.values()).index(current_mid) if current_mid in tier_models.values() else 0
        )
        new_model_id = tier_models[selected_model_name]

    # If model changed, reset the chain
    if new_model_id != st.session_state.model_id:
        st.session_state.model_id = new_model_id
        st.session_state.rag_chain = None
        # 🚀 Invalidation: Model change clears context and tracking
        st.session_state.last_docs = []
        st.session_state.last_query_embedding = None
        st.toast(f"Switched to model: {st.session_state.model_id}", icon="🤖")

    # 🛠️ Phase 4: Auto-Specialist Agent
    from config import ENABLE_AUTO_SPECIALIST as _DEFAULT_AUTO
    st.session_state.auto_specialist = st.toggle(
        "🤖 Auto-Specialist Agent",
        value=st.session_state.get("auto_specialist", _DEFAULT_AUTO),
        help="Automatically switch to the best model for code, reasoning, or general tasks."
    )
    if st.session_state.auto_specialist:
        st.caption("✨ *Orchestrating the best models for your query...*")

    st.divider()

    # ── Pinned Context (Architecture A) ────────────────────────────────
    st.write("---")
    st.markdown("### 📌 Pinned Context")
    st.caption("Pin a full file to the cache for **Total Vision** (Architecture A).")
    
    file_to_pin = st.text_input(
        "Absolute File Path",
        placeholder="C:\\path\\to\\file.v",
        help="Paste the full path to a specific file (e.g., your top-level Verilog module)."
    )
    
    if file_to_pin and os.path.exists(file_to_pin):
        if os.path.isdir(file_to_pin):
            st.warning("📁 **Directory detected!** Pinned Context is designed for single files. To analyze this entire folder, use the **Ingestion** section below.")
            
            # Intelligent Helper: List files in the directory
            try:
                files = [f for f in os.listdir(file_to_pin) if os.path.isfile(os.path.join(file_to_pin, f))]
                if files:
                    selected_subfile = st.selectbox("Pick a specific file to pin:", options=files)
                    if st.button("🚀 Pin Selected File", use_container_width=True):
                        full_path = os.path.join(file_to_pin, selected_subfile)
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            st.session_state.pinned_file = full_path
                            st.session_state.pinned_content = f"FILE: {full_path}\n\n{f.read()}"
                        st.success(f"Pinned {selected_subfile}!")
                        st.rerun()
                else:
                    st.info("No files found in this directory.")
            except Exception as e:
                st.error(f"Error reading directory: {e}")
            
            # Quick Ingest Bridge
            if st.button("⚡ Ingest Folder for RAG Instead", use_container_width=True):
                st.session_state["pending_folder"] = file_to_pin
                st.toast("Ready to ingest below!", icon="📂")
        
        elif os.path.isfile(file_to_pin):
            if st.button("🚀 Pin to Cache", use_container_width=True):
                try:
                    with open(file_to_pin, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        st.session_state.pinned_file = file_to_pin
                        st.session_state.pinned_content = f"FILE: {file_to_pin}\n\n{content}"
                    st.success(f"Pinned {os.path.basename(file_to_pin)}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
    elif file_to_pin:
        st.error("Invalid file path or file not found.")

    if st.session_state.pinned_file:
        st.info(f"✅ Active: **{os.path.basename(st.session_state.pinned_file)}**")
        if st.button("❌ Unpin", use_container_width=True):
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
    if st.button("⚡ Ingest PDF", disabled=uploaded_pdf is None or st.session_state.ingestion_task is not None, use_container_width=True):
        # 🚀 Fix: Use background task for PDF ingestion
        import tempfile
        from pathlib import Path
        suffix = Path(uploaded_pdf.name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        
        from backend import AsyncIngestionTask
        task = AsyncIngestionTask(tmp_path, collection_name=collection_name, is_pdf=True)
        task.start()
        st.session_state.ingestion_task = task
        st.session_state.ingestion_done_processed = False
        st.rerun()

    st.divider()

    # ── Codebase folder ────────────────────────────────────────────────
    st.markdown("### 💻 Ingest Codebase")
    folder_path = st.text_input(
        "Local folder path",
        value=st.session_state.pop("pending_folder", "") if "pending_folder" in st.session_state else "",
        placeholder=r"C:\Users\HP\my_project",
    )
    if st.button("⚡ Ingest Folder", disabled=not folder_path or st.session_state.ingestion_task is not None, use_container_width=True):
        if not os.path.isdir(folder_path):
            st.error("❌ Directory not found. Check the path.")
        else:
            from backend import AsyncIngestionTask
            task = AsyncIngestionTask(folder_path, collection_name=collection_name, is_pdf=False)
            task.start()
            st.session_state.ingestion_task = task
            st.session_state.ingestion_done_processed = False
            st.rerun()

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
                # Invalidation: Switching collections clears context + source cache
                st.session_state.last_docs = []
                st.session_state.last_query_embedding = None
                st.session_state.pop(f"_source_names_cache_{chosen}", None)
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
        st.session_state.sentinel_state = ""
        st.session_state.last_docs = []
        st.session_state.last_query_embedding = None
        st.rerun()

    st.divider()

    # ── Developer tools ───────────────────────────────────────────────
    st.markdown("### 🛠️ Developer Tools")
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode, help="Show retrieval details and reranker scores")

    # ── Cache Strategy ────────────────────────────────────────────────
    import config as _cfg
    trust_native = st.toggle(
        "Trust Native Cache",
        value=_cfg.TRUST_NATIVE_CACHE,
        help=(
            "When ON: always retrieve fresh chunks every turn. "
            "If the same chunks return, the provider cache kicks in at ~10% cost. "
            "Avoids stale-context hallucinations on subtle topic shifts.\n\n"
            "When OFF: skip retrieval when queries are very similar (semantic cache). "
            "Faster, but risks serving wrong context if the topic subtly shifts."
        ),
    )
    if trust_native != _cfg.TRUST_NATIVE_CACHE:
        _cfg.TRUST_NATIVE_CACHE = trust_native
        st.toast(
            "Always-retrieve mode ON" if trust_native else "Semantic cache ON",
            icon="🔄" if trust_native else "⚡",
        )

    st.divider()

    # ── Metadata Filtering (Phase 1b) ────────────────────────────────
    st.markdown("### 🔍 Retrieval Filters")
    st.caption("Narrow retrieval to specific file types.")
    filter_options = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go",
                      ".v", ".sv", ".html", ".css", ".md", ".json", ".sql"]
    selected_filters = st.multiselect(
        "Filter by extension",
        options=filter_options,
        default=[],
        help="Only retrieve chunks from files with these extensions. Leave empty for all."
    )
    st.session_state.filter_extensions = selected_filters if selected_filters else []

    st.divider()

    # ── Summarize-and-Pin (Phase 2b) ─────────────────────────────────
    st.markdown("### 📝 Summarize & Pin")
    st.caption("For large files: pin a compact summary instead of the full file.")
    summarize_path = st.text_input(
        "File to summarize",
        placeholder="C:\\path\\to\\large_file.py",
        key="summarize_path_input",
    )
    if summarize_path and os.path.isfile(summarize_path):
        if st.button("📝 Summarize & Pin", use_container_width=True):
            from backend import summarize_document_for_pin
            summary = summarize_document_for_pin(summarize_path, max_chars=3000)
            if summary:
                st.session_state.pinned_file = summarize_path
                st.session_state.pinned_content = (
                    f"FILE (SUMMARIZED): {summarize_path}\n\n{summary}"
                )
                st.success(f"Pinned summary of {os.path.basename(summarize_path)} ({len(summary)} chars vs full file)")
                st.rerun()
            else:
                st.error("Could not generate summary — file may be empty.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN AREA — Chat Interface
# ═══════════════════════════════════════════════════════════════════════════

# 📊 Phase 5: Universal Telemetry Dashboard
with st.sidebar.expander("📊 Live Performance Insights", expanded=False):
    if not st.session_state.metrics_history:
        st.info("Run a query to see live performance data.")
    else:
        import pandas as pd
        df = pd.DataFrame(st.session_state.metrics_history)
        
        # 1. Savings Chart
        st.write("**Cumulative Token Savings**")
        df["Cumulative Cached"] = df["cached_tokens"].cumsum()
        st.area_chart(df, y="Cumulative Cached", x="turn", use_container_width=True)
        
        # 2. Specialist Breakdown
        st.write("**Specialist Agent Workload**")
        s_df = pd.DataFrame(list(st.session_state.specialist_counts.items()), columns=["Agent", "Count"])
        st.bar_chart(s_df, x="Agent", y="Count", use_container_width=True)
        
        # 3. Recall Health
        avg_score = df["relevance_score"].mean()
        st.metric("Avg Recall Health", f"{avg_score:.2f}", help="Score from local BGE re-ranker (0.0 - 1.0)")
        st.progress(max(0.0, min(1.0, avg_score)))

st.divider()

# 🤖 Private AI Knowledge Base & Code Assistant

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

# ── 🚀 BACKGROUND INGESTION PROGRESS ─────────────────────────────────────────
if st.session_state.ingestion_task:
    task = st.session_state.ingestion_task
    
    if task.status == "running":
        progress_val = min(1.0, task.progress)
        st.info(f"⚡ **Background Ingestion in Progress...**")
        st.progress(progress_val)
        st.caption(f"Currently: {task.current_step}")
        import time
        time.sleep(1) # Simple polling mechanism
        st.rerun()
        
    elif task.status == "done" and not st.session_state.ingestion_done_processed:
        # Success state: Handle the completed result
        _, added = task.result
        st.success(f"✅ **Ingestion Complete!** Added **{added}** new chunks.")
        st.session_state.active_collection = task.collection_name
        st.session_state.rag_chain = None
        st.session_state.last_docs = []
        # Invalidate source cache so detect_force_retrieval rescans
        st.session_state.pop(f"_source_names_cache_{task.collection_name}", None)
        st.session_state.ingestion_done_processed = True
        # Keep success message visible for a bit
        if st.button("Clear Notification", key="clear_ingest"):
            st.session_state.ingestion_task = None
            st.rerun()

    elif task.status == "error":
        st.error(f"❌ **Ingestion Failed:** {task.error}")
        if st.button("Clear Error", key="clear_ingest_err"):
            st.session_state.ingestion_task = None
            st.rerun()

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
        coll = st.session_state.active_collection
        if not coll:
            coll = "default"
            st.toast("No collection selected — using 'default'", icon="⚠️")
        db = load_existing_chroma(collection_name=coll)
        
        # 🚀 Fix: If no DB but we have physical pinned content, we can still chat!
        has_pinned = st.session_state.get("pinned_file") is not None
        
        if db is None and not has_pinned:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ No knowledge base found. "
                    "Please ingest a PDF or codebase first using the sidebar."
                )
            st.stop()
        try:
            st.session_state.vector_db = db  # 🚀 Fix: Persist DB for embedding access (may be None)
            st.session_state.rag_chain = build_rag_chain(db, model=st.session_state.model_id)
        except ValueError as e:
            with st.chat_message("assistant"):
                st.error(f"⚙️ {e}")
            st.stop()

    chain = st.session_state.rag_chain

    # 🚀 Architectural Overhaul: Update sentinel state from background thread
    if st.session_state.get("sentinel_future") and st.session_state.sentinel_future.done():
        try:
            new_state = st.session_state.sentinel_future.result()
            if new_state:
                st.session_state.sentinel_state = new_state
                if st.session_state.get("debug_mode"):
                    st.toast("✅ Sentinel Summary Updated (Background)", icon="🤖")
        except Exception as e:
            st.error(f"Sentinel background error: {e}")
        st.session_state.sentinel_future = None

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
            full_response = {"answer": "", "context": []}
            st.session_state.token_usage = {}

            def response_generator():
                # 🚀 Native-First Cache: We always retrieve context to prevent hallucinations.
                # The backend handles union, deduplication, and deterministic sorting.
                cached_docs = st.session_state.get("last_docs", [])
                last_emb = st.session_state.get("last_query_embedding")
                is_forced = detect_force_retrieval(user_input, st.session_state.active_collection)

                pinned_to_send = st.session_state.get("pinned_content", "None pinned.")
                
                # Build history: rely on Sentinel State when available
                if st.session_state.sentinel_state and st.session_state.sentinel_state != "No summary generated yet.":
                    # If we have a summary, only keep the immediate context (last 4 messages)
                    truncated_history = lc_history[-4:]
                    truncated_history = _truncate_ai_in_history(truncated_history)
                else:
                    # Fallback to ghost logic before the first summarization
                    if len(lc_history) <= GHOST_HISTORY_MAX:
                        truncated_history = _truncate_ai_in_history(lc_history)
                    else:
                        anchor = lc_history[:2]
                        # Keep BOTH Human and AI messages in the ghost section,
                        # but aggressively truncate AI responses so the model
                        # still sees its own prior answers in condensed form.
                        ghost_section = lc_history[2:-GHOST_HISTORY_WINDOW]
                        ghosts = []
                        for msg in ghost_section:
                            if isinstance(msg, AIMessage):
                                trimmed = msg.content[:GHOST_AI_CHARS]
                                if len(msg.content) > GHOST_AI_CHARS:
                                    trimmed += "\n... [truncated]"
                                ghosts.append(AIMessage(content=trimmed))
                            else:
                                ghosts.append(msg)
                        window = lc_history[-GHOST_HISTORY_WINDOW:]
                        truncated_history = _truncate_ai_in_history(anchor + ghosts + window)

                    # Hard token budget: drop oldest ghost messages until under budget
                    def _est_tokens(msgs):
                        return sum(len(m.content) for m in msgs) // 4
                    while _est_tokens(truncated_history) > MAX_HISTORY_TOKENS and len(truncated_history) > 4:
                        # Remove the 3rd message (first ghost after anchor pair)
                        truncated_history.pop(2)

                stream_iter = chain.stream({
                    "input": user_input,
                    "chat_history": truncated_history,
                    "full_source_context": pinned_to_send,
                    "exclude_file": st.session_state.get("pinned_file"),
                    "cached_docs": cached_docs,
                    "last_query_embedding": last_emb,
                    "force_retrieval": is_forced,
                    "collection_name": st.session_state.active_collection or "default",
                    "sentinel_state": st.session_state.sentinel_state,
                    "filter_extensions": st.session_state.filter_extensions or None,
                    "auto_specialist": st.session_state.auto_specialist,
                })

                generation_id = None
                for chunk in stream_iter:
                    # ── 🤖 INTELLIGENT METADATA ───────────────────────────
                    if "intent" in chunk:
                        intent = chunk["intent"]
                        if intent == "FOLLOW-UP":
                            st.toast("Local LLM: Contextual Refinement 🤖", icon="🧠")
                        elif intent == "SEMANTIC-HIT":
                            st.toast("⚡ Semantic Cache Hit: Reusing Context", icon="🔥")
                        else:
                            st.toast("Local LLM: New Concept Detected 🤖", icon="✨")

                    if "specialist_active" in chunk:
                        specialist = chunk["specialist_active"]
                        if specialist and specialist != st.session_state.model_id:
                            st.toast(f"🦾 Specialist Engaged: {specialist}", icon="⚡")

                    if "query_embedding" in chunk:
                        st.session_state.last_query_embedding = chunk["query_embedding"]

                    if "context" in chunk:
                        full_response["context"] = chunk["context"]
                        st.session_state.last_docs = chunk["context"]

                    if "answer" in chunk:
                        full_response["answer"] += chunk["answer"]
                        yield chunk["answer"]

                    # Capture usage metadata
                    raw = chunk.get("raw_chunk")
                    if raw:
                        if hasattr(raw, "response_metadata") and raw.response_metadata:
                            gen_id = raw.response_metadata.get("id")
                            if gen_id: generation_id = gen_id
                            
                            new_usage = extract_usage_metadata(raw)
                            if new_usage: st.session_state.token_usage = new_usage
                            if st.session_state.debug_mode:
                                st.session_state["debug_meta"] = raw.response_metadata
                    
                    # Phase 5: Capture specialist engagement forpie chart
                    from config import SPECIALIST_MAPPING
                    if "specialist_active" in chunk:
                        spec = chunk["specialist_active"]
                        for label, m_id in SPECIALIST_MAPPING.items():
                            if m_id == spec:
                                st.session_state.specialist_counts[label] += 1
                                break
                    
                    # Phase 5: Capture re-rank score
                    if "top_relevance_score" in chunk:
                        st.session_state.last_relevance_score = chunk["top_relevance_score"]
                    
                    # 🚀 Architectural Overhaul: Capture background sentinel future
                    if "sentinel_future" in chunk and chunk["sentinel_future"]:
                        st.session_state.sentinel_future = chunk["sentinel_future"]

                # POST-STREAM USAGE FETCH
                # Store generation_id so the fetch can run after write_stream
                # returns, keeping it out of the hot streaming path entirely.
                if generation_id and not st.session_state.token_usage.get("input") and not st.session_state.model_id.startswith(OLLAMA_PREFIX):
                    st.session_state["_pending_generation_id"] = generation_id
            
            # Execute streaming
            answer = st.write_stream(response_generator())
            result = full_response

            # Post-stream usage fetch — runs after response is fully rendered
            # so the 1-second OpenRouter delay doesn't stall the displayed text.
            pending_gen_id = st.session_state.pop("_pending_generation_id", None)
            if pending_gen_id:
                fetched = _fetch_generation_usage(pending_gen_id)
                if fetched:
                    st.session_state.token_usage = fetched

            # Phase 5: Commit metrics to history
            usage = st.session_state.token_usage
            st.session_state.metrics_history.append({
                "turn": len(st.session_state.metrics_history) + 1,
                "input_tokens": usage.get("input", 0),
                "output_tokens": usage.get("output", 0),
                "cached_tokens": usage.get("cache_read", 0),
                "relevance_score": st.session_state.get("last_relevance_score", 0.0)
            })
            
        except Exception as e:
            st.error(f"❌ LLM error: {e}")
            st.stop()

        # Show source documents and Cache Stats
        source_docs = result.get("context", [])
        sources_meta = []
        
        # 🚀 Telemetry: Token Usage & Caching
        if st.session_state.get("token_usage"):
            usage = st.session_state.token_usage
            u_in = usage.get("input") or 0
            u_out = usage.get("output") or 0
            u_total = usage.get("total") or (u_in + u_out)
            c_read = usage.get("cache_read") or 0
            
            telemetry = f"📥 **{u_in}** In | 🤖 **{u_out}** AI | 📊 **{u_total}** Total"
            if c_read > 0:
                telemetry += f" | ⚡ **{c_read}** Cached"
            
            st.caption(telemetry)
            
            # Show Debug Info if enabled
            if st.session_state.debug_mode and st.session_state.get("debug_meta"):
                with st.expander("🐞 Debug Metadata"):
                    st.json(st.session_state.debug_meta)

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
        if st.session_state.debug_mode and source_docs:
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
