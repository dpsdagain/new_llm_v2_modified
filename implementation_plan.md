# Private AI Knowledge Base & Code Assistant — Implementation Plan

## Goal

Build a locally-hosted RAG web application that can analyze codebases and PDFs, using **100% free models** via OpenRouter, local HuggingFace embeddings, and ChromaDB — all runnable on 16GB RAM.

---

## Project Structure

```
f:\Gemini_anti\new_llm\
├── .env                    # API keys (gitignored)
├── requirements.txt        # Python dependencies
├── config.py               # Centralized configuration & env loading
├── backend.py              # Data ingestion: loaders, chunkers, embedding, ChromaDB
├── rag_chain.py            # Retriever + OpenRouter LLM + LangChain retrieval chain
├── app.py                  # Streamlit frontend (UI, chat, file upload)
├── chroma_db/              # Persistent ChromaDB storage (auto-created)
└── Private AI Architecture Document.md  # (existing) reference doc
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `.env` | Stores `OPENROUTER_API_KEY` securely |
| `requirements.txt` | All pip dependencies |
| `config.py` | Loads env vars, defines model names, chunk sizes, DB paths, embedding model config |
| `backend.py` | `load_and_chunk_pdf()`, `load_and_chunk_codebase()`, `ingest_into_chroma()` — the entire ingestion pipeline |
| `rag_chain.py` | `get_retriever()`, `get_llm()`, `build_rag_chain()` — the query pipeline |
| `app.py` | Streamlit UI: sidebar for upload/folder input, chat interface, session state management, response streaming |

---

## Proposed Changes

### 1. Environment & Dependencies

#### [NEW] `.env`
- Stores `OPENROUTER_API_KEY=sk-or-v1-...`

#### [NEW] `requirements.txt`
- Dependencies: `streamlit`, `langchain`, `langchain-community`, `langchain-chroma`, `langchain-openai`, `sentence-transformers`, `chromadb`, `pypdf`, `python-dotenv`

---

### 2. Configuration Module

#### [NEW] `config.py`
- Load `.env` via `python-dotenv`
- Constants: `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `FREE_MODEL`, `PAID_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHROMA_DB_DIR`, `RETRIEVER_K`
- Model selection helper for future upgrade hook

---

### 3. Backend (Ingestion Engine)

#### [NEW] `backend.py`

**PDF Loading:**
- Use `langchain_community.document_loaders.PyPDFLoader`
- Function: `load_and_chunk_pdf(file_path) -> list[Document]`

**Codebase Loading:**
- Use `langchain_community.document_loaders.DirectoryLoader` + `TextLoader`
- Support common code extensions: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`, `.go`, `.rs`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`, `.html`, `.css`
- Function: `load_and_chunk_codebase(directory_path) -> list[Document]`

**Chunking:**
- `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])`
- Language-aware separators for code files

**Embedding + ChromaDB:**
- `HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")`
- `Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")`
- Function: `ingest_into_chroma(documents, collection_name) -> Chroma`
- Function: `load_existing_chroma(collection_name) -> Chroma | None`

---

### 4. RAG Chain (Query Pipeline)

#### [NEW] `rag_chain.py`

**LLM Configuration:**
- Use `langchain_openai.ChatOpenAI` pointed at OpenRouter
- Default model: `meta-llama/llama-3-8b-instruct:free` (zero-cost)
- `temperature=0.0` for reproducibility

**Retriever:**
- `db.as_retriever(search_kwargs={"k": 4})`

**Chain:**
- Use `langchain.chains.create_retrieval_chain` with `create_stuff_documents_chain`
- System prompt template optimized for code analysis and document Q&A
- Returns `{"answer": ..., "context": ...}`

---

### 5. Streamlit Frontend

#### [NEW] `app.py`

**Sidebar:**
- PDF file uploader (`st.file_uploader`)
- Local folder path text input
- "Ingest" button to trigger ingestion pipeline
- Progress bar during ingestion
- Collection selector for existing ChromaDB collections

**Chat Interface:**
- `st.chat_message` for user/assistant messages
- `st.chat_input` for question entry
- Session state for chat history
- Source document display (expandable)

**Streaming:**
- Use `StreamingStdOutCallbackHandler` or Streamlit's native `st.write_stream`

---

## Improvements Over Base Document

1. **Language-aware chunking** — Use `RecursiveCharacterTextSplitter.from_language()` for code files to respect syntax boundaries
2. **Collection management** — Allow multiple ingestion sessions (different PDFs/codebases) as named collections
3. **Source attribution** — Show which chunks were retrieved alongside answers
4. **Temp file handling** — Proper cleanup of uploaded PDFs using `tempfile`
5. **Error handling** — Graceful handling of API rate limits, empty queries, missing API keys
6. **Caching** — `@st.cache_resource` for embedding model and ChromaDB to avoid reloading

---

## User Review Required

> [!IMPORTANT]
> The provided OpenRouter API key will be stored in a `.env` file in the project directory. Please confirm this is acceptable.

> [!WARNING]
> First-time embedding model download (`BAAI/bge-small-en-v1.5`, ~130MB) will take a few minutes on first run.

---

## Open Questions

1. **Virtual environment**: Should I create a new Python venv inside the project, or use the system Python? (I'll default to creating a venv for isolation.)
2. **Code file extensions**: The plan supports `.py, .js, .ts, .java, .cpp, .c, .h, .go, .rs, .md, .txt, .json, .yaml, .yml, .toml, .cfg, .ini, .html, .css`. Any additional extensions needed?

---

## Verification Plan

### Automated Tests
1. Run `pip install -r requirements.txt` — confirm clean install
2. Run `streamlit run app.py` — confirm UI loads without errors
3. Test PDF ingestion with a sample PDF
4. Test codebase ingestion by pointing at the project's own directory
5. Test chat query to confirm OpenRouter API responds

### Manual Verification
- Upload a PDF via the UI, ingest it, and ask questions about its content
- Point at a code directory, ingest it, and ask code-related questions
- Verify streaming responses appear in the chat interface
