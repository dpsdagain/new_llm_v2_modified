# **Architecture Document: Private AI Knowledge Base & Code Assistant**

## **🎯 1\. Project Goal**

To build a locally-hosted web application capable of analyzing massive codebases and lengthy PDFs using Retrieval-Augmented Generation (RAG).

**Key Strategy:** The system is designed for **Zero-Cost Prototyping**. It will initially utilize 100% free AI models via OpenRouter (e.g., Llama 3 or Gemma) to build, test, and debug the architecture without incurring API charges. Once the system is perfected, the user can seamlessly switch to premium flagship models (like Claude 4.6 or GPT-5.4) simply by changing one line of code.

## **🏗️ 2\. The Tech Stack**

This stack is chosen specifically to run efficiently on a standard 16GB RAM Windows machine with zero local infrastructure cost.

* **Frontend UI:** Streamlit (Fast, pure Python web interface; no HTML/CSS required).  
* **Orchestration Framework:** LangChain (The glue that connects your files, the database, and the AI).  
* **Vector Database:** ChromaDB (Runs entirely locally, file-based, highly efficient).  
* **Embeddings Model:** HuggingFaceBgeEmbeddings (Runs locally on your CPU/GPU to turn text into vectors for free, keeping data private and avoiding API costs during document ingestion).  
* **LLM Provider (Zero-Cost Setup):** OpenRouter API. We will configure the API to specifically target free models like meta-llama/llama-3-8b-instruct:free or google/gemma-2-9b-it:free to bypass all paywalls during development.

## **⚙️ 3\. System Architecture (Data Flow)**

The system is divided into two main pipelines: **Data Ingestion** (happens once per document/codebase) and **Querying** (happens every time you chat).

### **Pipeline A: Data Ingestion (Reading & Storing) \- *100% Free / Local***

1. **Upload/Point:** User uploads a PDF via the UI or points the app to a local code directory.  
2. **Load:** LangChain Document Loaders read the raw text.  
3. **Chunk:** RecursiveCharacterTextSplitter breaks the massive text into small blocks (e.g., 1000 tokens each with a 200-token overlap).  
4. **Embed:** The local HuggingFace model converts these text chunks into mathematical vectors.  
5. **Store:** The vectors and the original text chunks are saved into the local ChromaDB.

### **Pipeline B: Querying (Chatting with Context) \- *Powered by Free API***

1. **Prompt:** User asks a question in the Streamlit UI.  
2. **Embed Query:** The question is converted into a vector locally.  
3. **Retrieve:** ChromaDB performs a "similarity search" and instantly returns the top 4 most relevant chunks of code/text.  
4. **Construct Prompt:** LangChain bundles a System Prompt \+ the retrieved chunks \+ the user's question.  
5. **Generate:** This bundled prompt is sent to OpenRouter, specifically requesting a free model tier.  
6. **Response:** The LLM streams the answer back to the Streamlit UI.

## **🗺️ 4\. Step-by-Step Implementation Plan**

### **Step 1: Environment & API Setup**

* **Goal:** Set up the Python virtual environment and secure API keys.  
* **Tasks:**  
  * Ensure Microsoft C++ Build Tools are installed on Windows.  
  * Create requirements.txt (streamlit, langchain, langchain-chroma, sentence-transformers, openai, pypdf, python-dotenv).  
  * Set up a .env file to securely store OPENROUTER\_API\_KEY.

### **Step 2: Build the "Ingestion Engine" (backend.py)**

* **Goal:** Write functions to process PDFs and Code.  
* **Tasks:**  
  * Write load\_and\_chunk\_pdf(file\_path) and load\_and\_chunk\_codebase(directory\_path).  
  * Configure the text splitter parameters (chunk\_size=1000, chunk\_overlap=200).

### **Step 3: Initialize Local Vector Database**

* **Goal:** Set up ChromaDB and the local Embedding model.  
* **Tasks:**  
  * Initialize HuggingFaceBgeEmbeddings (model: BAAI/bge-small-en-v1.5).  
  * Write a function to ingest the chunks from Step 2 into a persistent ChromaDB directory (./chroma\_db).

### **Step 4: Build the Free RAG Retrieval Chain**

* **Goal:** Connect the database to the OpenRouter LLM using the free tier.  
* **Tasks:**  
  * Configure LangChain's ChatOpenAI wrapper to point to OpenRouter:  
    llm \= ChatOpenAI(  
        base\_url="\[https://openrouter.ai/api/v1\](https://openrouter.ai/api/v1)",  
        api\_key=os.getenv("OPENROUTER\_API\_KEY"),  
        model="meta-llama/llama-3-8b-instruct:free", \# Zero-cost model  
        temperature=0.0  
    )

  * Create a retriever from the ChromaDB instance (db.as\_retriever(search\_kwargs={"k": 4})).  
  * Build the LangChain create\_retrieval\_chain.

### **Step 5: Build the Streamlit Frontend (app.py)**

* **Goal:** Create the user interface.  
* **Tasks:**  
  * Create a sidebar for uploading PDFs or entering a local folder path.  
  * Create a chat interface (st.chat\_message) to display user questions and AI responses.  
  * Connect the chat input to the RAG Retrieval Chain from Step 4\.

## **🚀 5\. Future Upgrades (Phase 2\)**

1. **Premium Model Switch:** Once the system works perfectly, change the model string in Step 4 from ...:free to anthropic/claude-3.5-sonnet and load $5 into OpenRouter for production-grade coding.  
2. **Conversation Memory:** Add ConversationBufferMemory so the AI remembers follow-up questions.  
3. **Model Selector UI:** Add a dropdown in Streamlit to switch between Free and Paid models directly from the browser without changing the backend code.