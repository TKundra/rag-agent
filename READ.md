# RAG Chatbot Project

A **Retrieval-Augmented Generation (RAG)** project that allows users to query their own `.txt` documents and get answers using **semantic search + LLM**.  

This project uses:

- **ChromaDB** for vector storage  
- **Google Gemini Embeddings** for embedding documents  
- **LLaMA 3.3 (via Groq)** for conversational AI  
- Python and LangChain ecosystem

---

## 🗂 Project Structure
```
project_root/
│
├── configs/
│ ├── init.py
│ └── config.py # Centralized config for embeddings, LLM, vector DB
│
├── pipelines/
│ ├── init.py
│ └── ingestion_pipeline.py # Load docs, split, generate embeddings, persist Chroma DB
│
├── rag_pipeline.py # Chat interface (history-aware + retrieval + generation)
├── docs/ # Your .txt documents to ingest
├── db/chroma_db/ # Persistent ChromaDB vector store
└── .env # API keys for Google Gemini / Groq
```

---

## ⚡ Features

- **Document Ingestion:** Reads all `.txt` files in `docs/`, splits them into semantic chunks, and stores them as embeddings in ChromaDB.  
- **RAG Chat:** Users can ask questions, including **follow-ups**, with conversation history support.  
- **History-aware Question Rewriting:** Converts follow-up questions into standalone questions for accurate retrieval.  
- **Embeddings:** Uses **Google Gemini** for high-quality embeddings.  
- **LLM Interaction:** Uses **LLaMA 3.3 via Groq** for generating answers.  
- **Persistent DB:** Vector store is persisted in `db/chroma_db/`, avoiding re-ingestion.

---

## 🛠 Requirements

- Python 3.10+  
- Install dependencies:

```bash
pip install -r requirements.txt
```
```bash
pip install langchain langchain-community langchain_text_splitters langchain_openai python_dotenv langchain_chroma langchain-google-genai langchain-groq
```

### .env
```
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=
GEN_AI_API_KEY=
```

### ingestion
```
python -m pipelines.ingestion_pipeline
```

### start chat
```
python rag_pipeline.py
```