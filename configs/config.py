import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

load_dotenv()

# =============================
# Envs
# =============================

GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GEN_AI_API_KEY = os.getenv("GEN_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =============================
# Paths
# =============================

DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db"

# =============================
# Embeddings
# =============================

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        api_key=GEN_AI_API_KEY
    )

# =============================
# Vector Store
# =============================

def get_vector_store():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )

# =============================
# LLM
# =============================

def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
    )